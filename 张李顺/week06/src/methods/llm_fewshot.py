import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from ..bio import entities_to_tags, tags_to_entities
from ..data import dump_json, ensure_outputs, load_split, sample_text
from ..metrics import evaluate_tag_sequences
from ..rate_limit import RateLimiter, retry_call
from ..config import deepseek_concurrency_limit


TYPE_ALIASES = {
    "PER": "PER", "PERSON": "PER", "人名": "PER", "人物": "PER",
    "ORG": "ORG", "ORGANIZATION": "ORG", "机构": "ORG", "组织": "ORG",
    "LOC": "LOC", "LOCATION": "LOC", "地点": "LOC", "地名": "LOC",
}


def _entity_json(sample):
    return {"entities": tags_to_entities(sample["tokens"], sample["ner_tags"])}


def _prompt(train, sample, k):
    examples = []
    for ex in train[:k]:
        examples.append("文本：" + sample_text(ex) + "\n输出：" + json.dumps(_entity_json(ex), ensure_ascii=False))
    return [
        {"role": "system", "content": "抽取中文命名实体。类型只允许 PER、ORG、LOC。只输出 JSON：{\"entities\":[{\"text\":\"\",\"type\":\"PER\",\"start\":0,\"end\":1}]}。start/end 为字符下标，end 不包含。"},
        {"role": "user", "content": "\n\n".join(examples) + "\n\n文本：" + sample_text(sample) + "\n输出："},
    ]


def _call(config, messages, limiter=None):
    if limiter:
        limiter.wait()
    payload = {"model": config.llm.model, "messages": messages, "temperature": config.llm.temperature}
    if config.llm.use_response_format:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(
        config.llm.base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {config.llm.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=config.llm.timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_retry(config, messages, limiter):
    return retry_call(
        lambda: _call(config, messages, limiter),
        retries=config.llm.retry_times,
        base_sleep=config.llm.retry_base_sleep,
        max_sleep=config.llm.retry_max_sleep,
        retry_statuses=config.llm.retry_statuses,
        jitter=config.llm.retry_jitter,
    )


def _loads(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if match:
            return json.loads(match.group(1))
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _normalize(obj, text, n):
    raw = obj.get("entities") if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        raise ValueError("missing entities list")

    entities, invalid, used = [], 0, set()
    for item in raw:
        if not isinstance(item, dict):
            invalid += 1
            continue
        raw_type = str(item.get("type", ""))
        etype = TYPE_ALIASES.get(raw_type.upper(), TYPE_ALIASES.get(raw_type, ""))
        ent_text = str(item.get("text", ""))
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or text[start:end] != ent_text:
            pos = text.find(ent_text) if ent_text else -1
            while pos >= 0 and any(i in used for i in range(pos, pos + len(ent_text))):
                pos = text.find(ent_text, pos + 1)
            start, end = pos, pos + len(ent_text) if pos >= 0 else -1
        if etype not in ("PER", "ORG", "LOC") or start < 0 or end <= start or end > n:
            invalid += 1
            continue
        used.update(range(start, end))
        entities.append({"start": start, "end": end, "type": etype, "text": text[start:end]})
    return entities, invalid


def _predict_one(idx, sample, train, config, limiter):
    text = sample_text(sample)
    raw, entities, bad = "", [], 0
    extraction_error, api_error = 0, ""
    try:
        raw = _call_retry(config, _prompt(train, sample, config.llm.few_shot_k), limiter)
        try:
            entities, bad = _normalize(_loads(raw), text, len(sample["tokens"]))
        except Exception:
            extraction_error = 1
    except Exception as e:
        api_error = f"{type(e).__name__}: {str(e)[:config.llm.api_error_max_chars]}"

    tags, overlap_bad = entities_to_tags(sample["tokens"], entities)
    return {
        "idx": idx,
        "tags": tags,
        "raw_record": {"text": text, "raw": raw, "entities": entities, "api_error": api_error},
        "extraction_error": extraction_error,
        "invalid_items": bad + overlap_bad,
        "api_error": 1 if api_error else 0,
    }


def run_llm_fewshot(config):
    if not config.llm.api_key:
        raise RuntimeError("set deepseek_api_key or DEEPSEEK_API_KEY in .env first")
    ensure_outputs(config)
    train = load_split(config, "train")
    test = load_split(config, "test")
    if config.llm.max_samples:
        test = test[:config.llm.max_samples]

    n = len(test)
    pred_tags, raw_records = [None] * n, [None] * n
    extraction_errors, invalid_items, api_errors = 0, 0, 0
    limiter = RateLimiter(config.llm.requests_per_minute)
    workers = max(1, min(n or 1, config.llm.max_workers))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_predict_one, i, sample, train, config, limiter) for i, sample in enumerate(test)]
        for future in tqdm(as_completed(futures), total=n, desc=f"llm_fewshot/{workers}w/{config.llm.requests_per_minute}rpm"):
            item = future.result()
            i = item["idx"]
            pred_tags[i] = item["tags"]
            raw_records[i] = item["raw_record"]
            extraction_errors += item["extraction_error"]
            invalid_items += item["invalid_items"]
            api_errors += item["api_error"]

    report = evaluate_tag_sequences(
        test,
        pred_tags,
        "llm_fewshot",
        extraction_errors=extraction_errors,
        invalid_llm_items=invalid_items,
        order_definition=config.invalid_bio_order_definition,
    )
    report["api_errors"] = api_errors
    report["concurrency"] = {
        "max_workers": workers,
        "official_model_concurrency_limit": deepseek_concurrency_limit(config.llm.model),
        "requests_per_minute": config.llm.requests_per_minute,
        "retry_times": config.llm.retry_times,
        "retry_statuses": list(config.llm.retry_statuses),
    }
    dump_json({k: v for k, v in report.items() if k != "records"}, config.output.reports / "llm_fewshot.json")
    dump_json(raw_records, config.output.predictions / "llm_fewshot_raw.json")
    dump_json(report["records"], config.output.predictions / "llm_fewshot_test.json")
    print("llm_fewshot f1", report["overall"]["f1"], "extraction_errors", extraction_errors, "api_errors", api_errors)
