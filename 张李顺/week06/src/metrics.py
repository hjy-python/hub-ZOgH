from collections import Counter, defaultdict

from .bio import entity_key, invalid_bio_orders, summarize_invalid_bio_orders, tags_to_entities


def prf(tp, pred, gold):
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred": pred, "gold": gold}


def evaluate_tag_sequences(samples, pred_tags, method, extraction_errors=0, invalid_llm_items=0, order_definition=""):
    total = Counter()
    by_type = defaultdict(Counter)
    order_counts = Counter()

    records = []
    for i, (sample, pred) in enumerate(zip(samples, pred_tags)):
        tokens = sample["tokens"]
        gold = sample["ner_tags"]
        pred = pred[:len(tokens)] + ["O"] * max(0, len(tokens) - len(pred))
        gold_entities = tags_to_entities(tokens, gold)
        pred_entities = tags_to_entities(tokens, pred)
        gold_set = {entity_key(e) for e in gold_entities}
        pred_set = {entity_key(e) for e in pred_entities}
        hit_set = gold_set & pred_set

        total["tp"] += len(hit_set)
        total["pred"] += len(pred_set)
        total["gold"] += len(gold_set)
        order_counts.update(invalid_bio_orders(pred))

        for key in gold_set:
            by_type[key[2]]["gold"] += 1
        for key in pred_set:
            by_type[key[2]]["pred"] += 1
        for key in hit_set:
            by_type[key[2]]["tp"] += 1

        records.append({
            "id": i,
            "text": "".join(tokens),
            "gold": gold_entities,
            "pred": pred_entities,
        })

    return {
        "method": method,
        "overall": prf(total["tp"], total["pred"], total["gold"]),
        "per_type": {k: prf(v["tp"], v["pred"], v["gold"]) for k, v in sorted(by_type.items())},
        "invalid_bio_order_definition": order_definition,
        "invalid_bio_order_combinations": dict(order_counts),
        "invalid_bio_order_summary": summarize_invalid_bio_orders(dict(order_counts)),
        "extraction_errors": extraction_errors,
        "invalid_llm_items": invalid_llm_items,
        "records": records,
    }
