from collections import Counter

from .bio import tags_to_entities
from .data import dump_json, ensure_outputs, load_split


def _token_lengths(samples, model_name):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
        return [len(tokenizer(s["tokens"], is_split_into_words=True)["input_ids"]) for s in samples]
    except Exception:
        return [len(s["tokens"]) for s in samples]


def run_analysis(config):
    import matplotlib.pyplot as plt

    ensure_outputs(config)
    splits = {name: load_split(config, name) for name in ["train", "validation", "test"]}
    char_lens = {k: [len(s["tokens"]) for s in v] for k, v in splits.items()}
    token_lens = {k: _token_lengths(v, config.bert_crf.model_name) for k, v in splits.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in splits:
        axes[0].hist(char_lens[name], bins=config.analyze.bins, alpha=0.45, label=name)
        axes[1].hist(token_lens[name], bins=config.analyze.bins, alpha=0.45, label=name)
    axes[0].set_title("char length distribution")
    axes[1].set_title("bert token length distribution")
    for ax in axes:
        ax.set_xlabel("length")
        ax.set_ylabel("samples")
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.output.figures / "length_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in splits:
        ratios = [sum(x > m for x in token_lens[name]) / len(token_lens[name]) for m in config.analyze.max_lengths]
        ax.plot(config.analyze.max_lengths, ratios, marker="o", label=name)
    ax.set_title("truncation ratio by max_length")
    ax.set_xlabel("max_length")
    ax.set_ylabel("ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.output.figures / "truncation_ratio.png", dpi=160)
    plt.close(fig)

    dist = {}
    for split in ["train", "test"]:
        c = Counter()
        for s in splits[split]:
            c.update(e["type"] for e in tags_to_entities(s["tokens"], s["ner_tags"]))
        dist[split] = c

    types = list(config.entity_types)
    x = range(len(types))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.18 for i in x], [dist["train"][t] for t in types], width=0.36, label="train")
    ax.bar([i + 0.18 for i in x], [dist["test"][t] for t in types], width=0.36, label="test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(types)
    ax.set_title("entity type distribution")
    ax.set_ylabel("entities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.output.figures / "entity_distribution_train_test.png", dpi=160)
    plt.close(fig)

    stats = {
        "length": {k: {"samples": len(v), "max_char_len": max(char_lens[k]), "max_token_len": max(token_lens[k])} for k, v in splits.items()},
        "entity_distribution": {k: dict(v) for k, v in dist.items()},
    }
    dump_json(stats, config.output.reports / "analysis_stats.json")
    print("saved figures to", config.output.figures)
