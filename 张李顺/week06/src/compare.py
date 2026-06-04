from .bio import INVALID_BIO_ORDER_CATEGORIES, summarize_invalid_bio_orders
from .data import dump_json, ensure_outputs, load_json


def _load_reports(config, names):
    reports = {}
    for name in names:
        path = config.output.reports / f"{name}.json"
        if path.exists():
            reports[name] = load_json(path)
    return reports


def _rows(reports):
    rows = []
    for name, report in reports.items():
        overall = report["overall"]
        rows.append({
            "method": name,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "extraction_errors": report.get("extraction_errors", 0),
            "api_errors": report.get("api_errors", 0),
            "invalid_llm_items": report.get("invalid_llm_items", 0),
            "invalid_bio_orders": sum(report.get("invalid_bio_order_combinations", {}).values()),
        })
    return rows


def _invalid_details(reports):
    details = {}
    for name, report in reports.items():
        combos = report.get("invalid_bio_order_combinations", {})
        details[name] = {
            "total": sum(combos.values()),
            "definition": report.get("invalid_bio_order_definition", ""),
            "categories": summarize_invalid_bio_orders(combos),
        }
    return details


def _plot_overall(rows, path):
    import matplotlib.pyplot as plt

    metrics = ["precision", "recall", "f1"]
    x = range(len(metrics))
    width = 0.8 / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        ax.bar([j + offset for j in x], [row[m] for m in metrics], width=width, label=row["method"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("overall entity metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_type(reports, path):
    import matplotlib.pyplot as plt

    types = sorted({t for r in reports.values() for t in r.get("per_type", {})})
    x = range(len(types))
    width = 0.8 / max(1, len(reports))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, report) in enumerate(reports.items()):
        offset = (i - (len(reports) - 1) / 2) * width
        ax.bar([j + offset for j in x], [report.get("per_type", {}).get(t, {}).get("f1", 0) for t in types], width=width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(types)
    ax.set_ylim(0, 1)
    ax.set_title("per-type entity F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_errors(rows, path):
    import matplotlib.pyplot as plt

    fields = ["invalid_bio_orders", "extraction_errors", "invalid_llm_items", "api_errors"]
    x = range(len(fields))
    width = 0.8 / max(1, len(rows))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, row in enumerate(rows):
        offset = (i - (len(rows) - 1) / 2) * width
        ax.bar([j + offset for j in x], [row.get(f, 0) for f in fields], width=width, label=row["method"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_title("error counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_invalid_bio(details, path):
    import matplotlib.pyplot as plt

    categories = list(INVALID_BIO_ORDER_CATEGORIES)
    x = range(len(categories))
    width = 0.8 / max(1, len(details))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, item) in enumerate(details.items()):
        offset = (i - (len(details) - 1) / 2) * width
        ax.bar([j + offset for j in x], [item["categories"].get(c, {}).get("total", 0) for c in categories], width=width, label=name)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_title("invalid BIO order categories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_table(rows):
    print("\n[overall]")
    print("method\tprecision\trecall\tf1\textraction_errors\tinvalid_llm_items\tapi_errors\tinvalid_bio_orders")
    for r in rows:
        print(f"{r['method']}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t{r['extraction_errors']}\t{r['invalid_llm_items']}\t{r['api_errors']}\t{r['invalid_bio_orders']}")


def _print_invalid_details(details):
    print("\n[invalid BIO order definition]")
    print("Only these predicted BIO order errors are counted:")
    for key, desc in INVALID_BIO_ORDER_CATEGORIES.items():
        print(f"- {key}: {desc}")
    print("Not counted here: entity boundary mismatch, entity type mismatch with legal BIO, missing/spurious entities, malformed labels outside the I-X predecessor rule.")

    for method, item in details.items():
        print(f"\n[invalid BIO orders: {method}] total={item['total']}")
        if item["total"] == 0:
            print("none")
            continue
        for category, info in item["categories"].items():
            if info["total"] == 0:
                continue
            print(f"  {category} total={info['total']} | {info['description']}")
            for combo, count in info["combinations"].items():
                print(f"    {combo}: {count}")


def compare_reports(config):
    ensure_outputs(config)
    names = ["bert_crf", "llm_fewshot"]
    reports = _load_reports(config, names)
    rows = _rows(reports)
    details = _invalid_details(reports)

    dump_json(rows, config.output.reports / "compare.json")
    dump_json(details, config.output.reports / "compare_invalid_bio_orders.json")
    if rows:
        _plot_overall(rows, config.output.figures / "compare_overall_metrics.png")
        _plot_per_type(reports, config.output.figures / "compare_per_type_f1.png")
        _plot_errors(rows, config.output.figures / "compare_error_counts.png")
        _plot_invalid_bio(details, config.output.figures / "compare_invalid_bio_orders.png")
    _print_table(rows)
    _print_invalid_details(details)
    print("\nsaved compare figures to", config.output.figures)
