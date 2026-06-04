from collections import Counter


INVALID_BIO_ORDER_DEFINITION = "Only invalid predecessor -> I-X transitions are counted."
INVALID_BIO_ORDER_CATEGORIES = {
    "START_TO_I": "Sequence starts with I-X: START -> I-X.",
    "O_TO_I": "Outside tag followed by I-X: O -> I-X.",
    "TYPE_TO_WRONG_I": "Entity type switches into I-Y without B-Y: B/I-X -> I-Y where X != Y.",
}


def tag_type(tag):
    return tag.split("-", 1)[1] if "-" in tag else ""


def tags_to_entities(tokens, tags):
    entities = []
    start, etype = None, None

    def close(end):
        if start is not None:
            entities.append({
                "start": start,
                "end": end,
                "type": etype,
                "text": "".join(tokens[start:end]),
            })

    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O" or "-" not in tag:
            close(i)
            start, etype = None, None
            continue

        prefix, cur_type = tag.split("-", 1)
        if prefix == "B":
            close(i)
            start, etype = i, cur_type
        elif prefix == "I":
            if etype != cur_type:
                close(i)
                start, etype = i, cur_type
        else:
            close(i)
            start, etype = None, None
    return entities


def entity_key(entity):
    return entity["start"], entity["end"], entity["type"]


def invalid_bio_orders(tags):
    counts = Counter()
    prev = "START"
    for tag in tags:
        if tag.startswith("I-"):
            cur_type = tag_type(tag)
            if prev in ("START", "O"):
                counts[f"{prev}->{tag}"] += 1
            elif tag_type(prev) != cur_type:
                counts[f"{prev}->{tag}"] += 1
        prev = tag
    return counts


def invalid_bio_order_category(order):
    prev, cur = order.split("->", 1)
    if prev == "START" and cur.startswith("I-"):
        return "START_TO_I"
    if prev == "O" and cur.startswith("I-"):
        return "O_TO_I"
    if cur.startswith("I-") and tag_type(prev) != tag_type(cur):
        return "TYPE_TO_WRONG_I"
    return "OTHER"


def summarize_invalid_bio_orders(combinations):
    summary = {
        key: {"description": desc, "total": 0, "combinations": {}}
        for key, desc in INVALID_BIO_ORDER_CATEGORIES.items()
    }
    for order, count in sorted(combinations.items(), key=lambda x: (-x[1], x[0])):
        category = invalid_bio_order_category(order)
        summary.setdefault(category, {"description": "Other invalid BIO transition.", "total": 0, "combinations": {}})
        summary[category]["total"] += count
        summary[category]["combinations"][order] = count
    return summary


def entities_to_tags(tokens, entities):
    tags = ["O"] * len(tokens)
    invalid_items = 0
    for e in sorted(entities, key=lambda x: (x["start"], x["end"])):
        start, end, etype = e["start"], e["end"], e["type"]
        if start < 0 or end <= start or end > len(tokens) or any(t != "O" for t in tags[start:end]):
            invalid_items += 1
            continue
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end):
            tags[i] = f"I-{etype}"
    return tags, invalid_items
