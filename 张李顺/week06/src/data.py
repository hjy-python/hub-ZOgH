import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_labels(config):
    return load_json(config.data.root / config.data.labels)


def load_split(config, split):
    name = getattr(config.data, split)
    return load_json(config.data.root / name)


def sample_text(sample):
    return "".join(sample["tokens"])


def ensure_outputs(config):
    config.output.figures.mkdir(parents=True, exist_ok=True)
    config.output.reports.mkdir(parents=True, exist_ok=True)
    config.output.predictions.mkdir(parents=True, exist_ok=True)
