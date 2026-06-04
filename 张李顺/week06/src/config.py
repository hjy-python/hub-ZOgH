import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


def load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def getenv_any(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def getenv_int(*names, default=0):
    value = getenv_any(*names, default=str(default))
    try:
        return int(value)
    except ValueError:
        return default


def getenv_float(*names, default=0.0):
    value = getenv_any(*names, default=str(default))
    try:
        return float(value)
    except ValueError:
        return default


def getenv_int_tuple(*names, default=()):
    value = getenv_any(*names, default="")
    if not value:
        return default
    try:
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError:
        return default


def is_transformers_model_dir(path):
    p = Path(path)
    has_config = (p / "config.json").exists()
    has_weight = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    has_tokenizer = (p / "vocab.txt").exists() or (p / "tokenizer.json").exists()
    return p.exists() and has_config and has_weight and has_tokenizer


def find_modelscope_model(model_name="bert-base-chinese"):
    override = getenv_any("BERT_MODEL_PATH", "BERT_CRF_MODEL_PATH", "bert_model_path")
    if override:
        return override

    roots = [
        Path(os.getenv("MODELSCOPE_CACHE", "")) if os.getenv("MODELSCOPE_CACHE") else None,
        Path(os.getenv("MODELSCOPE_HOME", "")) if os.getenv("MODELSCOPE_HOME") else None,
        Path.home() / ".cache" / "modelscope" / "hub" / "models",
    ]
    for root in [r for r in roots if r and r.exists()]:
        for path in list(root.glob(f"*/{model_name}")) + list(root.glob(model_name)):
            if is_transformers_model_dir(path):
                return str(path)
    return model_name


def default_llm_model():
    return getenv_any("LLM_MODEL", "DEEPSEEK_MODEL", "deepseek_model", default="deepseek-chat")


def deepseek_concurrency_limit(model):
    name = str(model).lower()
    if name in {"deepseek-v4-pro"}:
        return 500
    if name in {"deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash"}:
        return 2500
    return 2500


@dataclass
class DataConfig:
    root: Path = Path("peoples_daily")
    train: str = "train.json"
    validation: str = "validation.json"
    test: str = "test.json"
    labels: str = "label_names.json"


@dataclass
class OutputConfig:
    figures: Path = Path("outputs/figures")
    reports: Path = Path("outputs/reports")
    predictions: Path = Path("outputs/predictions")


@dataclass
class AnalyzeConfig:
    max_lengths: Tuple[int, ...] = (64, 128, 256, 512)
    bins: int = 50


@dataclass
class BertCrfConfig:
    model_name: str = field(default_factory=lambda: find_modelscope_model("bert-base-chinese"))
    output_dir: Path = Path("outputs/bert_crf")
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "auto"


@dataclass
class LlmConfig:
    base_url: str = getenv_any("LLM_BASE_URL", "DEEPSEEK_BASE_URL", "deepseek_base_url", default="https://api.deepseek.com")
    api_key: str = getenv_any("LLM_API_KEY", "DEEPSEEK_API_KEY", "deepseek_api_key")
    model: str = field(default_factory=default_llm_model)
    output_dir: Path = Path("outputs/llm_fewshot")
    few_shot_k: int = 3
    temperature: float = 0.0
    timeout: int = 60
    max_samples: Optional[int] = None
    use_response_format: bool = True
    max_workers: int = getenv_int("LLM_MAX_WORKERS", "DEEPSEEK_MAX_WORKERS", "deepseek_max_workers", default=0)
    requests_per_minute: int = getenv_int("LLM_RPM", "DEEPSEEK_RPM", "deepseek_rpm", default=0)
    retry_times: int = getenv_int("LLM_RETRY_TIMES", "DEEPSEEK_RETRY_TIMES", "deepseek_retry_times", default=5)
    retry_base_sleep: float = getenv_float("LLM_RETRY_BASE_SLEEP", "DEEPSEEK_RETRY_BASE_SLEEP", default=1.0)
    retry_max_sleep: float = getenv_float("LLM_RETRY_MAX_SLEEP", "DEEPSEEK_RETRY_MAX_SLEEP", default=30.0)
    retry_jitter: float = getenv_float("LLM_RETRY_JITTER", "DEEPSEEK_RETRY_JITTER", default=0.3)
    retry_statuses: Tuple[int, ...] = field(default_factory=lambda: getenv_int_tuple("LLM_RETRY_STATUSES", "DEEPSEEK_RETRY_STATUSES", default=(429, 500, 502, 503, 504)))
    api_error_max_chars: int = getenv_int("LLM_API_ERROR_MAX_CHARS", "DEEPSEEK_API_ERROR_MAX_CHARS", default=300)

    def __post_init__(self):
        if self.max_workers <= 0:
            self.max_workers = deepseek_concurrency_limit(self.model)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)
    bert_crf: BertCrfConfig = field(default_factory=BertCrfConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    entity_types: Tuple[str, ...] = ("PER", "ORG", "LOC")
    invalid_bio_order_definition: str = "START/O -> I-X, or B/I-X -> I-Y when X != Y"


CONFIG = Config()
