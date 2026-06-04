# Peoples Daily NER

三种方法位置：

- `bert_crf`：BERT + 分类头 + CRF
- `llm_fewshot`：LLM few-shot
- `llm_sft`：仅占位

所有可调参数在 `src/config.py`。

BERT 默认会优先复用本机 ModelScope 缓存中的 `bert-base-chinese`；也可在 `.env` 中指定：

```bash
BERT_MODEL_PATH=C:\Users\17654\.cache\modelscope\hub\models\tiansz\bert-base-chinese
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python run.py analyze
python run.py bert_crf
python run.py llm_fewshot
python run.py llm_sft
python run.py compare
```

`compare` 会生成方法对比图和 BIO 顺序错误明细：

- `outputs/figures/compare_overall_metrics.png`
- `outputs/figures/compare_per_type_f1.png`
- `outputs/figures/compare_error_counts.png`
- `outputs/figures/compare_invalid_bio_orders.png`
- `outputs/reports/compare_invalid_bio_orders.json`

LLM 默认使用 DeepSeek Chat Completions 兼容接口，读取 `.env` 或环境变量：

```bash
deepseek_api_key=xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_WORKERS=0
DEEPSEEK_RPM=0
DEEPSEEK_RETRY_TIMES=5
DEEPSEEK_RETRY_STATUSES=429,500,502,503,504
```

也兼容 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`。

并发说明：DeepSeek 官方当前公开的是账号级并发连接数限制，不是 RPM。`DEEPSEEK_MAX_WORKERS=0` 表示按官方模型并发上限自动设置；实际线程数为 `min(样本数, 官方并发上限)`。`DEEPSEEK_RPM=0` 表示不做人为 RPM 限流。官方并发上限：`deepseek-v4-flash` / `deepseek-chat` / `deepseek-reasoner` 为 `2500`，`deepseek-v4-pro` 为 `500`。超过并发上限会返回 `HTTP 429`，程序会自动退避重试。官方文档：`https://api-docs.deepseek.com/quick_start/rate_limit`。

## 输出

- 图表：`outputs/figures/`
- 报告：`outputs/reports/`
- 预测：`outputs/predictions/`

## 评估口径

- 实体指标：实体级精确匹配，匹配键为 `(start, end, type)`，统计 precision / recall / f1。
- 错误顺序组合：只统计预测 BIO 序列中的非法顺序：
  - `START -> I-X`
  - `O -> I-X`
  - `B/I-X -> I-Y` 且 `X != Y`
- 不统计在 `invalid_bio_orders` 里的错误：合法 BIO 下的实体边界错、实体类型错、漏实体、多实体。
- LLM 专属：统计无法从模型输出中解析出约定 JSON 的样本数 `extraction_errors`。
