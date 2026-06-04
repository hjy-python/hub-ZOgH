import argparse

from src.config import CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["analyze", "bert_crf", "llm_fewshot", "llm_sft", "compare"])
    args = parser.parse_args()

    if args.cmd == "analyze":
        from src.analyze import run_analysis as run
    elif args.cmd == "bert_crf":
        from src.methods.bert_crf import run_bert_crf as run
    elif args.cmd == "llm_fewshot":
        from src.methods.llm_fewshot import run_llm_fewshot as run
    elif args.cmd == "llm_sft":
        from src.methods.llm_sft import run_llm_sft as run
    else:
        from src.compare import compare_reports as run
    run(CONFIG)


if __name__ == "__main__":
    main()
