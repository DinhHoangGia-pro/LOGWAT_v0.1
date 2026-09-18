"""Entrypoint to fine-tune the RoBERTa baseline (Reviewer #3)."""
from src.baselines.roberta import finetune_roberta


def main():
    result = finetune_roberta()
    print(result)


if __name__ == '__main__':
    main()
