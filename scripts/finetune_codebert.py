"""Entrypoint to fine-tune the CodeBERT baseline (Reviewer #3)."""
from src.baselines.codebert import finetune_codebert


def main():
    result = finetune_codebert()
    print(result)


if __name__ == '__main__':
    main()
