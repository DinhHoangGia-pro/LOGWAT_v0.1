from pathlib import Path

import pandas as pd

from src.preprocessing.tokenizer import web_security_tokenizer
from src.edges.semantic import semantic_edges as semantic_edges_new, semantic_pairs

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
AUGMENTED_CSV = DATA_DIR / 'augmented_web_attack.csv'
OUT_TXT = DATA_DIR / 'semantic_fix_safety_check.txt'


def semantic_edges_old(tokens, window=15):
    """Literal copy of the pre-fix semantic_edges(): exact single-token match only."""
    edges = []
    n = len(tokens)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            if (tokens[i], tokens[j]) in semantic_pairs:
                edges.append((i, j))
                edges.append((j, i))
    return edges


def main():
    df = pd.read_csv(AUGMENTED_CSV)
    benign = df[df['attack_type'] == 0].copy()
    n_benign = len(benign)
    assert n_benign == 10000, f"expected 10000 benign rows, got {n_benign}"

    old_positive = 0
    new_positive = 0
    new_only_examples = []

    for _, row in benign.iterrows():
        content = str(row.get('content', ''))
        tokens = web_security_tokenizer(content)

        old_edges = semantic_edges_old(tokens, window=15)
        new_edges = semantic_edges_new(tokens, window=15)

        has_old = len(old_edges) > 0
        has_new = len(new_edges) > 0

        if has_old:
            old_positive += 1
        if has_new:
            new_positive += 1

        if has_new and not has_old and len(new_only_examples) < 10:
            new_only_examples.append({
                'source_uid': row.get('source_uid', ''),
                'content': content,
                'tokens': tokens,
                'new_edges': new_edges,
            })

    old_pct = 100.0 * old_positive / n_benign
    new_pct = 100.0 * new_positive / n_benign
    delta_pct = new_pct - old_pct

    lines = []
    lines.append("=" * 90)
    lines.append("BUOC 1: KIEM TRA AN TOAN semantic_edges() FIX TREN TOAN BO 10.000 DONG BENIGN")
    lines.append("=" * 90)
    lines.append(f"\nTong so dong Benign kiem tra: {n_benign}")
    lines.append(f"\n(1) TRUOC fix (khop token don le tuyet doi): "
                  f"{old_positive}/{n_benign} dong co E_sem>0 ({old_pct:.2f}%)")
    lines.append(f"(2) SAU fix (ghep token qua noise char): "
                  f"{new_positive}/{n_benign} dong co E_sem>0 ({new_pct:.2f}%)")
    lines.append(f"\nChenh lech (2) - (1) = {delta_pct:+.2f} diem %")

    threshold = 5.0
    triggered = delta_pct > threshold
    lines.append(f"\nDieu kien dung (tang > {threshold}%): {'CO - DUNG LAI' if triggered else 'KHONG - AN TOAN, tiep tuc BUOC 2'}")

    if triggered:
        lines.append(f"\n--- {len(new_only_examples)} vi du benign bi sinh canh E_sem GIA sau fix (E_sem=0 truoc, >0 sau) ---")
        for ex in new_only_examples:
            lines.append(f"\n[{ex['source_uid']}]")
            lines.append(f"  content={ex['content']!r}")
            lines.append(f"  tokens={ex['tokens']}")
            lines.append(f"  new_edges={ex['new_edges']}")
    else:
        lines.append(f"\n(Khong can liet ke vi du vi khong vuot nguong dung.)")
        if new_only_examples:
            lines.append(f"Luu y: van co {len(new_only_examples)} vi du (trong so {n_benign}) sinh canh E_sem moi "
                          f"du khong vuot nguong 5% - liet ke de tham khao:")
            for ex in new_only_examples:
                lines.append(f"\n[{ex['source_uid']}]")
                lines.append(f"  content={ex['content']!r}")
                lines.append(f"  new_edges={ex['new_edges']}")

    output = "\n".join(lines)
    print(output)
    with open(OUT_TXT, 'w') as f:
        f.write(output + "\n")

    print(f"\nDa ghi: {OUT_TXT}")
    print(f"\nSTOP_CONDITION_TRIGGERED={triggered}")


if __name__ == '__main__':
    main()
