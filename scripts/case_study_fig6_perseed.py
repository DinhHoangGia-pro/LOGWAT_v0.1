"""Per-seed SQLi margins behind panel (b) of the new Fig. 6 (case_study_fig6.py prints edit margins as 5-seed means only).

Same payloads, checkpoints, edge edits and margin definition as scripts/case_study_fig6.py. Writes
data/case_study_fig6_perseed.json; does not touch data/case_study_fig6.txt.
"""
import json

from scripts.case_study_fig6 import PAYLOADS
from scripts.hgt_investigation_common import (DATA_DIR, SEEDS, apply_condition, load_model, logits_of, margin,
                                             uses_edge_attr)
from src.bag.graph_builder import build_single_graph

CONFIGS = (('gatv2', 'real'), ('gatv2', 'drop_sem'), ('gatv2', 'drop_seq'), ('hgt', 'real'), ('hgt', 'drop_sem'))


def main():
    res = {}
    for pname, text in PAYLOADS.items():
        g = build_single_graph(text, source_uid=pname, attack_type=1, use_edge_attr=True)
        for key, cond in CONFIGS:
            gg = apply_condition(g, cond)
            m = [round(margin(logits_of(load_model(key, s), [gg], uses_edge_attr(key)), 1).item(), 4) for s in SEEDS]
            res[f"{pname}|{key}|{cond}"] = m
            print(f"{pname:9s} {key:6s} {cond:9s} {[round(x, 2) for x in m]}  mean {sum(m) / len(m):+.2f}")
    (DATA_DIR / 'case_study_fig6_perseed.json').write_text(json.dumps(res, indent=1) + "\n")
    print("[+] wrote data/case_study_fig6_perseed.json")


if __name__ == '__main__':
    main()
