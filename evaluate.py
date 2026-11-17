"""evaluate.py - runs the checker over data/testset.json and prints a confusion matrix."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import engine

CLASSES = ["ai", "mixed", "human"]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else engine.DATA_DIR / "testset.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    checker = engine.CodeChecker()
    by_variant = defaultdict(lambda: [0, 0])
    matrix = {e: {p: 0 for p in CLASSES} for e in CLASSES}
    wrong = []
    for c in cases:
        r = checker.check(c["code"])
        matrix[c["expected"]][r["verdict"]] += 1
        v = c.get("variant", c["expected"])
        by_variant[v][1] += 1
        by_variant[v][0] += r["verdict"] == c["expected"]
        if r["verdict"] != c["expected"]:
            wrong.append((c["id"], c["expected"], r["verdict"], r["score"]))
    n = len(cases)
    right = sum(matrix[k][k] for k in CLASSES)
    print(f"Accuracy: {right}/{n} = {right / n:.1%}\n")
    print("expected \\ predicted   " + "  ".join(f"{p:>6}" for p in CLASSES))
    for e in CLASSES:
        print(f"{e:>20}   " + "  ".join(f"{matrix[e][p]:>6}" for p in CLASSES))
    f1s = []
    for k in CLASSES:
        tp = matrix[k][k]
        fp = sum(matrix[o][k] for o in CLASSES if o != k)
        fn = sum(matrix[k][o] for o in CLASSES if o != k)
        p = tp / (tp + fp) if tp + fp else 0
        r_ = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r_ / (p + r_) if p + r_ else 0
        f1s.append(f1)
        print(f"{k:>6}: precision {p:.2f}  recall {r_:.2f}  F1 {f1:.2f}")
    print(f"macro-F1: {sum(f1s) / len(f1s):.3f}")
    if len(by_variant) > 3:
        print("\nBy variant (correct/total):")
        for v, (ok, tot) in by_variant.items():
            print(f"  {v:12} {ok}/{tot}")
    for w in wrong:
        print("miss:", w)


if __name__ == "__main__":
    main()
