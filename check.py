"""check.py - command-line interface.  python check.py file.py   |   cat file.py | python check.py -"""
from __future__ import annotations

import sys

import engine


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python check.py <file | ->")
        raise SystemExit(2)
    src = sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1], encoding="utf-8").read()
    r = engine.CodeChecker().check(src)
    print(f"{r['label']}  (AI-style score {r['score']}/100, {r['language']})")
    for f in r["findings"]:
        where = f"line {f['line']}" if f["line"] else "file"
        print(f"  [{where:>8}] {f['title']}: {f['snippet'] or f['message']}")
    print(f"{r['removed_comments']} redundant comment(s) removable. {r['disclaimer']}")


if __name__ == "__main__":
    main()
