"""
engine.py - the AI Code Checker engine (standard library only).

Looks for the stylistic fingerprints that AI assistants leave in source code: narrating comments,
numbered step comments, template docstrings, assistant-style phrasing, placeholders, generic names,
catch-and-print error handling and so on. It returns a 0-100 "AI-style score", a label,
line-level findings, and a cleaned copy with the redundant comments removed.

IMPORTANT: this measures *style*, not authorship. Humans (especially beginners and people taught
to comment every line) write like this too, and AI output can be edited until it looks human.
"""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MAX_CHARS = 30000

PATTERNS = json.loads((DATA_DIR / "patterns.json").read_text(encoding="utf-8"))

# rule id -> (title, category, weight, description)
RULES = {
    "narration": ("Narrating comment", "comment", 2.5,
                  "Comment restates the line below it instead of explaining why."),
    "step_comment": ("Numbered step comment", "comment", 4.0,
                     "'Step 1:', '1.' style comments walk through the code like a tutorial."),
    "banner": ("Section banner", "comment", 2.0,
               "Decorative divider comments such as '# ===== Helpers ====='."),
    "ai_phrase_strong": ("Assistant-style phrasing", "comment", 2.5,
                         "Phrases like \"Here's\", \"Feel free to\", \"Example usage\"."),
    "ai_phrase_soft": ("Tutorial-style phrasing", "comment", 1.2,
                       "Phrases like \"Note that\", \"ensure that\", \"handle edge cases\", \"This function\"."),
    "placeholder_comment": ("Placeholder / elision comment", "comment", 5.0,
                            "'rest of the code', 'your logic here', 'implement this' left in the file."),
    "comment_density": ("Over-commented file", "comment", 3.0,
                        "A large share of the file is comment lines."),
    "polished_comments": ("Uniformly polished comments", "comment", 2.5,
                          "Nearly every comment is a capitalised, neatly worded sentence."),
    "inline_overload": ("Many end-of-line comments", "comment", 2.0,
                        "Lots of code lines carry a trailing comment."),
    "emoji": ("Emoji in code", "style", 3.0, "Emoji in source, comments or output strings."),
    "placeholder_value": ("Placeholder value", "code", 3.0,
                          "Strings like YOUR_API_KEY, path/to/file, example.com, lorem ipsum."),
    "generic_func_name": ("Generic function name", "code", 2.0,
                          "Names such as process_data, helper_function, my_function."),
    "generic_vars": ("Generic variable names", "code", 2.0,
                     "Repeated use of data / result / temp / output as variable names."),
    "broad_except": ("Catch-and-print error handling", "code", 2.0,
                     "Broad except/catch that only prints or swallows the error."),
    "immediate_return": ("Variable assigned then returned", "code", 1.5,
                         "'result = expr' immediately followed by 'return result'."),
    "redundant_else": ("Else after return", "code", 1.5,
                       "An else branch directly after a return or throw."),
    "bool_compare": ("Redundant boolean comparison", "code", 1.5,
                     "'== True', '== False', '== None' style comparisons."),
    "status_print": ("Status print statement", "code", 1.5,
                     "print('Processing...'), console.log('Done') style messages."),
    "defensive_check": ("Boilerplate input validation", "code", 1.2,
                        "isinstance/typeof guard that only raises a TypeError."),
    "docstring_boilerplate": ("Template docstring", "doc", 2.0,
                              "Args/Returns docstring wrapped around a tiny function."),
    "docstring_generic": ("'This function...' docstring", "doc", 2.0,
                          "Docstring that opens with 'This function', 'Helper function to...'."),
    "uniform_docs": ("Every function documented and typed", "doc", 3.0,
                     "All functions have a docstring and complete type hints."),
    "jsdoc_boilerplate": ("Template JSDoc", "doc", 1.5,
                          "JSDoc block with @param and @returns on every function."),
}

ACTION_VERBS = set("""initialize initialise init create define declare set get check loop iterate return import print call
add increment decrement calculate compute open close update store save convert handle process read write load parse
validate assign append remove delete display show send fetch retrieve build make generate instantiate start end sort
filter map find count sum reverse swap test run use apply perform extract""".split())
STOP = set("the a an of to in for and if is it this that with on by as from each all are be we then now".split())
DIRECTIVES = ("!", "-*-", "type:", "noqa", "pylint", "pragma", "eslint", "istanbul", "prettier", "ts-ignore",
              "ts-nocheck", "@ts-", "jshint", "flake8", "fmt:", "isort", "mypy")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF☀-➿⭐⬆⬇]")
PLACEHOLDER_RES = [re.compile(p, re.I) for p in PATTERNS["placeholder_regex"]]
PLACEHOLDER_VALUE = re.compile(PATTERNS["placeholder_value_regex"], re.I)
GENERIC_FUNC = re.compile(
    r"^(process|handle|perform|do|run|execute)(data|task|request|input|stuff|logic|operation|action)$"
    r"|^helper(function)?\d*$|^(my|example|sample)(function|func|method)\d*$|^mainfunction$"
    r"|^calculateresult$|^process$|^func\d*$|^handler$|^dosomething$")
GENERIC_VARS = {"data", "result", "results", "temp", "tmp", "output", "res", "val", "value", "obj", "var1", "var2"}
STATUS_PRINT = re.compile(
    r"""\b(?:print|console\.log|System\.out\.println|printf|fmt\.Println)\s*\(\s*f?["'`]\s*(?:starting|processing|done|"""
    r"""finished|completed?|successfully|loading|initiali[sz]ing|running|error\b|\w+ (?:successfully|completed))""",
    re.I)
BOOL_COMPARE = re.compile(r"[=!]=\s*(?:True|False)\b|[=!]==?\s*(?:true|false)\b|[=!]=\s*None\b")

FUNC_DEF_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(\w+)"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"),
    re.compile(r"^\s*(?:(?:public|private|protected|static|final)\s+)+[\w<>\[\],]+\s+(\w+)\s*\("),
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\("),
]


# --------------------------------------------------------------------------- preprocessing
def split_words(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[A-Za-z]+", text)


def stem(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def detect_language(source: str) -> str:
    py = js = 0
    for line in source.split("\n"):
        if re.match(r"\s*(def |class \w+.*:\s*$|import \w+\s*$|from \S+ import |elif |if __name__|async def )", line):
            py += 2
        if re.match(r"\s*#(?!include|define|pragma|if|endif|else|ifdef|ifndef)", line):
            py += 1
        if re.match(r"\s*((try|else|finally):|except\b.*:|(for|while|if|elif|with)\b.*:)\s*(#.*)?$", line) \
                or re.search(r"\bprint\(|\bself\.|\bNone\b|\bTrue\b|\bFalse\b", line) and not line.rstrip().endswith((";", "{")):
            py += 1
        if re.search(r"\s#\s", line) and not line.rstrip().endswith((";", "{")):
            py += 1
        if re.search(r"\b(function|const|let|var)\b|=>|;\s*$|\{\s*$|^\s*//|System\.out|public\s+(static|class)|#include|\bfmt\.", line):
            js += 1
    return "python" if py >= js and py > 0 else "c-style"


def scan(source: str, lang: str):
    """Split source into comments and code-only lines (strings kept, comments removed)."""
    lines = source.split("\n")
    comments: list[dict] = []
    code_only: list[str] = []
    in_str: str | None = None
    block: dict | None = None
    for idx, line in enumerate(lines):
        lineno = idx + 1
        j = 0
        code: list[str] = []
        while j < len(line):
            if block is not None:
                end = line.find("*/", j)
                piece = line[j:] if end == -1 else line[j:end]
                block["pieces"].append(piece.strip().lstrip("*").strip() if block["pieces"] else piece.strip())
                if end == -1:
                    j = len(line)
                else:
                    block["end"] = lineno
                    comments.append(block)
                    block = None
                    j = end + 2
                continue
            if in_str is not None:
                k = j
                found = False
                while k < len(line):
                    if line[k] == "\\":
                        k += 2
                        continue
                    if line.startswith(in_str, k):
                        k += len(in_str)
                        found = True
                        break
                    k += 1
                k = min(k, len(line))
                code.append(line[j:k])
                j = k
                if found or (len(in_str) == 1 and in_str != "`"):
                    in_str = None
                continue
            ch = line[j]
            if lang == "python":
                if ch == "#":
                    comments.append({"line": lineno, "end": lineno, "col": j, "text": line[j + 1:].strip(),
                                     "inline": bool("".join(code).strip()), "block": False, "doc": False})
                    break
                if line.startswith('"""', j) or line.startswith("'''", j):
                    in_str = line[j:j + 3]
                    code.append(in_str)
                    j += 3
                    continue
                if ch in "'\"":
                    in_str = ch
                    code.append(ch)
                    j += 1
                    continue
            else:
                if line.startswith("//", j):
                    comments.append({"line": lineno, "end": lineno, "col": j, "text": line[j + 2:].lstrip("/").strip(),
                                     "inline": bool("".join(code).strip()), "block": False, "doc": False})
                    break
                if line.startswith("/*", j):
                    block = {"line": lineno, "col": j, "pieces": [], "inline": bool("".join(code).strip()),
                             "block": True, "doc": line.startswith("/**", j)}
                    j += 2
                    continue
                if ch in "'\"`":
                    in_str = ch
                    code.append(ch)
                    j += 1
                    continue
            code.append(ch)
            j += 1
        code_only.append("".join(code))
    for c in comments:
        if c["block"]:
            c["text"] = "\n".join(p for p in c.pop("pieces") if p).strip()
            c["end"] = c.get("end", c["line"])
    comments.sort(key=lambda c: (c["line"], c["col"]))
    return lines, comments, code_only


def python_info(source: str):
    """Docstring line ranges and function nodes via ast (empty if the code does not parse)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set(), [], False
    doc_lines: set[int] = set()
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node)
    return doc_lines, funcs, True


# --------------------------------------------------------------------------- comment rules
def is_narration(text: str, code: str) -> bool:
    t = text.strip().rstrip(".:")
    if not t or not code.strip():
        return False
    low = t.lower()
    if any(m in low for m in PATTERNS["explain_markers"]) or re.search(r"https?://|\d{2,}", t):
        return False
    raw = re.findall(r"[A-Za-z]+", t)
    if not raw or len(raw) > 9:
        return False
    first = raw[0].lower()
    content = [stem(w) for w in raw if w.lower() not in STOP]
    code_tokens = {stem(w) for w in split_words(code)}
    if first in ACTION_VERBS:
        rest = content[1:]
        overlap = sum(1 for w in rest if w in code_tokens) / len(rest) if rest else 1.0
        return overlap >= 0.5 or len(content) <= 2
    overlap = sum(1 for w in content if w in code_tokens) / len(content) if content else 0.0
    return len(content) >= 2 and overlap >= 0.8


def match_phrase(text: str):
    low = text.lower()
    for p in PATTERNS["strong_phrases"]:
        if p in low:
            return "ai_phrase_strong", p
    for p in PATTERNS["soft_phrases"]:
        if p in low:
            return "ai_phrase_soft", p
    return None, None


# --------------------------------------------------------------------------- the checker
class CodeChecker:
    def info(self) -> dict:
        return {"rules": len(RULES), "languages": ["python", "c-style (JS/TS/Java/C/C++/C#/Go)"]}

    def list_rules(self) -> list[dict]:
        return [{"id": k, "title": v[0], "category": v[1], "weight": v[2], "description": v[3]} for k, v in RULES.items()]

    def check(self, source: str, language: str = "auto") -> dict:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Paste some code to check.")
        source = source.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        if len(source) > MAX_CHARS:
            raise ValueError(f"Code is too long ({len(source):,} characters). Limit is {MAX_CHARS:,}.")
        lang = language if language in ("python", "c-style") else detect_language(source)
        lines, comments, code_only = scan(source, lang)
        doc_lines, funcs, parsed = python_info(source) if lang == "python" else (set(), [], False)

        findings: list[dict] = []
        seen: set[tuple] = set()

        def add(rule, line, message=None, fix=None, weight=None, end=None):
            key = (rule, line)
            if key in seen:
                return
            seen.add(key)
            title, cat, w, desc = RULES[rule]
            findings.append({
                "rule": rule, "title": title, "category": cat,
                "weight": weight if weight is not None else w,
                "line": line, "end": end or line,
                "snippet": lines[line - 1].strip()[:140] if line else "",
                "message": message or desc, "fix": fix or "",
            })

        # --- line classification
        comment_only: set[int] = set()
        for c in comments:
            code_part = code_only[c["line"] - 1][: c["col"]].strip() if not c["block"] else code_only[c["line"] - 1].strip()
            if not c["inline"] and not code_part:
                comment_only.update(range(c["line"], c["end"] + 1))
        code_lines = [i + 1 for i, l in enumerate(code_only)
                      if l.strip() and (i + 1) not in comment_only and (i + 1) not in doc_lines]
        loc = len(code_lines)
        code_set = set(code_lines)

        def next_code(after: int) -> str:
            for n in range(after + 1, min(len(code_only), after + 4) + 1):
                if n in code_set:
                    return code_only[n - 1]
            return ""

        # --- comment rules
        removable: dict[int, str] = {}
        human_signals = 0
        plain_comments = []
        for c in comments:
            text = c["text"]
            if not text or text.lower().startswith(DIRECTIVES):
                continue
            first_line = text.split("\n")[0].strip()
            low = text.lower()
            if any(m in low for m in PATTERNS["explain_markers"]) or re.search(r"\b(todo|fixme)\b", low):
                human_signals += 1
            if c["block"] and c["doc"]:
                if "@param" in text and "@return" in text:
                    add("jsdoc_boilerplate", c["line"], end=c["end"],
                        fix="Keep JSDoc only where types or intent are not obvious from the signature.")
                continue
            is_block_multi = c["block"] and c["end"] > c["line"]
            hit = False
            if any(r.search(text) for r in PLACEHOLDER_RES):
                add("placeholder_comment", c["line"], fix="Finish or remove this placeholder before submitting.", end=c["end"])
                hit = True
            if not c["inline"] and not hit:
                if re.search(r"[=\-#*_~]{3,}", first_line) and not is_block_multi:
                    add("banner", c["line"], fix="Delete the divider; use blank lines or modules instead.")
                    removable[c["line"]] = "line"
                    hit = True
            if not hit and re.match(r"^(step\s*\d+\b|phase\s*\d+\b|\d+\s*[.)]\s+\S)", first_line, re.I):
                add("step_comment", c["line"], fix="Drop the numbering; keep a comment only if it explains why.")
                removable[c["line"]] = "inline" if c["inline"] else "line"
                hit = True
            if not hit:
                rule, phrase = match_phrase(text)
                if rule:
                    add(rule, c["line"], message=f"Contains '{phrase}', typical of assistant-written explanations.",
                        fix="Rewrite in your own words or remove it.", end=c["end"])
                    hit = True
            if not hit and not is_block_multi:
                ctx = code_only[c["line"] - 1][: c["col"]] if c["inline"] else next_code(c["end"])
                if is_narration(first_line, ctx):
                    add("narration", c["line"], fix="Delete it. The code already says this; comment the reason instead.")
                    removable[c["line"]] = "inline" if c["inline"] else "line"
                    hit = True
            if not hit:
                plain_comments.append(c)

        # --- file-level comment statistics
        comment_line_count = len(comment_only)
        density = comment_line_count / (comment_line_count + loc) if (comment_line_count + loc) else 0.0
        if comment_line_count >= 5 and density > 0.30:
            heavy = density > 0.50
            add("comment_density", None, weight=5.0 if heavy else 3.0,
                message=f"{density:.0%} of non-blank lines are comments ({comment_line_count} comment lines vs {loc} code lines).")
        sentence_comments = [c for c in comments if c["text"] and not c["doc"] and not c["text"].lower().startswith(DIRECTIVES)]
        if len(sentence_comments) >= 4:
            caps = sum(1 for c in sentence_comments if c["text"][0].isupper())
            if caps / len(sentence_comments) >= 0.85:
                add("polished_comments", None,
                    message=f"{caps} of {len(sentence_comments)} comments start with a capital letter and read as tidy sentences.")
        inline_count = sum(1 for c in sentence_comments if c["inline"])
        if inline_count >= 4 and loc and inline_count / loc >= 0.2:
            add("inline_overload", None, message=f"{inline_count} of {loc} code lines carry a trailing comment.")

        # --- raw-line rules (emoji anywhere, placeholders in code)
        for i, raw in enumerate(lines, start=1):
            if EMOJI.search(raw):
                add("emoji", i, fix="Remove the emoji.")
            if i in code_set and PLACEHOLDER_VALUE.search(code_only[i - 1]):
                add("placeholder_value", i, fix="Replace with a real value or load it from configuration.")

        # --- code rules
        generic_lines = []
        prev_code = None
        for n in code_lines:
            line = code_only[n - 1]
            s = line.strip()
            for pat in FUNC_DEF_PATTERNS:
                m = pat.match(line)
                if m:
                    if GENERIC_FUNC.match(m.group(1).replace("_", "").lower()):
                        add("generic_func_name", n, message=f"'{m.group(1)}' says nothing about what it does.",
                            fix="Name it after the specific job it performs.")
                    break
            m = re.match(r"^\s*(?:const\s+|let\s+|var\s+)?(\w+)\s*(?::\s*[\w\[\], .]+)?\s*=(?!=)", line)
            if m and m.group(1) in GENERIC_VARS:
                generic_lines.append((n, m.group(1)))
            if BOOL_COMPARE.search(line):
                add("bool_compare", n, fix="Use the value directly (if x / if not x / is None).")
            if STATUS_PRINT.search(line):
                add("status_print", n, fix="Remove it, or use proper logging.")
            # variable assigned then returned
            if prev_code is not None:
                pm = re.match(r"^\s*(?:(?:const|let|var)\s+)?(\w+)\s*=(?!=)\s*(\S.*?);?\s*$", code_only[prev_code - 1])
                if pm and s.rstrip(";") == f"return {pm.group(1)}" and not code_only[prev_code - 1].strip().endswith(("(", "[", "{", ",")):
                    add("immediate_return", prev_code, end=n, fix=f"Return the expression directly.")
            prev_code = n
            # else after return
            if lang == "python":
                if re.match(r"^\s*else\s*:\s*$", line):
                    indent = len(line) - len(line.lstrip())
                    before = [k for k in code_lines if k < n]
                    if before:
                        last = code_only[before[-1] - 1]
                        owner = next((code_only[k - 1] for k in reversed(before)
                                      if len(code_only[k - 1]) - len(code_only[k - 1].lstrip()) == indent), "")
                        if re.match(r"^\s*(return|raise)\b", last) and re.match(r"^\s*(if|elif)\b", owner) \
                                and len(last) - len(last.lstrip()) > indent:
                            add("redundant_else", n, fix="Remove the else and dedent its body.")
            elif re.match(r"^\s*(\}\s*)?else\s*\{", line):
                before = [k for k in code_lines if k < n]
                if before and re.match(r"^\s*(return|throw)\b", code_only[before[-1] - 1]):
                    add("redundant_else", n, fix="Remove the else and dedent its body.")

        if len(generic_lines) >= 3:
            names = ", ".join(sorted({g for _, g in generic_lines}))
            add("generic_vars", generic_lines[0][0], message=f"{len(generic_lines)} assignments use generic names ({names}).",
                fix="Name variables after what they hold.")

        for n in code_lines:
            line = code_only[n - 1]
            if lang == "python":
                if re.match(r"^\s*except(?:\s+(?:Exception|BaseException))?(?:\s+as\s+\w+)?\s*:\s*(.*)$", line):
                    rest = re.match(r"^\s*except[^:]*:\s*(.*)$", line).group(1).strip()
                    body = rest or next_code(n).strip()
                    if body == "pass" or body.startswith("print("):
                        add("broad_except", n, fix="Catch the specific exception and handle or re-raise it.")
                if re.match(r"^\s*if\s+not\s+isinstance\(", line) and re.match(r"^\s*raise\s+(TypeError|ValueError)", next_code(n)):
                    add("defensive_check", n, fix="Drop it unless callers really can pass the wrong type.")
            else:
                if re.search(r"\bcatch\s*(\(\s*\w+\s*\))?\s*\{", line):
                    tail = line.split("{", 1)[1] if "{" in line else ""
                    if re.search(r"console\.(log|error|warn)|printStackTrace|System\.out|fmt\.Print", tail + next_code(n)):
                        add("broad_except", n, fix="Handle the error or let it propagate instead of only logging it.")
                if re.match(r"^\s*if\s*\(\s*(typeof\s+\w+\s*!==?|!Array\.isArray)", line) and re.match(r"^\s*throw\b", next_code(n)):
                    add("defensive_check", n, fix="Drop it unless callers really can pass the wrong type.")

        # --- python docstring rules
        if parsed:
            documented = 0
            typed = 0
            counted = 0
            for fn in funcs:
                doc = ast.get_docstring(fn, clean=True)
                body_len = (fn.end_lineno or fn.lineno) - fn.lineno
                if fn.name.startswith("__") and fn.name.endswith("__"):
                    continue
                counted += 1
                if doc:
                    documented += 1
                    first = doc.strip().split("\n")[0].strip().lower()
                    if re.match(r"^(this|the)\s+(function|method|class|module|script)\b|^(function|method|helper)\s+(to|that|which|for)\b|^helper\b|^a (simple|basic)\b", first):
                        add("docstring_generic", fn.lineno, fix="Start with what it does: 'Return the total price with tax.'")
                    if re.search(r"\bArgs?:|:param\b|\bParameters\b\s*\n\s*-{3,}", doc) and re.search(r"\bReturns?:|:return|\bReturns\b\s*\n\s*-{3,}", doc) and body_len <= 12:
                        add("docstring_boilerplate", fn.lineno, fix="A one-line docstring is enough for a function this small.")
                a = fn.args
                params = [p for p in a.posonlyargs + a.args + a.kwonlyargs if p.arg not in ("self", "cls")]
                if all(p.annotation is not None for p in params) and fn.returns is not None:
                    typed += 1
            if counted >= 3 and documented == counted and typed == counted:
                add("uniform_docs", None, message=f"All {counted} functions have a docstring and complete type hints.")

        # --- score
        per_rule: dict[str, list[float]] = {}
        for f in findings:
            per_rule.setdefault(f["rule"], []).append(f["weight"])
        contributions = {}
        for rule, ws in per_rule.items():
            contributions[rule] = sum(w * (0.85 ** i) for i, w in enumerate(sorted(ws, reverse=True)))
        total = sum(contributions.values())
        offset = 0.8 * min(human_signals, 5)
        total = max(0.0, total - offset)
        denom = 10 + 0.12 * loc
        score = round(100 * (1 - math.exp(-total / denom)))
        if score >= 55:
            label, key = "Likely AI-generated style", "ai"
        elif score >= 28:
            label, key = "Mixed - worth reviewing", "mixed"
        else:
            label, key = "Likely human-written style", "human"

        findings.sort(key=lambda f: (f["line"] is not None, f["line"] or 0, f["rule"]))
        summary = []
        for rule, ws in sorted(per_rule.items(), key=lambda kv: -contributions[kv[0]]):
            summary.append({"rule": rule, "title": RULES[rule][0], "category": RULES[rule][1],
                            "count": len(ws), "points": round(contributions[rule], 1)})
        reasons = [f"{s['count']} x {s['title']}" for s in summary[:4]]

        # --- cleaned copy: only the redundant comments are touched
        out = []
        for i, raw in enumerate(lines, start=1):
            mode = removable.get(i)
            if mode == "line":
                continue
            if mode == "inline":
                col = next((c["col"] for c in comments if c["line"] == i and c["inline"]), None)
                if col is not None:
                    raw = raw[:col].rstrip()
            out.append(raw)
        cleaned = "\n".join(out)
        removed = len(lines) - len(out) + sum(1 for m in removable.values() if m == "inline")

        notes = []
        if loc < 6:
            notes.append("Very short snippet: the score is low-confidence. Paste a full file for a better read.")
        if lang == "python" and not parsed:
            notes.append("The Python did not parse, so docstring and type-hint rules were skipped.")
        if human_signals:
            notes.append(f"{human_signals} comment(s) explain a reason, workaround or TODO, which reads as human context.")

        return {
            "label": label, "verdict": key, "score": score, "language": lang,
            "reasons": reasons, "summary": summary, "findings": findings, "notes": notes,
            "stats": {"code_lines": loc, "comment_lines": comment_line_count, "comment_density": round(density, 2),
                      "findings": len(findings), "human_signals": human_signals, "functions": len(funcs)},
            "cleaned": cleaned, "removed_comments": removed,
            "disclaimer": "This measures writing style, not authorship. Treat it as a hint to review, not proof.",
        }
