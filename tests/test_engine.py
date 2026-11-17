import json
import random
import string
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402
import engine  # noqa: E402

C = engine.CodeChecker()


def rules(code, **kw):
    return {f["rule"] for f in C.check(code, **kw)["findings"]}


class CommentRules(unittest.TestCase):
    def test_narration_flagged(self):
        self.assertIn("narration", rules("# Initialize the counter\ncounter = 0\n"))

    def test_reason_comment_not_flagged(self):
        self.assertNotIn("narration", rules("# retry because the API is flaky on mondays\nretries = 3\n"))

    def test_step_comment(self):
        self.assertIn("step_comment", rules("# Step 1: load\nx = load()\n"))

    def test_banner(self):
        self.assertIn("banner", rules("# ===== Helpers =====\ndef f():\n    pass\n"))

    def test_placeholder_comment(self):
        self.assertIn("placeholder_comment", rules("def f():\n    pass\n# ... rest of the code remains the same\n"))

    def test_strong_phrase(self):
        self.assertIn("ai_phrase_strong", rules("# Here's how to compute it\nx = 1\n"))

    def test_directive_comments_ignored(self):
        self.assertEqual(rules("import os  # noqa: F401\n"), set())

    def test_hash_inside_string_is_not_comment(self):
        r = C.check('url = "http://x.com/#frag"\nprint(url)\n')
        self.assertEqual(r["stats"]["comment_lines"], 0)

    def test_c_style_block_and_line_comments(self):
        r = C.check("// Initialize the total\nlet total = 0;\n/* Return the total */\nreturn total;\n")
        self.assertIn("narration", {f["rule"] for f in r["findings"]})
        self.assertEqual(r["language"], "c-style")


class CodeRules(unittest.TestCase):
    def test_emoji(self):
        self.assertIn("emoji", rules('print("done \u2705")\n'))

    def test_placeholder_value(self):
        self.assertIn("placeholder_value", rules('KEY = "YOUR_API_KEY"\n'))

    def test_broad_except_print(self):
        self.assertIn("broad_except", rules("try:\n    x()\nexcept Exception as e:\n    print(e)\n"))

    def test_specific_except_ok(self):
        self.assertNotIn("broad_except", rules("try:\n    x()\nexcept KeyError:\n    print('missing')\n"))

    def test_immediate_return(self):
        self.assertIn("immediate_return", rules("def f(a):\n    result = a * 2\n    return result\n"))

    def test_redundant_else(self):
        self.assertIn("redundant_else", rules("def f(a):\n    if a:\n        return 1\n    else:\n        return 2\n"))

    def test_for_else_not_flagged(self):
        self.assertNotIn("redundant_else", rules("def f(a):\n    for i in a:\n        if i:\n            break\n    else:\n        return 2\n"))

    def test_bool_compare(self):
        self.assertIn("bool_compare", rules("if flag == True:\n    pass\n"))

    def test_generic_names(self):
        self.assertIn("generic_func_name", rules("def process_data(x):\n    return x\n"))

    def test_docstring_boilerplate(self):
        code = 'def f(a):\n    """Add one.\n\n    Args:\n        a: number\n\n    Returns:\n        number\n    """\n    return a + 1\n'
        self.assertIn("docstring_boilerplate", rules(code))


class Scoring(unittest.TestCase):
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            C.check("   \n")

    def test_too_long_rejected(self):
        with self.assertRaises(ValueError):
            C.check("x = 1\n" * 10000)

    def test_clean_code_scores_low(self):
        r = C.check("def add(a, b):\n    return a + b\n")
        self.assertEqual(r["verdict"], "human")

    def test_ai_sample_scores_high(self):
        samples = json.loads((engine.DATA_DIR / "samples.json").read_text(encoding="utf-8"))
        r = C.check(next(s for s in samples if s["id"] == "ai-python")["code"])
        self.assertEqual(r["verdict"], "ai")
        self.assertGreaterEqual(r["score"], 70)

    def test_human_sample_scores_low(self):
        samples = json.loads((engine.DATA_DIR / "samples.json").read_text(encoding="utf-8"))
        r = C.check(next(s for s in samples if s["id"] == "human-python")["code"])
        self.assertEqual(r["verdict"], "human")

    def test_score_in_range(self):
        for case in json.loads((engine.DATA_DIR / "testset.json").read_text(encoding="utf-8")):
            self.assertTrue(0 <= C.check(case["code"])["score"] <= 100)

    def test_unparseable_python_still_works(self):
        r = C.check("def broken(:\n    # Return the value\n    return value\n", language="python")
        self.assertIn("did not parse", " ".join(r["notes"]))


class Cleaning(unittest.TestCase):
    def test_removes_only_flagged_comments(self):
        src = "# Initialize the counter\ncounter = 0\n# retry because upstream drops connections\nretries = 3\n"
        r = C.check(src)
        self.assertEqual(r["cleaned"], "counter = 0\n# retry because upstream drops connections\nretries = 3\n")

    def test_inline_comment_stripped_code_kept(self):
        r = C.check("total = 0  # Initialize the total\nprint(total)\n")
        self.assertIn("total = 0\n", r["cleaned"])
        self.assertNotIn("Initialize", r["cleaned"])

    def test_code_unchanged_when_nothing_flagged(self):
        src = "def add(a, b):\n    return a + b\n"
        self.assertEqual(C.check(src)["cleaned"], src)

    def test_cleaned_python_still_parses(self):
        import ast
        samples = json.loads((engine.DATA_DIR / "samples.json").read_text(encoding="utf-8"))
        for s in samples:
            if s["language"] == "python":
                ast.parse(C.check(s["code"])["cleaned"])


class Fuzz(unittest.TestCase):
    def test_random_text_never_crashes(self):
        rnd = random.Random(7)
        alphabet = string.printable + "\u2705\u00e9\U0001F680"
        for _ in range(200):
            text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 300)))
            if text.strip():
                C.check(text)
                C.check(text, language="python")
                C.check(text, language="c-style")


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, path, data=None, raw=None):
        body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=body,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read()) if "json" in resp.headers["Content-Type"] else resp.read()
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index(self):
        status, body = self.req("/")
        self.assertEqual(status, 200)
        self.assertIn(b"AI Code Checker", body)

    def test_check(self):
        status, body = self.req("/api/check", {"code": "# Return the result\nreturn result\n"})
        self.assertEqual(status, 200)
        self.assertIn("score", body)

    def test_bad_json(self):
        self.assertEqual(self.req("/api/check", raw=b"{nope")[0], 400)

    def test_missing_field(self):
        self.assertEqual(self.req("/api/check", {"text": "x"})[0], 400)

    def test_empty_code(self):
        self.assertEqual(self.req("/api/check", {"code": " "})[0], 400)

    def test_rules_and_samples(self):
        self.assertEqual(self.req("/api/rules")[0], 200)
        self.assertEqual(self.req("/api/samples")[0], 200)

    def test_404(self):
        self.assertEqual(self.req("/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
