from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).parents[4] / "examples/train/rwkv/rollout_gsm8k.py"
spec = importlib.util.spec_from_file_location("rollout_gsm8k", SCRIPT)
rollout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout)


def test_rwkv_fake_think_answer_extraction():
    assert rollout.answer_text(">reason</think>\\boxed{4}") == r"\boxed{4}"
    assert rollout.answer_text(">plain answer") == "plain answer"


def test_text_match_uses_final_answer_and_option_mapping():
    assert rollout.text_score(
        "Earlier I considered 4.\nFinal answer: 5", "5", "question"
    )
    assert not rollout.text_score("Earlier answer: 4\nFinal answer: 5", "4", "question")
    query = "Which?\na: wrong\nb: The polynomial hierarchy collapses\nc: other"
    assert rollout.text_score("Final answer: b", "The polynomial hierarchy collapses", query)


def test_code_python_and_cpp_execution():
    fence = chr(96) * 3
    tests = {"inputs": [""], "outputs": ["3\n"]}
    assert rollout.run_code(">" + fence + "python\nprint(3)\n" + fence, tests) == (True, "ok")
    cpp = fence + 'cpp\n#include <iostream>\nint main(){std::cout << 3 << "\\n";}\n' + fence
    assert rollout.run_code(cpp, tests) == (True, "ok")


def test_long_context_selection_uses_token_budget(tmp_path: Path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": list(range(len(messages[0]["content"]))) }

    domain = tmp_path / "Long_Context"
    domain.mkdir()
    path = domain / "part.jsonl"
    path.write_text(
        '{"uuid":"Long_Context_1","query":"123456","ground_truth":"5","domain":"Long_Context"}\n'
        '{"uuid":"Long_Context_2","query":"12","ground_truth":"2","domain":"Long_Context"}\n'
    )
    rows = list(rollout.rows(tmp_path, 4, Tokenizer(), 10, 5))
    assert [row["uuid"] for row in rows] == ["Long_Context_2"]


def test_write_output_keeps_question_ids_disjoint(tmp_path: Path):
    from collections import Counter
    groups = {
        "Math": {
            "Math_1": {"row": {"uuid": "Math_1", "query": "q", "ground_truth": "1"},
                       "counts": Counter(correct=1, wrong=0, unanswered=0),
                       "samples": {"correct": {"response": "1", "sample_index": 0, "reason": "math_verify"}}},
        },
        "Code": {
            "Code_1": {"row": {"uuid": "Code_1", "query": "q", "ground_truth": {"inputs": [""], "outputs": [""]}},
                       "counts": Counter(correct=0, wrong=1, unanswered=0),
                       "samples": {"wrong": {"response": "", "sample_index": 0, "reason": "wrong_output"}}},
        },
    }
    rollout.write_output(tmp_path, groups, 8192, 2)
    files = [tmp_path / f"{status}_samples.jsonl" for status in ("correct", "wrong", "unanswered")]
    ids = [line.split('"uuid": "')[1].split('"')[0] for file in files for line in file.read_text().splitlines()]
    assert len(ids) == len(set(ids))
