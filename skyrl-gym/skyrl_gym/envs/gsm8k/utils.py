# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.














\

import re
from typing import Any, Dict, Optional, Tuple


_BOXED_START_RE = re.compile(r"\\boxed\s*\{")


def _find_balanced_brace(text: str, brace_start: int) -> Optional[Tuple[str, int]]:
    if brace_start < 0 or brace_start >= len(text) or text[brace_start] != "{":
        return None

    depth = 0
    for index in range(brace_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index], index
    return None


def _iter_boxed(text: str):
    for match in _BOXED_START_RE.finditer(text):
        parsed = _find_balanced_brace(text, match.end() - 1)
        if parsed is not None:
            yield parsed[0], match.start()


def _normalize_answer(answer: Any, *, remove_commas: bool = True) -> Optional[str]:
    if answer is None:
        return None
    normalized = str(answer).strip().replace("$", "")
    if remove_commas:
        normalized = normalized.replace(",", "")
    return normalized or None


def extract_solution(solution_str, method="strict", *, remove_commas=True):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # Prefer the final boxed answer used by math-capable models, while
        # retaining the original GSM8K ``####`` format as a fallback.
        boxed = list(_iter_boxed(solution_str))
        if boxed:
            final_answer = _normalize_answer(boxed[-1][0], remove_commas=remove_commas)
        else:
            solution = re.search(r"####\s+(-?[0-9][0-9.,]*)", solution_str)
            final_answer = (
                _normalize_answer(solution.group(1), remove_commas=remove_commas) if solution is not None else None
            )
    elif method == "flexible":
        answer = re.findall(r"(-?[0-9.,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def _extract_after_think(solution_str: str) -> Optional[Tuple[str, str]]:
    """Extract the thought and answer regions from a strict-CoT response."""
    if not isinstance(solution_str, str) or solution_str.count("</think>") != 1:
        return None

    thought, answer_region = solution_str.split("</think>", 1)
    # ``open_think`` pre-fills ``<think`` in the prompt, so the generated
    # response commonly starts with ``>`` to complete that opening tag.
    if thought.startswith(">"):
        thought = thought[1:]
    if not thought.strip():
        return None
    return thought, answer_region


def _normalize_math_answer(answer: str) -> str:
    """Remove thousands separators in scalar positions, not tuples or sets."""
    answer = re.sub(r"\{\s*,\s*\}", ",", answer)
    number = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    for prefix in (r"^", r"\\(?:[dt]?frac|sqrt)\{", r"\\[dt]?frac\{[^{}]*\}\{"):
        answer = re.sub(
            rf"({prefix})({number})(?![\d,])",
            lambda match: match[1] + match[2].replace(",", ""),
            answer,
        )
    return answer


def compute_strict_score(
    solution_str: str,
    ground_truth: str,
    *,
    ended_eod: bool,
    truncated: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Score a complete strict-CoT GSM8K response.

    The prompt may prefill ``<think>``. Therefore the generated response only
    needs one non-empty thought region followed by an answer region.
    """
    parsed = _extract_after_think(solution_str)
    thought = parsed[0] if parsed is not None else ""
    answer_region = parsed[1] if parsed is not None else ""
    extracted_answer = (
        extract_solution(answer_region, method="strict", remove_commas=False) if parsed is not None else None
    )
    normalized_ground_truth = _normalize_answer(ground_truth, remove_commas=False)
    structural_format_valid = bool(parsed is not None and extracted_answer is not None)
    answer_parseable = False
    is_correct = False
    if structural_format_valid and normalized_ground_truth is not None:
        from math_verify import parse, verify

        try:
            expected = parse(f"$\\boxed{{{_normalize_math_answer(normalized_ground_truth)}}}$")
            candidate = parse(f"$\\boxed{{{_normalize_math_answer(extracted_answer)}}}$")
            answer_parseable = bool(expected and candidate)
            is_correct = bool(answer_parseable and verify(expected, candidate, strict=False))
        except Exception:
            answer_parseable = False
    strict_reward = float(is_correct and structural_format_valid and ended_eod and not truncated)

    details = {
        "extracted_answer": extracted_answer,
        "ground_truth_answer": normalized_ground_truth,
        "has_think_close": parsed is not None,
        "think_close_count": solution_str.count("</think>"),
        "thought_nonempty": bool(thought.strip()),
        "answer_after_think": extracted_answer is not None,
        "structural_format_valid": structural_format_valid,
        "answer_parseable": answer_parseable,
        "ended_eod": bool(ended_eod),
        "truncated": bool(truncated),
        "is_correct": bool(is_correct),
        "strict_reward": strict_reward,
    }
    return strict_reward, details


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
