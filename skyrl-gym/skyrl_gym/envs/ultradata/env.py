"""Native UltraData rewards shared by offline rollouts and online RL."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import orjson

from skyrl_gym.envs.gsm8k.env import GSM8kEnv

DOMAINS = ("Math", "Code", "Long_Context", "Knowledge")


def answer_text(raw: str) -> str:
    text = raw.split("</think>", 1)[1] if "</think>" in raw else raw
    text = text.strip()
    return text[1:].lstrip() if text.startswith(">") else text


def text_score(raw: str, gold: str, query: str) -> bool:
    text = answer_text(raw)
    value = None
    start = text.rfind("\\boxed")
    if start >= 0:
        left = text.find("{", start)
        if left >= 0:
            depth = 0
            for index in range(left, len(text)):
                depth += text[index] == "{"
                depth -= text[index] == "}"
                if depth == 0:
                    value = text[left + 1 : index]
                    break
    if value is None:
        markers = r"(?:final\s+answer|correct\s+answer|answer|答案|最终答案|正确答案)\s*(?:is|是|为|[:：=])"
        matches = list(re.finditer(markers + r"\s*(.+)", text, re.IGNORECASE))
        value = matches[-1].group(1) if matches else text.splitlines()[-1] if text else ""

    candidates = [value.strip()]
    options = {
        match.group(1).upper(): match.group(2).strip()
        for match in re.finditer(r"(?m)^\s*([A-Ja-j]|\d{1,2})\s*[-.):、]\s*(.+?)\s*$", query)
    }
    labels = list(
        re.finditer(
            r"(?:^\s*[\[(]?|(?:option|choice|statement|选项)\s*(?:is\s*)?)([A-J]|\d{1,2})(?=\s*[).:：]|\s*$|\s+(?:is|would|是|为))",
            value,
            re.IGNORECASE,
        )
    )
    if labels and (label := labels[-1].group(1).upper()) in options:
        candidates.append(options[label])

    target = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", str(gold)).casefold().replace("−", "-")
    target = " ".join(target.strip(" `*_\t\r\n.,;:。；：").split())
    return bool(target) and any(
        target
        == " ".join(
            re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", str(candidate))
            .casefold()
            .replace("−", "-")
            .strip(" `*_\t\r\n.,;:。；：")
            .split()
        )
        for candidate in candidates
    )


def run_code(raw: str, gold: dict[str, Any]) -> tuple[bool, str]:
    text = answer_text(raw)
    matches = list(re.finditer(r"\x60\x60\x60\s*([\w+.-]*)\s*\n(.*?)\n\x60\x60\x60", text, re.DOTALL))
    language = matches[0].group(1).casefold() if matches else "python"
    program = "\n".join(match.group(2) for match in matches) if matches else text
    if language in {"py", "python", "python3", ""}:
        command_prefix, suffix = ["/usr/bin/python3"], ".py"
    elif language in {"cpp", "c++", "cc", "cxx"}:
        command_prefix, suffix = [], ".cpp"
    else:
        return False, "unsupported_language"
    if not program.strip():
        return False, "empty_program"
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            path.chmod(0o755)
            source = path / ("solution" + suffix)
            source.write_text(program, encoding="utf-8")
            source.chmod(0o644)
            if suffix == ".cpp":
                executable = path / "solution"
                subprocess.run(
                    ["g++", "-O2", "-std=c++17", str(source), "-o", str(executable)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                command_prefix = [str(executable)]
            command = (
                ["sudo", "-n", "-u", "nobody", "-H", "timeout", "10", *command_prefix, str(source)]
                if suffix == ".py"
                else ["sudo", "-n", "-u", "nobody", "-H", "timeout", "10", *command_prefix]
            )
            env = {"PATH": "/usr/bin:/bin", "HOME": directory}
            for stdin, expected in zip(gold["inputs"], gold["outputs"], strict=True):
                result = subprocess.run(
                    command,
                    input=str(stdin),
                    text=True,
                    cwd=directory,
                    capture_output=True,
                    timeout=10,
                    env=env,
                    check=False,
                )
                if result.returncode == 124:
                    return False, "timeout"
                if result.returncode:
                    return False, "runtime_error"
                if result.stdout.rstrip() != str(expected).rstrip():
                    return False, "wrong_output"
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError:
        return False, "compile_error"
    except (OSError, UnicodeError, KeyError):
        return False, "execution_error"


class UltraDataEnv(GSM8kEnv):
    """Single-turn UltraData scoring with per-domain RWKV prompt templates."""

    def __init__(self, env_config: Any = None, extras=None):
        source = extras["reward_spec"]
        with open(source["source_path"], "rb") as handle:
            handle.seek(source["source_offset"])
            row = orjson.loads(handle.readline())
        extras = {**extras, "reward_spec": {"ground_truth": row["ground_truth"]}}
        super().__init__(env_config, extras)
        self.domain = extras["data_source"]
        if self.domain not in DOMAINS:
            raise ValueError(f"Unknown UltraData domain: {self.domain}")
        self.query = row["query"]
        self.strict_reward = self.domain == "Math"

    def init(self, prompt):
        prompt, metadata = super().init(prompt)
        metadata["chat_template_kwargs"] = {
            "rwkv_prompt_template": "bot",
            "rwkv_generation_prompt": "open_think" if self.domain == "Math" else "fake_think",
        }
        return prompt, metadata

    def _get_reward(self, action: str) -> float:
        if self.strict_reward:
            # The generator already scores this action in set_generation_metadata
            # on its main thread. Math-Verify's signal timeout cannot run in the
            # environment thread pool used for step().
            return self._strict_reward_details["strict_reward"]
        if self._generation_metadata["truncated"] or not action.strip():
            reward, reason = False, "truncated_or_empty"
        elif self.domain == "Code":
            reward, reason = run_code(action, self.ground_truth)
        else:
            reward, reason = text_score(action, self.ground_truth, self.query), "answer_match"
        self._strict_reward_details = {"acc": float(reward), "scoring_reason": reason}
        return float(reward)

    def get_metrics(self):
        return dict(self._strict_reward_details)
