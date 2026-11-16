import hashlib
import atexit
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import select
import sys
import threading
import tempfile
import time


class ScoringResourceError(RuntimeError):
    pass


_slots = threading.BoundedSemaphore(int(os.environ.get("SKYRL_MATH_WORKERS", "8")))
_SCORING_TIMEOUT_SECONDS = max(0.1, float(os.environ.get("SKYRL_MATH_TIMEOUT_SECONDS", "3")))
_log_lock = threading.Lock()
_pool_lock = threading.Lock()
_worker_pools = {}
_all_workers = set()


def record(event):
    destination = Path(os.environ["SKYRL_MATH_ARCHIVE"])
    destination.mkdir(parents=True, exist_ok=True)
    with _log_lock, (destination / "scoring.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n")


class _ScoringWorker:
    def __init__(self, command):
        self.error_file = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.error_file,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.requests = 0
        with _pool_lock:
            _all_workers.add(self)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.error_file.close()
        with _pool_lock:
            _all_workers.discard(self)

    def run(self, request, rss_bytes):
        import psutil

        started = time.monotonic()
        reason = None
        peak_rss = 0
        output = bytearray()
        pending_recorded = False
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False).encode() + b"\n")
            self.process.stdin.flush()
            monitor = psutil.Process(self.process.pid)
            poller = select.poll()
            poller.register(self.process.stdout, select.POLLIN | select.POLLHUP | select.POLLERR)
            while True:
                try:
                    peak_rss = max(peak_rss, monitor.memory_info().rss)
                except psutil.NoSuchProcess:
                    reason = "worker_exit"
                    break
                if peak_rss > rss_bytes:
                    reason = "memory_limit"
                    break
                if self.process.poll() is not None:
                    reason = "worker_exit"
                    break
                elapsed = time.monotonic() - started
                if elapsed >= _SCORING_TIMEOUT_SECONDS:
                    reason = "timeout"
                    break
                if not pending_recorded and elapsed >= 30:
                    record({"event": "pending", "worker_pid": self.process.pid, "request": request})
                    pending_recorded = True
                ready = poller.poll(100)
                if ready:
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        reason = "worker_exit"
                        break
                    output.extend(chunk)
                    if b"\n" in output:
                        break
        except (BrokenPipeError, OSError):
            reason = "worker_exit"
        error = ""
        if reason is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.error_file.seek(0)
            error = self.error_file.read().decode(errors="replace")
        self.requests += 1
        return {
            "returncode": 0 if reason is None else self.process.returncode,
            "reason": reason,
            "stdout": output.decode(errors="replace"),
            "stderr": error,
            "seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss,
            "worker_pid": self.process.pid,
            "worker_requests": self.requests,
        }


def run_memory_guarded(command, request, rss_bytes):
    key = tuple(command)
    with _pool_lock:
        pool = _worker_pools.setdefault(key, [])
        worker = pool.pop() if pool else None
    if worker is None:
        worker = _ScoringWorker(command)
    reusable = False
    try:
        outcome = worker.run(request, rss_bytes)
        reusable = outcome["reason"] is None and worker.process.poll() is None
        return outcome
    finally:
        if reusable:
            with _pool_lock:
                pool.append(worker)
        else:
            worker.close()


def _shutdown_workers():
    with _pool_lock:
        workers = list(_all_workers)
        _worker_pools.clear()
    for worker in workers:
        worker.close()


atexit.register(_shutdown_workers)


def score(solution_str, ground_truth, *, ended_eod, truncated):
    request = {
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "ended_eod": ended_eod,
        "truncated": truncated,
    }
    request_id = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--worker-loop"]
    record(
        {
            "event": "submitted",
            "request_id": request_id,
            "ground_truth": ground_truth,
            "response_chars": len(solution_str),
        }
    )
    with _slots:
        for attempt, rss_bytes in enumerate((1024**3, 2 * 1024**3), 1):
            outcome = run_memory_guarded(command, request, rss_bytes)
            event = {"request_id": request_id, "attempt": attempt, "memory_limit_bytes": rss_bytes, **outcome}
            if outcome["returncode"] == 0:
                try:
                    result = json.loads(outcome["stdout"])
                    reward, details = result["reward"], result["details"]
                    if type(reward) not in (int, float) or reward not in (0, 1) or not isinstance(details, dict):
                        raise ValueError("Malformed scoring result")
                except (ValueError, KeyError, TypeError):
                    event["reason"] = "invalid_worker_result"
                else:
                    record({**event, "event": "scored", "reward": reward})
                    return reward, details
            archive = Path(os.environ["SKYRL_MATH_ARCHIVE"])
            archive.mkdir(parents=True, exist_ok=True)
            with (archive / (request_id + f"-attempt-{attempt}-{time.time_ns()}.json")).open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump({"request": request, "outcome": event}, stream, ensure_ascii=False, indent=2)
            if outcome["reason"] == "timeout":
                details = {
                    "extracted_answer": None,
                    "ground_truth_answer": ground_truth,
                    "has_think_close": False,
                    "think_close_count": solution_str.count("</think>"),
                    "thought_nonempty": False,
                    "answer_after_think": False,
                    "structural_format_valid": False,
                    "answer_parseable": False,
                    "ended_eod": bool(ended_eod),
                    "truncated": bool(truncated),
                    "is_correct": False,
                    "strict_reward": 0.0,
                    "scoring_timeout": True,
                }
                record({**event, "event": "scored_timeout", "reward": 0.0, "timeout_seconds": _SCORING_TIMEOUT_SECONDS})
                return 0.0, details
            record({**event, "event": "retry" if attempt == 1 else "failed"})
    raise ScoringResourceError(
        "Mathematical scoring worker failed after two attempts without time limits; "
        "no reward was assigned. Preserved request " + request_id + " in " + os.environ["SKYRL_MATH_ARCHIVE"]
    )


def _bounded_integer_value(expression, max_bits=4096):
    import sympy

    if isinstance(expression, sympy.Integer):
        value = int(expression)
    elif isinstance(expression, sympy.Add):
        value = 0
        for argument in expression.args:
            value += _bounded_integer_value(argument, max_bits)
            if value.bit_length() > max_bits:
                raise ValueError("Integer exceeds certificate budget")
    elif isinstance(expression, sympy.Mul):
        value = 1
        for argument in expression.args:
            factor = _bounded_integer_value(argument, max_bits)
            if value and factor and value.bit_length() + factor.bit_length() > max_bits:
                raise ValueError("Integer exceeds certificate budget")
            value *= factor
    elif isinstance(expression, sympy.Pow):
        base = _bounded_integer_value(expression.base, max_bits)
        exponent = _bounded_integer_value(expression.exp, max_bits)
        if exponent < 0 or (base == 0 and exponent == 0):
            raise ValueError("Unsupported integer power")
        if abs(base) > 1 and base.bit_length() * exponent > max_bits:
            raise ValueError("Integer exceeds certificate budget")
        value = pow(base, exponent)
    else:
        raise ValueError("Unsupported integer expression")
    if value.bit_length() > max_bits:
        raise ValueError("Integer exceeds certificate budget")
    return value


def _integer_residue(expression, modulus):
    import sympy

    if isinstance(expression, sympy.Integer):
        return int(expression) % modulus
    if isinstance(expression, sympy.Add):
        return sum(_integer_residue(argument, modulus) for argument in expression.args) % modulus
    if isinstance(expression, sympy.Mul):
        residue = 1
        for argument in expression.args:
            residue = residue * _integer_residue(argument, modulus) % modulus
        return residue
    if isinstance(expression, sympy.Pow):
        base = _integer_residue(expression.base, modulus)
        exponent = _bounded_integer_value(expression.exp)
        if exponent < 0 or (base == 0 and exponent == 0):
            raise ValueError("Unsupported integer power")
        return pow(base, exponent, modulus)
    raise ValueError("Unsupported integer expression")


def integer_modular_counterexample(gold, target):
    import sympy

    if not isinstance(gold, sympy.Integer) or not isinstance(target, sympy.Expr):
        return None
    for modulus in (101, 103, 107, 109):
        try:
            expected = _integer_residue(gold, modulus)
            actual = _integer_residue(target, modulus)
        except (ValueError, RecursionError):
            return None
        if expected != actual:
            return {
                "kind": "exact_integer_modular_counterexample",
                "modulus": modulus,
                "expected_residue": expected,
                "actual_residue": actual,
            }
    return None


def rational_counterexample(gold, target):
    import sympy

    gold = gold if isinstance(gold, list) else [gold]
    target = target if isinstance(target, list) else [target]
    expected = [value for value in gold if isinstance(value, sympy.Rational)]
    expressions = [value for value in target if isinstance(value, sympy.Expr)]
    if len(expected) != 1 or len(expressions) != 1:
        return None
    if any(not isinstance(value, (sympy.Rational, str)) for value in gold):
        return None
    if any(not isinstance(value, (sympy.Expr, str)) for value in target):
        return None
    if set(value for value in gold if isinstance(value, str)) & set(
        value for value in target if isinstance(value, str)
    ):
        return None
    expression = expressions[0]
    modular_witness = integer_modular_counterexample(expected[0], expression)
    if modular_witness is not None:
        return modular_witness
    symbols = sorted(expression.free_symbols, key=str)
    if not symbols or any(not isinstance(symbol, sympy.Symbol) for symbol in symbols):
        return None
    operations = (sympy.Add, sympy.Mul, sympy.Abs, sympy.sin, sympy.cos, sympy.tan, sympy.exp, sympy.log)
    nodes = list(sympy.preorder_traversal(expression))
    if any(
        not (
            isinstance(node, sympy.Symbol)
            or (node.is_Atom and node.is_number and node.is_finite is True)
            or isinstance(node, operations)
            or (isinstance(node, sympy.Pow) and isinstance(node.exp, sympy.Rational))
        )
        for node in nodes
    ):
        return None
    for point in (0, 1, 2, -1):
        integer = sympy.Integer(point)
        if any(
            getattr(integer, "is_" + assumption, None) is not value
            for symbol in symbols
            for assumption, value in symbol.assumptions0.items()
        ):
            continue
        substitution = {symbol: integer for symbol in symbols}
        if any(
            isinstance(node, sympy.Pow) and node.exp == 0 and node.base.xreplace(substitution) == 0 for node in nodes
        ):
            continue
        actual = expression.xreplace(substitution)
        if isinstance(actual, sympy.Rational) and actual != expected[0]:
            witness = {
                "kind": "exact_rational_counterexample",
                "expected": str(expected[0]),
                "substitution": {str(symbol): str(value) for symbol, value in substitution.items()},
            }
            try:
                witness["actual"] = str(actual)
            except ValueError:
                witness["actual_numerator_hex"] = hex(int(actual.p))
                witness["actual_denominator_hex"] = hex(int(actual.q))
            return witness
    return None


def worker(persistent=False):
    import resource
    import ctypes
    import signal
    import faulthandler

    parent_pid = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent_pid:
        return
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    site = Path(sys.executable).resolve().parents[1]
    environment = Path(sys.executable).parent.parent
    candidates = [
        environment / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
        site / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    specification = importlib.util.spec_from_file_location("isolated_math_utils", Path(__file__).with_name("utils.py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    import math_verify

    original_verify = math_verify.verify
    certificates = []

    def certified_verify(gold, target, *args, **kwargs):
        try:
            witness = rational_counterexample(gold, target)
        except Exception:
            witness = None
        if witness is not None:
            certificates.append(witness)
            return False
        return original_verify(gold, target, *args, **kwargs)

    math_verify.verify = certified_verify
    requests = sys.stdin if persistent else [json.dumps(json.load(sys.stdin))]
    for line in requests:
        request = json.loads(line)
        certificates.clear()
        faulthandler.dump_traceback_later(30, repeat=True)
        try:
            reward, details = module._compute_strict_score_local(**request)
            result = {"reward": reward, "details": details, "certificates": list(certificates)}
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        finally:
            faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__" and sys.argv[1:] == ["--worker"]:
    worker()
elif __name__ == "__main__" and sys.argv[1:] == ["--worker-loop"]:
    worker(persistent=True)
