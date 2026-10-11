"""Projection-inspired sampling for offline SFT data.

Algorithm reference: Karan, Chen & Du, arXiv:2610.02140v1, Appendix C.2.
Function prototypes: aakaran/finetuning-with-sampling, boost_math.py /
boost_sci.py, commit aa7080885f114fdc5d141f19c09f53008a1c061e.
This implements the published greedy mean-logprob approximation, not exact MH.
"""

import random
from collections.abc import Mapping

import numpy as np
from loguru import logger

SEQ_PROMPT = (
    "Study the question and expert solution below. Continue the partial response in your own words, "
    "preserving the expert's reasoning and final answer. Return only the continuation, "
    "including the reasoning needed to complete the solution.\n\n"
    "Question:\n{PROBLEM}\n\nExpert solution:\n{SOLUTION}\n\nPartial response:\n{RESPONSE}\n"
)


class vLLMAutoregressiveSampler:
    """Use one fixed model for expert-conditioned proposals and question-only scoring."""

    def __init__(
        self,
        llm,
        *,
        model_type="chat",
        chat_template_kwargs=None,
        repetition_penalty=1.1,
        sampling_params=None,
    ):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.model_type = model_type
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.sampling_params = {
            "temperature": 0.6,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": repetition_penalty,
        }
        self.sampling_params.update(sampling_params or {})
        self.max_model_len = llm.llm_engine.model_config.max_model_len
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        self.rwkv_separator_id = None
        if self.model_type == "chat" and self.chat_template_kwargs.get("rwkv_prompt_template") == "bot":
            self.rwkv_separator_id = self.tokenizer.encode("✿", add_special_tokens=False)[0]
            self.stop_token_ids.append(self.rwkv_separator_id)

    def format_prompt(self, text):
        if self.model_type == "base":
            return self.tokenizer.encode(text)
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        return list(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)

    def check_context(self, ids, new_tokens):
        if len(ids) + new_tokens > self.max_model_len:
            raise ValueError(
                f"Context needs {len(ids) + new_tokens} tokens, exceeds max-model-len={self.max_model_len}; "
                "increase the explicit context budget (prompts are not truncated)"
            )


def get_logprobs_vLLM(p, question, current):
    """Mean response logprob under the unprivileged, untempered base model."""
    from vllm import SamplingParams

    single = isinstance(question, str)
    questions = [question] if single else question
    currents = [current] if single else current
    contexts = [p.format_prompt(q) for q in questions]
    prompts = []
    for context, response in zip(contexts, currents, strict=True):
        ids = context + response
        p.check_context(ids, 1)
        prompts.append({"prompt_token_ids": ids})
    results = p.llm.generate(
        prompts[0] if single else prompts,
        SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=0),
        use_tqdm=False,
    )
    scores = []
    for context, response, result in zip(contexts, currents, results, strict=True):
        values = [
            item[token].logprob for token, item in zip(response, result.prompt_logprobs[len(context) :], strict=True)
        ]
        scores.append(sum(values) / len(values))
    return scores[0] if single else scores


def proposal_vLLM(p, question, solution, current, ext_len, rng):
    from vllm import SamplingParams

    single = isinstance(question, str)
    rows = (
        [(question, solution, current, ext_len, rng)]
        if single
        else zip(question, solution, current, ext_len, rng, strict=True)
    )
    prompts, params, currents = [], [], []
    for q, s, response, length, generator in rows:
        if p.rwkv_separator_id is not None:
            prompt = f"Expert solution:\n{s}\nQuestion:\n{q}"
        else:
            prompt = SEQ_PROMPT.format(
                PROBLEM=q,
                SOLUTION=s,
                RESPONSE=p.tokenizer.decode(response, skip_special_tokens=True),
            )
        ids = p.format_prompt(prompt) + response
        p.check_context(ids, length)
        prompts.append({"prompt_token_ids": ids})
        currents.append(response)
        params.append(
            SamplingParams(
                max_tokens=length,
                **p.sampling_params,
                stop_token_ids=p.stop_token_ids,
                seed=generator.randrange(2**31),
            )
        )
    results = p.llm.generate(
        prompts[0] if single else prompts,
        params[0] if single else params,
        use_tqdm=False,
    )
    proposals = []
    for response, result in zip(currents, results, strict=True):
        output = result.outputs[0]
        if not output.token_ids:
            raise ValueError("Proposal returned no tokens")
        response_ids = response + list(output.token_ids)
        finished = output.finish_reason == "stop"
        if p.rwkv_separator_id is not None and response_ids[-1] == p.rwkv_separator_id:
            # Match the native generator's EOS append after a turn-ending stop.
            response_ids.append(p.tokenizer.eos_token_id)
            finished = True
        proposals.append((response_ids, finished))
    return proposals[0] if single else proposals


def has_repetition(tokens):
    """Detect three adjacent copies of a 16–128 token span, using repeated n-gram matching."""
    ids = np.asarray(tokens)
    for width in range(16, len(ids) // 3 + 1):
        mismatches = np.concatenate(([0], np.cumsum(ids[width:] != ids[:-width])))
        if np.any(mismatches[2 * width :] == mismatches[: -2 * width]):
            return True
    return False


def proj_samp_vLLM(
    p,
    question,
    solution,
    *,
    mcmc_steps=10,
    max_new_tokens=1856,
    block_num=32,
    seed=0,
    skip_repetition=False,
):
    """Extend by blocks, then greedily resample suffixes from any previous position."""
    if max_new_tokens < 1 or block_num < 1 or max_new_tokens % block_num or mcmc_steps < 0:
        raise ValueError("Require positive max_new_tokens/block_num, exact divisibility, and non-negative mcmc_steps")
    single = isinstance(question, str)
    questions = [question] if single else question
    solutions = [solution] if single else solution
    seeds = [seed] if single else seed
    rngs = [random.Random(s) for s in seeds]
    states = [
        dict(
            response_ids=[],
            target_log_prob_cur=None,
            finish_reason="length",
            attempts=0,
            acceptances=0,
            rejected_repetitions=0,
        )
        for _ in questions
    ]
    active = list(range(len(questions)))

    def propose(indices, prefixes, lengths):
        args = (
            [questions[i] for i in indices],
            [solutions[i] for i in indices],
            prefixes,
            lengths,
            [rngs[i] for i in indices],
        )
        if single:
            return [proposal_vLLM(p, *(a[0] for a in args))]
        return proposal_vLLM(p, *args)

    def score(indices, responses):
        if single:
            return [get_logprobs_vLLM(p, questions[indices[0]], responses[0])]
        return get_logprobs_vLLM(p, [questions[i] for i in indices], responses)

    for block in range(block_num):
        proposals = propose(
            active,
            [states[i]["response_ids"] for i in active],
            [max_new_tokens // block_num] * len(active),
        )
        finished = {}
        for i, (gen, ended) in zip(active, proposals, strict=True):
            states[i]["response_ids"] = gen
            finished[i] = ended
            if skip_repetition and has_repetition(gen):
                states[i]["finish_reason"] = "repetition"
        active = [i for i in active if states[i]["finish_reason"] != "repetition"]
        if not active:
            break
        scores = score(active, [states[i]["response_ids"] for i in active])
        for i, value in zip(active, scores, strict=True):
            states[i]["target_log_prob_cur"] = value
        for _ in range(mcmc_steps):
            cuts = [rngs[i].randrange(len(states[i]["response_ids"])) for i in active]
            proposals = propose(
                active,
                [states[i]["response_ids"][:cut] for i, cut in zip(active, cuts, strict=True)],
                [len(states[i]["response_ids"]) - cut for i, cut in zip(active, cuts, strict=True)],
            )
            candidates = []
            for i, (candidate, ended) in zip(active, proposals, strict=True):
                states[i]["attempts"] += 1
                if skip_repetition and has_repetition(candidate):
                    states[i]["rejected_repetitions"] += 1
                else:
                    candidates.append((i, candidate, ended))
            if candidates:
                values = score([i for i, _, _ in candidates], [gen for _, gen, _ in candidates])
                for (i, gen, ended), value in zip(candidates, values, strict=True):
                    if value > states[i]["target_log_prob_cur"]:
                        states[i]["response_ids"] = gen
                        states[i]["target_log_prob_cur"] = value
                        finished[i] = ended
                        states[i]["acceptances"] += 1
        for i in active:
            state = states[i]
            ended = finished[i] or p.tokenizer.eos_token_id in state["response_ids"]
            logger.info(
                "Sampling row {} block {}/{}: tokens={}, mean_logprob={:.4f}, accepted={}/{}, finished={}",
                i,
                block + 1,
                block_num,
                len(state["response_ids"]),
                state["target_log_prob_cur"],
                state["acceptances"],
                state["attempts"],
                ended,
            )
            if ended:
                state["finish_reason"] = "stop"
        active = [i for i in active if states[i]["finish_reason"] == "length"]
        if not active:
            break
    return states[0] if single else states
