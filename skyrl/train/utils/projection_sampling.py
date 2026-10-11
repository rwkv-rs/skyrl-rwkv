"""Projection-inspired sampling for offline SFT data.

Algorithm reference: Karan, Chen & Du, arXiv:2610.02140v1, Appendix C.2.
Function prototypes: aakaran/finetuning-with-sampling, boost_math.py /
boost_sci.py, commit aa7080885f114fdc5d141f19c09f53008a1c061e.
This implements the published greedy mean-logprob approximation, not exact MH.
"""

import random
from collections.abc import Mapping

from loguru import logger

SEQ_PROMPT = (
    "Study the question and expert solution below. Continue the partial response in your own words, "
    "preserving the expert's reasoning and final answer. Return only the continuation, "
    "including the reasoning needed to complete the solution.\n\n"
    "Question:\n{PROBLEM}\n\nExpert solution:\n{SOLUTION}\n\nPartial response:\n{RESPONSE}\n"
)


class vLLMAutoregressiveSampler:
    """Use one fixed model for expert-conditioned proposals and question-only scoring."""

    def __init__(self, llm, *, model_type="chat", chat_template_kwargs=None, repetition_penalty=1.1):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.model_type = model_type
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.repetition_penalty = repetition_penalty
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

    context = p.format_prompt(question)
    ids = context + current
    p.check_context(ids, 1)
    result = p.llm.generate(
        {"prompt_token_ids": ids},
        SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1, prompt_logprobs=0),
        use_tqdm=False,
    )[0]
    values = [item[token].logprob for token, item in zip(current, result.prompt_logprobs[len(context) :], strict=True)]
    return sum(values) / len(values)


def proposal_vLLM(p, question, solution, current, ext_len, rng):
    from vllm import SamplingParams

    if p.rwkv_separator_id is not None:
        prompt = f"Expert solution:\n{solution}\nQuestion:\n{question}"
    else:
        prompt = SEQ_PROMPT.format(
            PROBLEM=question,
            SOLUTION=solution,
            RESPONSE=p.tokenizer.decode(current, skip_special_tokens=True),
        )
    ids = p.format_prompt(prompt) + current
    p.check_context(ids, ext_len)
    result = p.llm.generate(
        {"prompt_token_ids": ids},
        SamplingParams(
            max_tokens=ext_len,
            temperature=0.6,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=p.repetition_penalty,
            stop_token_ids=p.stop_token_ids,
            seed=rng.randrange(2**31),
        ),
        use_tqdm=False,
    )[0].outputs[0]
    if not result.token_ids:
        raise ValueError("Proposal returned no tokens")
    response_ids = current + list(result.token_ids)
    finished = result.finish_reason == "stop"
    if p.rwkv_separator_id is not None and response_ids[-1] == p.rwkv_separator_id:
        # Match the native generator's EOS append after a turn-ending stop.
        response_ids.append(p.tokenizer.eos_token_id)
        finished = True
    return response_ids, finished


def proj_samp_vLLM(p, question, solution, *, mcmc_steps=10, max_new_tokens=1856, block_num=32, seed=0):
    """Extend by blocks, then greedily resample suffixes from any previous position."""
    if max_new_tokens < 1 or block_num < 1 or max_new_tokens % block_num or mcmc_steps < 0:
        raise ValueError("Require positive max_new_tokens/block_num, exact divisibility, and non-negative mcmc_steps")
    rng = random.Random(seed)
    gen = []
    attempts = acceptances = 0
    finished = False
    for block in range(block_num):
        gen, finished = proposal_vLLM(p, question, solution, gen, max_new_tokens // block_num, rng)
        score = get_logprobs_vLLM(p, question, gen)
        for _ in range(mcmc_steps):
            attempts += 1
            cut = rng.randrange(len(gen))
            candidate, candidate_finished = proposal_vLLM(p, question, solution, gen[:cut], len(gen) - cut, rng)
            candidate_score = get_logprobs_vLLM(p, question, candidate)
            if candidate_score > score:
                gen, score, finished = candidate, candidate_score, candidate_finished
                acceptances += 1
        logger.info(
            "Sampling block {}/{}: tokens={}, mean_logprob={:.4f}, accepted={}/{}, finished={}",
            block + 1,
            block_num,
            len(gen),
            score,
            acceptances,
            attempts,
            finished,
        )
        if finished or p.tokenizer.eos_token_id in gen:
            finished = True
            break
    return {
        "response_ids": gen,
        "target_log_prob_cur": score,
        "finish_reason": "stop" if finished else "length",
        "attempts": attempts,
        "acceptances": acceptances,
    }
