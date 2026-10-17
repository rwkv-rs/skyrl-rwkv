"""Run the existing lighteval-rwkv producer against SkyRL's inference pool.

Prototype: run_rwkv_sft.sh's LightEval producer. Tasks, prompts, extraction,
grading and streaming results remain owned by lighteval-rwkv. Only completion
budgets are adapted to this experiment's explicit total-context limit.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


def create_eval_server_pool(cfg, worker_locations):
    """Create native vLLM servers on offloaded trainer GPUs, as in ServerGroup."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from skyrl.backends.skyrl_train.inference_servers.common import SERVER_PORT_STRIDE
    from skyrl.backends.skyrl_train.inference_servers.engine_utils import build_engine_runtime_env
    from skyrl.backends.skyrl_train.inference_servers.server_pool import ServerActorPool
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
    from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import VLLMServerActor

    args = build_vllm_cli_args(cfg)
    args.tensor_parallel_size = args.pipeline_parallel_size = args.data_parallel_size = 1
    args.enable_sleep_mode = False
    actors = []
    for index, (node_id, gpu_id) in enumerate(worker_locations):
        actor = ray.remote(VLLMServerActor).options(
            num_gpus=0,
            num_cpus=0.2,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env=build_engine_runtime_env(
                use_expandable_segments=cfg.generator.inference_engine.use_expandable_segments
            ),
        )
        actors.append(
            actor.remote(
                args,
                start_port=18000 + index * SERVER_PORT_STRIDE,
                server_idx=index,
                distributed_executor_backend="mp",
                mp_cuda_visible_devices=str(gpu_id),
                colocated_training=True,
                enable_ray_prometheus_stats=cfg.generator.inference_engine.enable_ray_prometheus_stats,
            )
        )
    return ServerActorPool(actors)


def set_generation_budgets(docs, model, tokenizer, context_length, strip_template):
    """Use the producer's exact messages and the inference tokenizer/template."""
    for doc in docs:
        messages = strip_template(model.prompt_manager.prepare_prompt_api(doc))
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **model.config.extra_body["chat_template_kwargs"],
        )
        doc.generation_size = context_length - len(prompt_ids)
        if doc.generation_size < 1:
            raise ValueError(f"Evaluation prompt exceeds the {context_length}-token context budget")


def evaluate(args):
    project = args.project.expanduser().resolve()
    sys.path[:0] = [str(project / "src"), str(project)]
    # The producer's bounded math verifier starts a fresh Python subprocess.
    os.environ["PYTHONPATH"] = os.pathsep.join([str(project / "src"), str(project), os.environ.get("PYTHONPATH", "")])
    import litellm
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.metrics.metrics import Metrics
    from lighteval.models.endpoints.litellm_model import (
        _strip_question_answer_completion_template,
    )
    from lighteval.pipeline import ParallelismManager, PipelineParameters
    from lighteval.tasks.requests import SamplingMethod
    from src.rwkv_eval.configs import ModelEndpoint
    from src.rwkv_eval.main import RWKVExactMatches, _litelm_model
    from src.rwkv_eval.main import RWKVPipeline as ProducerPipeline
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    class RWKVPipeline(ProducerPipeline):
        def _run_model_for_docs(self, sampling_docs):
            set_generation_budgets(
                sampling_docs.get(SamplingMethod.GENERATIVE, []),
                self.model,
                tokenizer,
                args.context_length,
                _strip_question_answer_completion_template,
            )
            return super()._run_model_for_docs(sampling_docs)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    endpoint = ModelEndpoint(args.url, "unused", args.model, args.concurrency, args.context_length)
    config = _litelm_model(
        endpoint, [endpoint], "CoT", "bot", args.context_length, 42, cache_dir=str(output / ".lighteval_cache")
    )
    # The producer uses Doc.generation_size when no global completion cap is set.
    config.generation_parameters.max_new_tokens = None
    litellm.cache = None
    tracker = EvaluationTracker(output_dir=str(output), save_details=True, save_streaming_completions=True)
    pipeline = RWKVPipeline(
        args.task,
        PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            streaming_evaluation=True,
            max_samples=args.max_samples,
        ),
        tracker,
        model_config=config,
    )
    for task in pipeline.tasks_dict.values():
        metric = Metrics.avg_at_n_math(sample_params={"n": args.num_samples})
        metric.sample_level_fn.compute_score = RWKVExactMatches().compute
        task.metrics = task.config.metrics = [metric]
        task.num_samples = [args.num_samples]
        task.sampling_methods = [SamplingMethod.GENERATIVE]
    tracker.task_config_logger.log(pipeline.tasks_dict)
    pipeline.evaluate()
    pipeline.save_and_push_results()
    results = pipeline.get_results()
    metrics, scores = {}, {}
    for name in pipeline.tasks_dict:
        values = results["results"][name.replace("|", ":")]
        score = next(value for key, value in values.items() if not key.endswith("_stderr"))
        count = pipeline.task_sample_counts[name]
        completions = pipeline.task_completion_counts[name]
        if completions != args.num_samples * count or pipeline.task_avg_k[name] != args.num_samples:
            raise ValueError(f"Incomplete avg@{args.num_samples} evaluation: {name}")
        row = {
            f"avg@{args.num_samples}": float(score),
            "num_questions": count,
            "num_completions": completions,
            "truncation_rate": pipeline.task_truncation_counts[name] / completions,
        }
        scores[name] = row
        metrics.update({f"eval/lighteval/{args.task}/{key}": value for key, value in row.items()})
    (output / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (output / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "completed.json").write_text(
        json.dumps(
            {
                "global_step": args.step,
                "model_name": args.model,
                "task": args.task,
                "num_samples": args.num_samples,
                "context_length": args.context_length,
                "generation_budget": "context_length - actual prompt tokens",
                "prompt_template": "bot",
                "generation_prompt": "open_think",
                "producer": str(project),
                "completions_path": tracker.results["completions_path"],
                "producer_commit": subprocess.check_output(
                    ["git", "-C", str(project), "rev-parse", "HEAD"], text=True
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--task", default="aime24")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=25600)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.num_samples < 1 or args.context_length < 1:
        parser.error("RWKV evaluation requires positive num_samples and context_length")
    evaluate(args)


if __name__ == "__main__":
    main()
