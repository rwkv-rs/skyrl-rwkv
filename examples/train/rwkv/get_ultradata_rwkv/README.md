# RWKV UltraData filtering

This recipe builds an RWKV-specific filtered corpus from the **unfiltered
upstream source data used to construct UltraData-RL-2609**. The launcher first
downloads and normalizes those sources with `prepare_source.py`, then runs the
canonical multi-GPU rollout in `train/common/rollout.py` with the RWKV eval
sampling policy. It removes a question only when every sampled response is
correct. Questions with a partial pass rate or no correct responses remain in
the output.

The released UltraData-RL-2609 files are not a valid input here: they already
contain the construction-time pass-rate-1 filter for the MiniCPM checkpoint.
The source preparation stage writes each row in the verifier schema used by the
rollout code:

```text
uuid, query, ground_truth, source, domain
```

The default rollout uses the shared eval sampling preset and 64 samples per
question, matching the pass-rate screening budget used by the rollout
implementation. That preset uses temperature 0.96, top-p 0.76, top-k 32,
presence penalty 1.0, frequency penalty 0.1, and penalty decay 0.988. The
rollout output contains `correct_counts_by_question.jsonl`, and the filtered
dataset contains the original domain shards plus a `manifest.json` with
per-domain counts.

Run the full filtering job with:

```bash
bash examples/train/rwkv/get_ultradata_rwkv/run.sh
```

The launcher writes the unfiltered source shards to
`$HOME/data/ultradata-rwkv-source/files` and uses the RWKV checkpoint at
`$HOME/Weights/RWKV/hf/rwkv7-g1k-1.5b-20260930-ctx25600` by default. Set
`SOURCE_ROOT`, `MODEL`, `ROLLOUT_OUTPUT`, and `FILTERED_OUTPUT` to use different
locations. Set `PREPARE_SOURCE=false` to reuse an existing normalized source
directory, or set `PREPARE_LIMIT_PER_SOURCE` for a bounded preparation smoke
run. Set `BASE_URL` to a comma-separated list of already-running
OpenAI-compatible completion endpoints instead of starting one vLLM-RWKV
engine per visible GPU.

The normalized source corpus uses the four domain directories:
`Math`, `Code`, `Long_Context`, and `Knowledge`. The release describes its
difficulty-calibration inputs as a union of DAPO-Math-17k, DeepScaleR, and
DeepMath for Math; OpenScienceReasoning-2 for Knowledge; HotpotQA, Qasper,
MuSiQue, and in-house long-context data; and OpenCodeReasoning,
OpenCodeReasoning-2, and HardTests for Code. The preparation manifest records
the source provenance and the unverifiable OpenCode releases. Those releases
provide model solutions but no public executable tests, so they are skipped;
HardTests supplies code rows with public test cases for the native verifier.
The manifest also records the in-house synthetic long-context source as
unavailable because it is not publicly downloadable.
The recipe then reruns the pass-rate screen with the selected RWKV checkpoint
and keeps every item that RWKV does not solve on every sample.
