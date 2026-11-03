# IMPRESSION-002 Journal

## Phase 0 — Repository and environment baseline

- Continued from verified IMPRESSION-001B artifacts; no backbone, adapter, registry, or prior report was overwritten.
- The 255-entry GGUF registry reloads. Layer 0 candidates include genuine attention Q/K/V/O projections and FFN gate/up/down projections. The verified base revision remains `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.
- Runtime: Python 3.10.6, Torch 2.5.1+cu121, Transformers 4.56.2, PEFT 0.21.2; RTX 2050 with 4,294,967,296 bytes VRAM (4 GiB); system RAM 24,919,851,008 bytes.
- Initial available RAM was 13.1 GB. The fixed 20% reserve is about 4.98 GB. GPU showed 159 MiB in use and about 3.84 GiB free. Full BF16 model parameters require 6,425,512,064 bytes (about 5.98 GiB) before activations and optimizer state, so full-model GPU placement was rejected before execution.
- No `.git` repository is present; the isolated `experiments/impression_002/` tree is the recoverable work area.
- An unrelated `pytest tests/ -q --disable-warnings --maxfail=5` process was observed. It remained active without readable output. Focused tests will be run separately and no claim about that full run will be made.
- One lower-cost `gpt-6-luna` executor is implementing controller/configuration/ledger foundations. It is not running model training or delegated subagents.

## Bounded measurement and resource decision

- Full [BASE_MODEL] BF16 CPU candidates were preflighted and rejected before execution. The conservative memory requirement including the reserve exceeded available RAM; GPU placement was already infeasible from the parameter lower bound. Therefore no full-model training, task-level held-out evaluation, or full model baseline timing was produced.
- As a bounded fallback, isolated probes loaded the exact pinned BF16 gate and query Safetensors tensors. Each trained a small local adapter against a deterministic synthetic rank-4 residual objective and measured local costs, memory, validation residual, and serialization/reload parity. These are component microbenchmarks, not causal-LM training.
- A deterministic measurement controller compared the two successful one-repeat profiles, applied the query component config, validated it against the registry, persisted it, and restored it through a fresh configuration store. The output hash matched after restart restore.
- Probe evidence: query total local wall time 6.5845 s versus gate 14.1085 s; peak RSS 488,005,632 versus 543,735,808 bytes; transfer time 0 for both; peak VRAM 0. Query post-probe residual 0.008011 versus gate 0.009947, while both were worse than their own initial residuals (query 0.000351; gate 0.000347). Single repeats do not establish stable/generalizable gains.
- The successful local query probe is retained as the selected *probe* configuration only. No physical relocation, multi-node placement, dependency rewiring, capacity change, or full-model adaptive runtime change occurred.
- Failed and resource-rejected attempts are preserved in `logs/` and `profiles/`; see the final report for distinctions and acceptance status.
