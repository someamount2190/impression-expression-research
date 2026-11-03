# IMPRESSION-001B Journal

## Scope and preservation

- Continued from IMPRESSION-001 in the existing [SYSTEM_NAME] repository. No original adapter, report, registry, reference, or [BASE_MODEL] checkpoint was overwritten or removed.
- Used one delegated `gpt-6-luna` coding task for harness/audit implementation. No recursive delegation. No additional model executor was run concurrently.
- Pinned base remained `[BASE_MODEL_REPOSITORY]` at `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`.

## Findings and fix

- Historical IMPRESSION-001 reported a 0.625 max logit delta from CPU training to fresh reload, while two fresh processes agreed. The original training logits had been replaced by a fresh-process artifact, so that original output was not reconstructed.
- IMPRESSION-001 used a CPU base with `hf_device_map={"": "cpu"}`. PEFT 0.21.2 `PeftModel.from_pretrained` detects CPU/disk `hf_device_map` and defaults adapter reintegration to `device_map="auto"` unless overridden. With the RTX 2050 available, this moved part of the fresh model to `cuda:0`. The new harness caught a CUDA parameter after adapter load (`_parameter_fingerprint` failed converting a `cuda:0` tensor to NumPy); the failing capture preserved its incomplete staging data. Earlier reload logs also contained CuBLAS warnings and did not check every parameter device after PEFT attachment.
- Fixed both fresh reload sites in `training/impression_selective_training.py` to pass `device_map={"": "cpu"}`. The 001B harness pins the same CPU map and fails if any parameter leaves CPU. It checks frozen-base hashes after adapter attachment and records tokenizer/runtime identity.
- Changed deterministic execution from `warn_only=True` to strict deterministic algorithms, set `CUBLAS_WORKSPACE_CONFIG` before importing torch, and persist the relevant environment and numerical settings.
- Fixed reference order: the controlled run persists training-process logits and inputs atomically before saving the adapter. A live-versus-saved adapter byte hash check is preserved.

## Controlled result

- Actual one-step BF16 CPU update of the real layer-zero FFN gate LoRA; 45,056 adapter parameters; base model frozen. No full-backbone training.
- Controlled adapter: `checkpoints/impression_001b/controlled-02`.
- Training-process reference: `experiments/impression_001b/references/controlled-02-training-process/`.
- Adapter live tensor hashes matched serialized Safetensors hashes exactly. The before/after frozen-base SHA-256 was identical: `d31f019f638f0f90d2fbf6d9ed05e2dc8653410b016a392c9a20ac3e32a70feb`.
- Training reference shape: `[1, 16, 128258]` (2,052,128 float32 logits). On CPU-pinned reload, training versus fresh reload yielded max/mean absolute difference 0.0, RMSE 0.0, zero mismatches, allclose true at `rtol=1e-4, atol=1e-4`.
- A second independent fresh CPU process produced bit-identical logits to the first CPU reload (same zero-difference metrics).
- Same-process PEFT unload/reload preserved the output exactly (max/mean absolute difference 0.0; zero mismatches), matched the original training-process reference exactly, and retained the same frozen-base fingerprint.
- CPU baseline versus adapter-disabled output was identical (max/mean absolute difference 0.0). The adapter changed only the intended selected LoRA parameters; other base weights stayed frozen.
- An initial controlled pair before the fix reproduced max diff 0.5625 and mean diff 0.0421749; both base and saved adapter hashes were equal, but reload had mixed device placement. This is retained in the original references and comparison JSON.
- A fresh-load attempt that failed the 20% RAM reserve stopped before the optimizer update and did not overwrite any artifact. Another exact training/reload run was therefore not started; the already preserved controlled-02 reference passed after the fix.

## Verification

- `python -m pytest experiments/impression_001b/test_verify_parity.py -q`: 2 passed.
- `python -m py_compile experiments/impression_001b/verify_parity.py`: passed.
- Same-process unload/reload command: `python experiments/impression_001b/verify_parity.py roundtrip --id controlled-02-same-process --adapter-path checkpoints/impression_001b/controlled-02 --training-reference controlled-02-training-process`; passed with exact parity.
- Focused regression command covering registry, selective training, distributed transformer invariants/convergence, EXPRESSION integration/hardening, and the new harness: 43 passed, 1 skipped in 38.67 seconds.
- A separate `python -m pytest tests/ -q --disable-warnings --maxfail=5` process was observed running independently, but no output or completion status was available. It is not counted as a passing run. The known pyarrow collection issue remains an unverified full-suite limitation for this turn.

## Reproduction commands

```powershell
python experiments/impression_001b/verify_parity.py one-step --id controlled-02
python experiments/impression_001b/verify_parity.py capture --id controlled-02-cpu-reload --variant adapter_enabled --adapter-path checkpoints/impression_001b/controlled-02 --input-reference experiments/impression_001b/references/controlled-02-training-process/input_ids.npy
python experiments/impression_001b/verify_parity.py compare --a controlled-02-training-process --b controlled-02-cpu-reload --output controlled-02-training-v-cpu-reload
python experiments/impression_001b/verify_parity.py capture --id controlled-02-cpu-reload-2 --variant adapter_enabled --adapter-path checkpoints/impression_001b/controlled-02 --input-reference experiments/impression_001b/references/controlled-02-training-process/input_ids.npy
python experiments/impression_001b/verify_parity.py compare --a controlled-02-cpu-reload --b controlled-02-cpu-reload-2 --output controlled-02-cpu-reload-repeat
```

No efficiency gain was measured. No adaptive controller, structural expansion, EXPRESSION training, or multi-node gradient work was started.
