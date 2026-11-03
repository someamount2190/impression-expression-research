# Reproduction guide

## Before running

1. Read the experimentâ€™s report, config, journal, and run-state file.
2. Check its frozen manifests and source revision. Preserve the original experiment directory.
3. Confirm the required model revision, framework versions, runtime (including Ollama where applicable), memory reserve, and hardware. Model weights and large checkpoint artifacts are not distributed here.
4. Run from the full [PROJECT_NAME] workspace only after checking each commandâ€™s output paths. Some commands overwrite receipts or logs. Use a disposable copy for expensive or mutating runs.
5. Do not train or select parameters on sealed test data. Keep counterfactual profiles separate from measured hardware runs.

## Selected report commands

The Impression 006 report records these repository-root commands for its bounded checks (paths refer to the original [PROJECT_NAME] workspace):

```powershell
python experiments/impression_006/frozen_sets/inventory_verified_outcomes.py
python -m experiments.impression_006.decision_diagnostics.run_prompt_ablation
python experiments/impression_006/adapter_inference/verify_gguf_mapping.py
python experiments/impression_006/adapter_inference/generation_trial.py
python -m pytest tests/test_capability_broker.py experiments/impression_002/tests experiments/impression_005/tests -q
```

The connector capture/training command is resource-intensive and previously ran about 40 minutes without reaching an optimizer update. Do not rerun it unchanged expecting a quick result. First profile or replace the frozen-prefix capture method while preserving the frozen data split and matched control.

The Expression Growth 001 and Impression 005/006 reports include their own commands and warnings. Some commands depend on local absolute paths or omitted base weights. Inspect before execution and adapt paths only in a working copy; record any deviation.

## Hashes and receipts

`research/SOURCE_MANIFEST.json` hashes the curated files in this repository. Experiment manifests hash their own source data and outputs. These serve different purposes; do not rewrite frozen experiment manifests simply to make a copied directory appear original.
