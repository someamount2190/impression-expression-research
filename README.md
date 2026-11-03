# Impression & Expression Research

A curated, evidence-led record of two related systems developed in a research project. This repository separates implementation artifacts from experimental claims and records what has and has not been demonstrated.

## Research areas

- **Impression**: methods and tooling for adapting, profiling, and evaluating [BASE_MODEL] parameters and computational structure.
- **Expression**: typed symbolic plans, interpreters, primitive operations, and experiments in discovering and reusing compositions.
- **Integration**: the planner/runtime boundary, verified execution, and experiments connecting learned model behavior to symbolic workflows.

## Evidence snapshot

The included reports describe a research prototype, not a validated general-purpose autonomous learning system. Key reported findings include:

- **Impression 004** reports a matched training-cycle reduction from 1,558.85 seconds (dense CPU) to a 905.27-second median (hybrid CUDA/CPU) across two hybrid repeats. Held-out loss was within the configured tolerance; this is a narrow workload result, not evidence of reasoning improvement.
- **Impression 005** reports that an externally selected rank-8 residual connector preserved initial logits, was trained and reloaded in a fresh process, and could be rolled back. The held-out loss comparison used one record; the model did not choose the architecture change.
- **Impression 006** reports prompt-order sensitivity in eight local-model decision calls. The broker rejected an infeasible hybrid proposal twice. Its decision corpus remained one workload family, so no decision-policy training or fair multi-family A/B/C trial was established.
- **Expression Growth 001** implements typed plans, an interpreter, and one trace-mined two-operation macro. The model did not generate valid plans in the measured generation attempts, and no symbolic learning benefit or token savings were demonstrated.

These are summaries of the included experiment reports, not independent replications. Read each report for protocol, sample size, unsuccessful attempts, and acceptance limits.

## Repository map

- `implementation/`: selected source files for Impression, Expression, and their symbolic runtime.
- `research/`: curated reports, experiment code, tests, frozen manifests, and small receipts, grouped by experiment.
- `research/SOURCE_MANIFEST.json`: file hashes and curation record.
- `docs/`: system overview, evidence standards, chronology, and reproducibility guidance.

Large checkpoints, model weights, adapter payloads, generated logits, machine telemetry streams, credentials, and unrelated application code are intentionally excluded. Some original experiment scripts expect the full development workspace and pinned local model files. They are included for inspection; rerunning them requires reconstructing their documented environment and paths.

## Reproducibility

Start with `docs/reproduction.md`. Verify artifact hashes before relying on a receipt. Do not treat historical runs as matched trials unless the report establishes identical inputs, objective, protocol, and resource conditions. Do not use sealed test data to select prompts, hyperparameters, or methods.

## Scope and status

This repository is a curated research snapshot, not the complete application and not a model release. The documentation reports the state recorded through 2026-11-04 (Asia/Taipei). Results may be revised as experiments are reproduced or new evidence is collected.

## Citation

A citation file is provided at `CITATION.cff`. Until an archival release exists, cite the repository URL and the specific experiment report and revision used.

## License

No license is included because ownership and redistribution rights for all source, derived datasets, and model-related material have not been established. Do not assume this repository grants permission to reuse its contents. Set an appropriate license before making the repository public.
