# Historical Operational Benchmark Audit

## Scope

Inputs inspected: `reports/blind_eval_traces_100.jsonl`, `reports/blind_eval_comparison_report.json`, `reports/blind_eval_summary.md`, `reports/instrumented_four_way_100.jsonl`, and `benchmarks/run_blind_evaluation_100.py`.

## Recomputed legacy trace counts

| Arm | First action | Schema valid | Completed | Recovery flags | Repeated failures | Mean tokens | Mean recorded latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unmodified [BASE_MODEL] | 0/100 | 0/100 | 0/100 | 0 | 0 | 76.45 | 6,439.239 ms |
| Prompt specialized | 0/100 | 0/100 | 0/100 | 0 | 16 | 67.75 | 7,413.418 ms |
| Legacy [SYSTEM_NAME] | 80/100 | 88/100 | 77/100 | 88 | 0 | 126.5 | 31.169 ms |

The summary's **440%** recovery rate is not a valid rate. It is 88 recovery flags divided by the 20 tasks that were not initially correct. The flags themselves are also not consistently evidence of a recovery.

## Attribution and comparability

The legacy [SYSTEM_NAME] benchmark arm is deterministic fixture-driven code. It does not make a [BASE_MODEL] inference call; for coding and file tasks it uses the task's reference `code` or `content`. Its completion rate cannot be attributed to model competence. Its recorded latency is local runtime only, while the raw and prompt arms include model requests. Those latencies are not comparable.

The legacy traces provide limited evidence that the captured raw and prompt model attempts often returned prose rather than valid action envelopes. They do not preserve complete prompts, responses, or provider request records, so the behavior cannot be independently reconstructed in full.

## Separate instrumented run

`reports/instrumented_four_way_100.jsonl` has 400 rows and richer per-call timing/action-source/fallback/tool evidence. Its `full_[SYSTEM_NAME]` arm records **16/100 verified completions**, **3 model-generated envelopes**, **466 deterministic fallback steps**, mean **11,463.244 ms** end-to-end, and mean **11,324.539 ms** model request time. This is a different run; do not combine its rates with the legacy trace. The generated task suite also lacks a frozen family-level split manifest.

## Conclusion

The historical evidence does not establish learned [BASE_MODEL] planning or comparable model-versus-runtime performance. Treat it as development diagnostics only. New EXPRESSION-GROWTH-001 data must have a frozen provenance and family/template split before training, with model-generated choices measured before runtime repair and system outcomes measured separately from model competence.
