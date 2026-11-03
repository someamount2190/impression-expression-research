# Evidence and limitations

This repository uses four practical evidence categories:

1. **Implemented** â€” code or a test demonstrates that a component exists and performs a specified operation.
2. **Measured** â€” a bounded run records a result for its declared inputs, hardware, and protocol.
3. **Replicated** â€” the key result recurs in independent runs or seeds under a preregistered comparison.
4. **Generalized** â€” performance holds across held-out workload families, with the relevant runtime and safety checks.

Do not promote a claim to a stronger category without corresponding evidence. In particular:

- Small token-loss changes do not establish better generated plans or task success.
- Historical receipts do not substitute for a new matched comparison.
- Counterfactual resource scenarios are not actual hardware observations.
- A broker rejection is a successful safety outcome, not a successful model decision.
- A macro replay verifies execution on its test cases; it does not establish model-originated discovery, unseen transfer, or compression.
- One or two prompts do not establish adapter quality or text-generation parity. Tensor mapping checks and full-logit parity are distinct.
- In-process consistency does not replace fresh-process loading and rollback checks.

Reports should disclose failed attempts, all denominators, repeats, seeds, split provenance, hardware, resource limits, deviations, and any unavailable measurements. Keep raw proposals separate from broker correction, fallback, execution, and final verified outcomes.
