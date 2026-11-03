# Experience ledger

`ExperienceLedger` appends one JSON object per line and never edits earlier
events. Use event types such as `profile`, `selection`, `configuration_transition`,
`rollback`, and `reload_validation`. Each event has a unique `event_id`, UTC
epoch timestamp, and a payload containing the measured profile or decision.

Profiles should retain the exact task/objective fingerprint, training and held
out IDs, checkpoint revision/hash, config ID, block ID, placement, phase timings,
RSS/VRAM, transfer/activation costs, adapter/base fingerprints, validation
metrics, and parity status. Missing measurements stay null. A profile that was
not measured on a device must not use a fabricated zero.

The controller is deterministic. Ledger rows are evidence for later analysis
and potential future learned policies; they do not imply that a policy has been
trained.
