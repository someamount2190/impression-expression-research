# System overview

## Impression

Impression is the projectâ€™s research track for modifying and evaluating neural computation. The curated implementation includes a model tensor registry and selective-training support. The experiment sequence explores layer partitioning, adapter training, runtime and hardware placement, decision policies, and a residual connector between model layers.

The evidence supports concrete infrastructure claimsâ€”bounded tensor handling, parameter updates, artifact reload, and selected measured execution profiles. It does not yet establish broad quality gains, an autonomously chosen graph change, or reliable local decision competence.

## Expression

Expression is the symbolic plan and execution track. It represents workflows as typed nodes that refer to registered operations, variables, dependencies, branches, and macro calls. An interpreter validates and executes plans through a broker, which remains responsible for permission and resource checks.

A macro can stand for a repeated sequence and thus offers a route to shorter plans. In the included Growth 001 evidence, one two-operation macro was mined from existing traces and replayed. The model did not discover or demonstrate use of that macro in generation. No token savings were measured.

## Boundary and maturity

A plan representation or broker-accepted proposal is not proof of model learning. A trained update is not proof of useful generalization. A manual architecture experiment is not autonomous adaptation. Reports should distinguish model output, deterministic selection, broker decision, execution, and independently verified outcome.
