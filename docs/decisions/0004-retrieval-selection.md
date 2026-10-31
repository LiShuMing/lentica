# 0004: Select retrieval configurations before test evaluation

Status: accepted for the local opt-in experiment, see spec 0004.

The first SciFact run showed that equal fusion can reduce retrieval quality when
its second route is weak. Keep single-route retrieval as a valid candidate rather
than requiring a hybrid winner. Add corpus-fitted sparse adapters behind the same
Retriever interface, leaving rank fusion and product APIs unchanged.

Exclude training queries matching normalized test text and keep duplicate texts
together during partitioning. Develop on 80% of retained training groups, select
among at most five family finalists on 20%, and require a validation quality gate.
Persist the result before invoking a separate frozen test evaluation command.
Store route definitions, hashes, versions, scores and partition identities so the
configuration can be audited and reproduced. Checksums prevent accidental drift;
they are not an authenticity or security mechanism.

The existing test results influenced the broad direction, so no new pristine
holdout claim is made. Future pretrained/ANN/persistent adapters can use the same
protocol, but each requires its own dependency, data and runtime evidence. Sparse
cosine results cannot stand in for semantic encoder evaluation or production
latency, and no measured experiment automatically changes product defaults.
