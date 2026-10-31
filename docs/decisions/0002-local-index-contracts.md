# ADR 0002: Explicit Local Index and Task Contracts

Status: accepted. Spec: 0002-index-task-consistency.

Retain SQLite and the modular monolith. Add source/stage traversal cursors,
versioned additive migrations, task ownership leases and explicit encoder spaces.
Keep separate task handlers and retrieval/scoring boundaries so later connectors
or indexes can be added without changing queue ownership or API contracts.

Pause drains work; restart/worker loss recovers expired leases rather than
reassigning every running row. Unknown semantic encoder provenance is excluded
from ranking, without deleting existing records or automatically reindexing files.
Document lexical retrieval caches a corpus revision independently of the query.
Database triggers enforce active-request uniqueness under SQLite's serialized
writes, including older SQLite runtimes without partial indexes. OCR exact-text
retrieval reads stored text independently of semantic encoder compatibility.
The V2 endpoint may return document citations through an additive field; existing
photo routes retain their response shape.

An external queue, vector service and lake storage would add operational cost
without current measurements. They remain replaceable implementation choices.
