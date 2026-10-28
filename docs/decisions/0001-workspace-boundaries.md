# ADR 0001: One Repository, Independent Runtimes

Status: accepted. Date: 2026-10-28. Spec: `0001-agentic-workspace`.

The selected projects share a local AI/GPU learning workflow but have incompatible
runtime and trust requirements. Consolidate source and governance in Lentica while
keeping application, example, model-service, GPU-lab, and learning dependencies
separate. Keep existing backend/frontend paths to preserve imports and commands.

Alternatives: separate repositories duplicate contributor rules and hinder common
refactors; one Python environment causes CUDA/Torch/GUI/provider dependencies to
be installed for users who only need local document search. Submodules add checkout
and permission complexity without useful shared history for these source trees.

Consequences: root CI uses CPU/mock checks and a workspace manifest; GPU and Windows
checks remain opt-in. Importing a module never opts a user into remote generation
or GUI control. Original directories are retained until their users/services have
explicitly switched. No cloud connector is automatically enabled in Lentica.
