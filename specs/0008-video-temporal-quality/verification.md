# Verification

Latest status (2026-10-31): the overall TODO is paused by the user, and video
inference remains deferred. Spec0009 product integration is separately complete;
the full local regression now passes 359 tests, superseding the pending regression
note below. There is still no completed multi-source video-quality report.
See [current status](../../docs/project-status.md). The following entries retain
their historical execution/goal state, not current authorization to resume.

2026-10-31: spec active, no video-quality implementation/results claimed yet.
Previous goal turn made verified progress: RELATION-1–4 code, two120-source native
controls, actual CPU/validation failures and final323 tests/build13 verified.
This turn's scoped memory search had no relevant hits; repository remains main
with unrelated tracked/untracked work preserved. No subagents or publication.

Public Charades project HTML403, old annotation address403 and guessed renamed
annotation object404 are retained as source-discovery failures. The explicit
public charades/ prefix listing(20 keys max,18 actual) identifies Charades.zip
3519822 bytes and Charades_v1_480.zip16339546533 bytes. Video ZIP64 tail65557
bytes identifies1022502-byte/9849-entry directory. Initial directory read31909
timed out after643560 bytes; exact remaining378942 bytes42755 completed and
directory parsed. No full video archive download. Annotation first55983 timed
out after1879584 bytes; exact remainder1640238 bytes92616 completed. Combined
annotation package size and official single-object ETag/MD5 match; ZIP CRC reads
of explicit metadata succeed. Annotation SHA256
c616913ef79c2ddde06d9c562eae57bb8901d459d7568a0d27bf09cbf33ae866.
Only class/test/README/license texts extracted, no upstream scripts executed.

Annotation design inspected public headers/classes/initial rows. Six exact native
class aliases and annotation-only eligibility yield16 distinct subjects from1863
official test videos/53 subjects. No pixel/model output has been observed for
these sources, and no videos yet acquired. Selection/config freeze must precede
pixel acquisition and inference. New source/order/window/evaluation code, tests,
real quality reports, complete checks and packaging remain required.

Continuation: frozen selection10495060cc616640d23e288b8a2b815f81ed4ada7d0b540cb98a796760edc08c
preceded acquisition of all16 original MP4s. Exact archive-entry Range/ETag/CRC/
size checks succeeded; only28928581 compressed bytes acquired, no16GB full
archive. Anonymous dataset b1ce838ea05d1b7ac80b370dfb907b25f7a3ffe0ba8d149a0e59e75bd4fe9860
verifies all originals without replacement. First source preparation failed due
to reader JSON separator mismatch; corrected the reader without changing the
pre-acquisition identity or references.

Window attempt1 failed on the incorrect VideoFrames.fps attribute; corrected to
frames_per_second. Attempt2 retained13 successful sources, then6.134fps yielded
only11–12 real samples per4s using historical ceil spacing. Attempt3 explicit
floor spacing later exceeded240 samples on9.396fps. Before any prediction,
inspected all16 rate/count headers and chose explicit nearest-integer spacing
(half upward), keeping240-frame/2GiB bounds and the old ceil default. Attempt4
completed16 sources/2579 frames/248 blind windows/512434388 PNG bytes. Actual
reload verified source/frame hashes, configuration/windows and16 distinct ordered
frames in every window. Output video-0008-charades-windows-4; failed directories
1–3 remain. No original changed, no selected source replaced.

Focused tests:20 initial passed, then25 passed in.61s after metric, failure-
denominator, final-byte-read and actual6fps controls. Initial metric test expected
one relevant proposal for a0..4 event, but >=.5 coverage also includes2..6;
corrected fixture expectations to2/3 without changing implementation/thresholds.
These are mechanism tests, not native-model quality evidence.

User explicitly deferred video because CPU is slow. Native run97208 reached
source2, then its owned execution session was interrupted; exit130 confirms
termination. No complete feature/quality report was produced. Further video
inference, single-frame comparison and repeats are deferred. VIDEO-MODEL-1/
VIDEO-EVAL-1 and promotion gates remain unaccepted. Retrieval goal stays active,
prioritizing non-video tasks. Complete new-code checks remain pending.
