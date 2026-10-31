# Independent video reference timing and explicit order

Status: active; owned by backend, implements spec0008 without product migration.

Keep feature inference, independent references, query/temporal composition and
final original-video reads separate. Frozen Charades source/subject identities
precede pixel acquisition; only original pixels enter VideoMAE and SigLIP. Public
ZIP64 Range acquisition selects exact bounded entries, never a whole-archive
fallback. Archive metadata/scripts are data; upstream code is not executed.

Blind4s/2s windows preserve16 distinct ordered frames. Low-rate source failures
motivated an opt-in nearest-integer sampling stride before any model inference;
ceil remains the old default and both failed attempts remain recorded. New output
byte bounds are opt-in too. Neither change silently rewrites existing artifacts.

Temporal sequence is an explicit ordered2..4-step operation over same-resource
candidate evidence, distinct from common-interval intersection. Validate all
identities/filters before bounded joins; preserve each original support and fail
on excess rather than truncating internally. Native action classification and
single-frame cosine remain separate score spaces and expose only ID rankings.

Model failures retain all queries/events in denominators. Resource recall cannot
stand for event localization: union proposal coverage, raw/clipped IoU and step
order are separately scored. Oracle joins use human intervals only and never
enter a real model index. No-answer results are reported without a learned reject
threshold or open-world claim. Two actual inference runs, fixed held-out gates
and packaging/compatibility checks remain required before declaring acceptance.
