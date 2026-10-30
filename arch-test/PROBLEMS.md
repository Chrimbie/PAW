# Problems with the PAW network — full audit (2026-10-30)

Architecture under audit (Christopher's, confirmed 2026-10-30): an MLP with
ReflexStore augmentation; learned material is weights, stored as chunks;
offload on activation; pull when needed; goal = no saturated or dead neurons.

What is NOT a problem (measured): offload fidelity is exact (oracle pull
matches per-task upper bounds); same-task chunk reuse across architectures
works (chunk test); live neurons stay fresh by construction when reset.

## 1. Measured problems (from our tests)

1. **Pull fails.** Centroid routing gets 0.70–0.81 of rows to the right task;
   routed accuracy (0.698/0.803/0.724) falls below the plain sequential
   baseline on A and B. The architecture currently underperforms an ordinary
   net on old tasks. Oracle pull is perfect, so the gap is purely retrieval.
2. **No cross-task transfer.** Chunk test Exp B: iris chunks on a new task
   with the same inputs scored 0.778 vs 0.833 random. The store is a memory,
   not a generalization engine. New tasks must train fresh neurons; old
   chunks do not help them.
3. **Saturation prevention is unproven.** Offload's live net showed fewer
   saturated units (0.5 vs 3.5), but it trained on one task against the
   baseline's three. No equal-load evidence that offload-on-activation
   actually prevents saturation or death over long continual runs. The
   architecture's central claim is still unverified.
4. **The plain alternative is currently better.** A sequential shared-hidden
   net beats the architecture on old-task retention today. Until pull is
   fixed, the store adds complexity without a measured win.

## 2. Mechanism holes (underspecified parts)

5. **The offload trigger is undefined.** Tested as task-boundary offload, not
   literal per-activation offload. Literal per-fire offload-and-clear would
   erase learning mid-training. "Learned enough to offload" has no criterion:
   too early gives fragmentary chunks, too late means saturation already
   happened. The mechanism's central trigger is hand-waved.
6. **Post-offload neuron state is unresolved.** Clear vs keep weights is
   still open. Cleared, the net loses live capability for that task until a
   pull; kept, saturation is not prevented. The central loop has an open
   branch.
7. **Dead neurons are untouched by the mechanism.** Offload fires on
   activation. A dead neuron never activates, so it never offloads, so it
   sits dead forever. The architecture's stated goal includes dead neurons,
   but its mechanism cannot reach them. Dead handling needs a separate,
   unspecified mechanism (re-seeding or replacement).
8. **The pull trigger is the unsolved familiarity problem.** "When needed"
   has no definition. Low confidence is miscalibrated (Test N showed
   confidently-wrong joint agreement); unfamiliarity detection failed in
   Tests L/M. Pull-when-needed inherits the entire unsolved
   selection/rejection problem from Tests A–O.
9. **No feedback from pull outcomes.** When a pull succeeds or fails, nothing
   updates: not chunk quality, not routing, not future pulls. The system
   does not improve with use.
10. **No update path for stored chunks.** Pulled chunks are frozen snapshots.
    If new data arrives within an old task, there is no defined way to
    update that task's chunks short of retraining and re-offloading.

## 3. Structural limits

11. **Deep-layer chunks are not portable.** A first-layer chunk lives in raw
    input space, so it transfers. A deeper-layer chunk's inputs are the
    previous layer's activations of its own net; dropped into another net
    without that context, it is meaningless. "Any net" holds for shallow nets
    and first layers only. Untested for deep nets, likely false there.
12. **Heads are coupled to tasks.** The output head (label space) travels
    with the chunks, so pulling chunks without knowing the task is useless,
    and knowing the task is exactly the routing problem. Head selection is
    the routing problem restated.
13. **Preprocessing travels with chunks.** Chunks were learned in
    standardized space; reuse requires identical preprocessing. Under
    distribution shift, stored chunks go stale, with no adaptation mechanism.
14. **Train/inference asymmetry.** Training uses live neurons; inference may
    use pulled chunks. The system is never trained end-to-end with retrieval
    in the loop, so pulled-chunk behavior at test time is untested during
    training.

## 4. Scale and robustness

15. **Unbounded store growth + crowding.** Every task offloads neurons plus
    a head, forever. Retrieval cost grows with store size, and routing
    accuracy degrades as more tasks crowd the space — the same centroid
    crowding that broke Test F at 96 sets. No pruning, merging, forgetting,
    or compression policy.
16. **Retrieval cost and latency.** Pull-on-need at inference means store
    queries and chunk loads on the inference path, from a disk layer. No
    budget, caching policy, or latency measurement.
17. **Single point of failure.** Live neurons are cleared after offload, so
    the store holds all learned material. If it corrupts or is lost, the net
    knows nothing. No redundancy, backup, or corruption detection policy
    for the store.
18. **No graceful degradation on wrong pull.** A misrouted pull gives a
    confidently wrong answer in the wrong label space — worse than no
    answer. The pull has no defined abstain/unknown state (v2.0 had one;
    this architecture does not).
19. **No measured advantage over saving old models.** Oracle pull equals a
    per-task model ensemble. The architecture needs a resource argument
    (fewer live parameters, less compute, less memory) that has not been
    made or measured. Until then, "store + pull" is an ensemble with extra
    steps.
20. **"Any net" is narrower than stated.** Chunks carry input size; a 64-dim
    chunk cannot serve a net with different input dims. Cross-input-space
    portability needs an adapter that does not exist. The honest claim is:
    any net with the same input space and compatible layer position.

## Bottom line

The shape is right and storage is solved. The problems cluster in three
places: **pull** (the measured failure and the inherited selection problem),
**mechanism details** (trigger, clear-vs-keep, dead neurons, feedback), and
**scale/robustness** (growth, crowding, single point of failure, no measured
advantage). Pull is the first optimization target because oracle pull is
already perfect — every point of routing improvement goes straight to the
final number.
