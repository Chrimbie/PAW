# Saturation prevention — live mechanism report (2026-10-30)

Workstream ordered by Christopher 2026-10-30 ("work on saturation
prevention"). Prior state: task-boundary offload kept live neurons fresh
by construction, but no during-training prevention mechanism had been
tested, and dead neurons had never appeared.

## Design (Mia's, fixed before running)

Variants, checked each epoch on a 256-row probe of train data:
- **none**: baseline, no intervention.
- **reset**: on trigger, snapshot the neuron as a chunk to the store and
  reinitialize it (small random weights, Adam moments zeroed).
- **freeze**: on trigger, snapshot the neuron and freeze it (gradients
  zeroed). It keeps its function but cannot saturate further.
Triggers: tanh saturation = mean |act| > 0.9; death = firing rate < 1%
(ReLU) or act std < 0.05 (tanh proxy).
Setup: 10 sequential binary digit tasks, hidden 12, seeds 0,1.
Success bar: mechanism arm keeps sat/dead below baseline throughout AND
late-task (7–10) learn accuracy within 0.05 of early-task (1–3) accuracy.

## Results (means over 2 seeds)

| arm | early | late | drop | sat trajectory (tasks 1–10) | dead final | chunks |
|-----|-------|------|------|-----------------------------|------------|--------|
| tanh none | 0.991 | 0.987 | 0.004 | 0.5 → 12/12 by task 5 | 0 | 0 |
| tanh reset | 0.991 | 0.991 | -0.001 | stays 0–2 | 0 | 261 |
| tanh freeze | 0.989 | 0.925 | 0.064 | plateaus 8/12 | 0 | 12 |
| relu none | 0.994 | 0.994 | 0.001 | n/a | 0 | 0 |
| relu reset | 0.994 | 0.994 | 0.001 | n/a | 0 | 0 |
| relu freeze | 0.994 | 0.994 | 0.001 | n/a | 0 | 0 |

## Verdict

- **Reset WORKS for tanh saturation.** It meets the pre-set bar: saturation
  held at 0–2 units against the baseline's 12/12, with zero learning cost
  (late accuracy 0.991 = early 0.991). Snapshot + reset on trigger is the
  working prevention mechanism. Cost to note: 261 snapshots over the run
  (~13 per task per seed) — the store grows with every intervention, so
  snapshot deduplication or pruning becomes a real concern at scale.
- **Freeze FAILS.** Late learning drops 0.064, past the 0.05 bar.
  Capacity exhaustion: as units freeze, fewer trainable units remain for
  new tasks. Freeze is the wrong variant.
- **Dead neurons remain untested.** No ReLU unit died in any arm, even
  baseline (dead final 0, zero triggers). Death is hard to produce in
  this regime (easy tasks, Adam, 12-unit hidden). A death-producing setup
  — higher learning rate, hostile initialization, or harder tasks — is
  still needed before any death-prevention claim can be tested.
- **Consistent with the earlier finding:** saturation does not stop new
  learning even unopposed (baseline drop 0.004). What the reset mechanism
  buys is unsaturated live neurons plus a store of snapshots; whether
  that matters beyond retention is still the open question from the
  problem-3 split verdict.

## Open next

1. A death-producing setup (ReLU death on demand), then test reset
   against it.
2. Snapshot growth control (deduplication/pruning) for the reset arm.
3. The trigger question itself: this test used a saturation threshold
   (mean |act| > 0.9), not literal per-activation offload — that
   operationalization is still Christopher's to confirm or replace.
