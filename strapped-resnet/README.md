# PAW strapped to ResNet18 — October 30, 2026

PAW's October 30 reframe in one line: the architecture is a regular
neural net with an augmented memory. These scripts test that claim
at full-fledged scale: ResNet18 (11,169,858 parameters) trained
sequentially on five binary CIFAR tasks, with its full state saved
per task into Christopher's ReflexStore, then pulled back by
different mechanisms.

Tasks (grayscale 16x16 standardized images, replicated to 3
channels): T0 = airplane/automobile, T1 = bird/cat, T2 = deer/dog,
T3 = frog/horse, T4 = ship/truck. Training protocol throughout:
Adam lr 0.001, batch 128, 5 epochs per task, seed 0.

## Versions

- `plain/` — the unstrapped baseline: sequential ResNet18 plus a
  fresh-per-task bound. `resnet18_cifar.py` + its run results.
  Sequential final old-mean 0.418 on the first run (run-to-run
  variation is large: 0.418, 0.5945, 0.6444, 0.5057, 0.6124 across
  five runs — no-pull retention is unstable by construction).
- `v1/` — first strapped build: sequential training + full
  state_dict snapshot per task into ReflexStore + task-keyed
  pull + pixel-MLP router + reopen check. Task-keyed pulled
  old-mean 0.8909 = immediates exactly; router-pulled 0.6759;
  reopen T0 exact. Store 223,788,270 bytes.
- `v2/` — adds checksum companion records, snapshot caching,
  full-data router training (max_iter 500, still unconverged),
  router-confidence abstention at 0.6, and a fit-ID arm (16
  labeled probes per task, 5 draws). Fit-ID 25/25 = 1.0000.
  Router accuracy 0.4183; pulled-router-as-wrong 0.2470 (not a
  regression — v2 abstains on ~64% where v1 forced answers);
  answered-only 0.7377 on ~32.5% coverage.
- `v3/` — adds the feature router: same MLP trained on
  final-trunk 512-d features instead of raw pixels. Routing
  0.5111 (vs pixel 0.4183), pulled 0.3738, answered-only 0.7910
  on ~46.4% coverage, and the feature fit converged within 500
  iterations where the pixel fit never did.
- `v4/` — adds snapshot-ensemble scoring: no training, no
  labels; each row runs through all five snapshots and the task
  is picked by max-prob or energy (identical rules for binary
  heads: max-prob = min-entropy = max-margin). Identification
  only 0.4275/0.4288, but pulled old-mean 0.7502/0.7534 at 100%
  coverage in 1.2 seconds — because misrouted rows are still
  ~66% correct, a regularity with no claimed mechanism yet.
  Ensemble abstention at 0.9 is useless (rates ~0.0015:
  snapshots are nearly always >0.9 confident, even foreign
  ones).

## Routing ladder (old-task means, v4 run)

fit-ID 1.000 (feedback available) > keyed 0.892 (task known) >
ensemble 0.750 (no feedback, no training, 100% coverage) >
feature-router answered-only 0.796 on ~46% coverage
(pulled-as-wrong 0.370) > pixel-router answered-only 0.745 on
~33% coverage (pulled-as-wrong 0.249).

Each version directory is runnable as shipped (script +
`cifar_test.py` + `reflex_optimized.py` + run notes). The
`resnet18_cifar_results.json` and `demo_cifar_results.json`
files bundled in each version are the reference numbers from
the plain-ResNet and PAW-MLP runs; the `paw_resnet18_results*.json`
files are that version's own run results. Run results were
produced on Christopher's PC (CUDA) and verified against his
pasted console output. `reflex_optimized.py` is Christopher's
file, byte-identical everywhere; do not modify it.
