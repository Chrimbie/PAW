PAW — Scaleton bundle (best runnable PAW implementation)
===========================================================
The Scaleton is Christopher's minimal PAW build: a fixed-topology
skeleton whose weights live in ReflexStore (disk), a health monitor
as the write-back trigger. No GN, no LaGS.

Run:  python3 paw_scaleton.py     (needs numpy only)

Files:
  paw_scaleton.py          entry point — 3 sequential linear tasks,
                           weights paged through ReflexStore
  paw_core_placeholder.py  the core (see caveat below)
  paw_health.py            health monitor (dead/saturation pressure)
  reflex_optimized_10_mfi4.py  ReflexStore, Christopher's file, unmodified
  paw_scaleton_spiral.py + paw_spiral_probe.py   spiral-task variant
  paw_scaleton_regions.py  regions variant

CAVEAT — READ FIRST:
The core in this bundle is an explicitly labeled PLACEHOLDER. It
implements Christopher's spec exactly (input -> hidden -> gate ->
hidden -> gate -> output, entirely linear, scalar-blend gates), but
it is Mia's stand-in: no file containing his original 4-neuron /
2-gate core was ever found in the workspace. Every result from this
bundle inherits that caveat.

Verified Oct 28, 2026 (fresh copy, re-run):
  3 tasks, 13 write-backs, resets = 0
  stored-weights MSE per task: A 0.3420, B 0.1202, C 2.2743
  (the zero-resets property holds; task C's MSE is honestly weak —
  a linear chain on a random task, not a solved benchmark)

The fuller Oct 2 assembly (LaGS baseline, GN wiring, continual
experiment) lives in workspace/goals/paw-punch-above-weight/
hidden_files/ — paw_baseline.py, paw_lags.py, paw_gn_wiring.py,
paw_continual.py — if you want the more complete version rather
than the minimal validated one.
