# PAW transferability test — run notes

## What this is

The test of Christopher's transferability claim: train ResNet18,
chunk the data, and an untrained single neuron should get similar
or exact results.

Read the honesty notes at the top of `paw_transfer.py` before the
numbers. The short version: literal weight transfer is impossible
by construction (ResNet weights don't fit one neuron; one neuron
can't express ResNet's function). What transfers is knowledge
through the store, and this script tests exactly that.

- Teacher: ResNet18 trained sequentially on the same 5 binary
  CIFAR tasks (T0 = airplane/automobile ... T4 = ship/truck),
  grayscale 16×16 standardized, Adam lr 0.001, batch 128,
  5 epochs/task, seed 0. Full state_dict snapshot per task into
  Christopher's ReflexStore, with companion SHA-256 records.
- Chunks: per task, one record per train example, in two key
  spaces: `transfer-pixel` (standardized pixels + label) and
  `transfer-feat` (final-trunk 512-d features + label). The final
  trunk is the shared feature space; snapshot trunks would need
  the task id to choose, which is circular.
- Arms:
  - (P) Pixel retrieval, untrained querier: nearest train-pixels
    chunk's label. No trained net anywhere.
  - (F) Feature retrieval, untrained querier: nearest
    final-trunk-feature chunk's label. The querier still trains
    nothing; the features come from the shared trunk.
  - (D) Distillation: a single 256→1 sigmoid neuron trained ONLY
    on the teacher's soft outputs for train inputs (never true
    labels). Control: the identical neuron trained on true
    labels, same init, so the soft-vs-true gap is honest.
- All retrieval runs AFTER finalize + close + reopen, so chunks
  must survive restart to count. k=1 (single chunk answers) and
  k=5 (majority vote) are both reported; Euclidean distance.

## Files

- `paw_transfer.py` — the test. Run this.
- `cifar_test.py` — data loading + task splits (imported).
- `reflex_optimized.py` — Christopher's ReflexStore (imported;
  do not modify).

## Requirements

- Python 3.10+
- `pip install torch numpy scikit-learn` (on a GPU machine, the
  standard torch install includes CUDA; on CPU-only, use the CPU
  wheel from https://pytorch.org/get-started/locally/)

## Run

From inside this folder:

```
python paw_transfer.py
```

- The script prints its device first (`cuda` if a GPU torch build
  is present, else `cpu`) and uses it automatically.
- First run downloads CIFAR-10 automatically (~170 MB, official
  Toronto URL) into `./data` and extracts it. If you already have
  the `cifar-10-batches-py` folder somewhere, set `PAW_CIFAR_DATA`
  to its parent folder instead of downloading.
- Output: printed per-task progress and
  `paw_transfer_results.json` written next to the script, plus
  `paw_transfer_store.rflx` (snapshots + ~100k chunk records,
  expect roughly 350–400 MB).

## What to watch

- Retrieval vs teacher: do untrained-querier numbers land near the
  teacher's ~0.89–0.95, or far below? Feature retrieval should
  beat pixel retrieval clearly if the store's knowledge is in the
  features; pixel retrieval is the literal single-neuron test.
- Distillation soft vs true: if soft ≈ true, the student learned
  the task purely through teacher chunks. If soft trails true
  badly, chunk fidelity (not capacity) is the limit — the control
  neuron shows what one neuron can do with direct labels.
- The "exact" question: retrieval matches the teacher only where
  nearest-chunk agreement holds. A gap between retrieval and
  teacher is expected; the question is how big it is, and whether
  the gap is a chunk problem or a neuron problem.
