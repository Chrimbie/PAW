# PAW: Single reusable compute-unit transfer — real-data benchmark

**Status: PASS for functional equivalence across three convolutional architectures and one 1.126-million-parameter dense architecture.**

## Objective

Train an ordinary multicell neural network on real labeled data, store **all of its learned weights unchanged** as individual destination-neuron records in PAW-compatible ReflexStore-backed storage, delete the source network, reopen the store, and evaluate its **original computation graph sequentially through a single reusable scalar multiply–accumulate (MAC) unit**. No receiver training.

This tests sequential **physical/computational reuse**, not changing the original mathematical graph or compressing its learned parameters. Convolutions, batch normalization, pooling, ReLU, a residual skip, and dense layers are implemented in the experimental receiver. The physical CPU still has many hardware execution units: “one unit” describes the **explicit software execution schedule** with no concurrent original-neuron MACs, not a one-neuron physical chip.

## Setup

- Data: `sklearn.datasets.load_digits`, 8×8 grayscale digits (dense model), nearest-neighbor enlarged to 16×16 for CNNs.
- Split: stratified 80/20 train/test, `random_state=42` = 1,437 training / 360 test images.
- Training: PyTorch Adam, no pretrained weights, no test data during fitting. Each source network trained before serialization.
- Persistence: `ReflexStore` from the PAW project, with a lightweight compatible `ChunkStore` task-snapshot/checksum adapter; each output channel or dense destination neuron gets a separate checksummed record. Batch-normalization parameters and running statistics are preserved. The original source model object is discarded before replay.
- Inference: Numba-jitted scalar inner loops in `benchmark.py` / `million_dense.py`, execution of each original logical unit in layer order. All network weight records retained on disk; only the currently needed record is loaded into the working receiver. Intermediate activations and graph metadata remain resident.
- Exactness: comparison is **same predicted class**, not bitwise equality of floating-point logits. Sub-0.00004 maximum absolute logit difference.

## Results

| Architecture | Source trainable parameters | Test accuracy: source and receiver | Predictions matching | Single-unit time for 360 samples | Relative slowdown (observed) |
|---|---:|---:|---:|---:|---:|
| PAW SmallCNN (10 classes) | 29,162 | 98.61% | 360/360 | 1.51 s | 68× |
| WideCNN | 242,890 | 98.61% | 360/360 | 6.45 s | 127× |
| ResidualCNN | 24,042 | 98.33% | 360/360 | 2.98 s | 71× |
| Dense 64→1024→1024→10 | 1,126,410 | 98.33% | 360/360 | 8.94 s | 1,463× |

**Combined: 1,440/1,440 identical predicted classes.** The original input-to-output network functionality was reproduced for these architectures and held-out examples with no changes to learned parameters. Timings were measured locally on CPU with a Numba sequential implementation and may vary substantially with hardware and measurement noise. These are not GPU-vs-GPU throughput comparisons.

The 1.126M-parameter dense model was saved in 2,058 neuron-chunk records (including output neurons), 5.07 MB of serialized data. Maximum one-weight-chunk size was 4,100 bytes. However, this **does not** mean total resident RAM was 4,100 bytes: the interpreter, graph, input, intermediate activations, and output buffers require additional memory.

An attempted extra-large CNN with ~one million parameters and full serial evaluation over all 360 samples exceeded the local execution time limit; it is **not** recorded as a pass. The million-parameter dense model passed instead.

## Reproduce

Install requirements into an isolated Python environment (Python 3.10+ recommended):

```bash
python -m pip install -r requirements.txt
python benchmark.py
python million_dense.py
```

Inspect `results.json`, `million_dense_results.json` and corresponding console logs. A CPU with adequate RAM is enough; no GPU is needed. `paw_minimal/reflex_optimized.py` is the existing underlying ReflexStore implementation, and `paw_minimal/store.py` is a minimal compatibility adapter (not a full PAW repo checkout).

## What this **does not** prove

1. That a conventional fixed one-neuron network with one fixed parameter set can replace a wide network; the receiver preserves the original model's layer topology, weights and intermediate activations.
2. That the original weights or total stored network memory are compressed or eliminated. Only the *simultaneously active neuron-computation schedule* is reduced to one logical MAC unit.
3. Efficient inference or a speed improvement. The receiver was much slower than the vectorized network.
4. General support for every architecture. In particular, transformer attention, layer normalization, tokenization, and autoregressive caches were not tested here.
5. Novelty of the general idea of serially evaluating a neural network. The contribution tested is this PAW-compatible persisted per-neuron execution prototype, not a claim of computational impossibility overcome.

## Artifacts

- `benchmark.py`: train SmallCNN, WideCNN, ResidualCNN and serially replay the real trained CNN graphs.
- `million_dense.py`: train a 1,126,410-parameter dense network and serially replay it.
- `paw_minimal/`: compatible storage adapter and original ReflexStore implementation.
- `results.json`, `million_dense_results.json`: the completed raw measured results.
- `console_log.txt`, `million_dense_log.txt`: raw script output.
