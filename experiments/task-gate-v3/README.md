# PAW TaskGate RGB optimization — controlled experiment

**Status: implemented and local unit-tested; no CIFAR-10 GPU result yet.**

PAW TaskGate v2 reached **68.03% joint 10-class accuracy** and **75.13% correct routing** on the official CIFAR-10 test set with ResNet-18 binary experts, using grayscale 16×16 images. The expert models scored 88.00% joint accuracy when the correct task was supplied. The likely information bottleneck in the gate is loss of original RGB channels and spatial resolution.

## Proposed optimization

1. Keep the **existing five trained binary experts unchanged**, still consuming standardized grayscale 16×16 inputs. No changes to PAW's persistent learning architecture.
2. Give *only the optional task router* the original RGB 32×32 images, preserving color and spatial detail. TaskGateRGB retains TaskGate v2's five-way output, convolutional feature blocks, and modest capacity, with an adaptive pool to keep the same dense head size.
3. Evaluate two predeclared training strategies on **identical expert weights and splits**: `rgb16` (16 epochs, horizontal flips) and `rgb32_crop` (32 epochs, horizontal flips, padded random crop, 0.05 label smoothing). Refit baseline grayscale TaskGate v2 (16 epochs) on the same data and held-out validation set.
4. Select the strongest RGB candidate *only on the 600 train-derived validation images*. Promote it *only if* it beats the active existing gate's **joint validation accuracy by at least 2 percentage points**. Score the already-selected candidate once on all 10,000 official test images. Do not select hyperparameters from official test results.
5. Persist and reopen the RGB gate, verify full weights fingerprint and original expert fingerprints, and assert that 600 validation predictions remain exactly equal after reopening. The receiver remains independently recoverable, with no expert retraining.

The gate trains on 49,400 official CIFAR-10 training images; the remaining 600 training images are held out for the validation-based selection decision. Test data are only read when final evaluation begins. The benchmark verifies exact class-aligned order between raw RGB and grayscale task partitions, and checks RGB train/validation overlap by image-content fingerprint.

## Reproduction

With a Windows x64 GPU runner checkout and a separate PAW checkout on the experimental branch:

```powershell
python benchmarks/paw_taskgate_rgb_cifar10.py --quick
python benchmarks/paw_taskgate_rgb_cifar10.py
```

The script can also take `--architectures SmallCNN WideCNN ResNet18`, but the default is **ResNet18 only** because the two RGB variants receive the identical training dataset for every expert architecture. Full mode is the accuracy benchmark; quick mode is only a smoke test.

### Important limits

- The RGB router requires raw 32×32 RGB images at inference; it **cannot** replace the existing grayscale-only `predict_auto(X)` interface without explicit input-path integration. No automatic changes to main PAW are made.
- This is a supervised, five-class image classifier trained on labeled task membership, and is *not* an unsupervised universal task discovery mechanism.
- More input detail adds preprocessing, memory, and convolutional inference cost; reported timing measures are hardware-dependent.
- The original 68.03% score is a previous matched-protocol run, **not** a result for the new RGB candidate.
- A single validation split and run may be noisy. Successful promotion should be followed by multi-seed replication and a matched data-budget baseline before integration.

## Files

- `task_gate_rgb.py`: opt-in trained five-way RGB gate with persistent storage compatibility.
- `test_task_gate_rgb.py`: local CPU unit tests for shapes, crop/flip augmentation, data-overlap rejection, model save/reopen consistency, weight corruption and expert identity mismatch.
- `run_rgb_cifar10.py`: runner benchmark source, installed as `benchmarks/paw_taskgate_rgb_cifar10.py`.