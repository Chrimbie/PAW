"""PAW v3.0 data: CIFAR-10, grayscale 16x16, standardized.

Downloads the official Python batches directly from Toronto
(~170 MB) into ./data on first run, or uses PAW_CIFAR_DATA.
No torchvision dependency. prepare() returns five binary
tasks: T0={0,1} T1={2,3} T2={4,5} T3={6,7} T4={8,9}, each a
tuple (Xtr, Xte, ytr_binary, yte_binary, ytr_global,
yte_global) with 256-dim standardized features.
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "PAW_CIFAR_DATA",
    os.path.join(os.path.dirname(_HERE), "data"))

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def ensure_data():
    batches = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if os.path.isdir(batches):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    print(f"CIFAR-10 data not found at {batches}; "
          f"downloading (~170 MB) ...", flush=True)
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(DATA_DIR)
    print("download + extract done", flush=True)


def load_cifar():
    def load_batch(path):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"])

    Xs, ys = [], []
    for i in range(1, 6):
        Xb, yb = load_batch(os.path.join(
            DATA_DIR, "cifar-10-batches-py", f"data_batch_{i}"))
        Xs.append(Xb)
        ys.append(yb)
    Xtr = np.vstack(Xs)
    ytr = np.concatenate(ys)
    Xte, yte = load_batch(os.path.join(DATA_DIR, "cifar-10-batches-py",
                                       "test_batch"))
    return Xtr, ytr, Xte, yte


def to_gray16(X):
    img = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    gray = img.mean(axis=1)
    small = gray.reshape(-1, 16, 2, 16, 2).mean(axis=(2, 4))
    return small.reshape(len(X), -1)


def prepare(include_raw=False):
    Xtr_raw, ytr, Xte_raw, yte = load_cifar()
    raw_train, raw_test = to_gray16(Xtr_raw), to_gray16(Xte_raw)
    mu, sd = raw_train.mean(axis=0), raw_train.std(axis=0) + 1e-8
    Xtr, Xte = (raw_train - mu) / sd, (raw_test - mu) / sd
    tasks = []
    for classes in TASKS:
        remap = {c: i for i, c in enumerate(classes)}
        mtr = np.isin(ytr, classes)
        mte = np.isin(yte, classes)
        task = (Xtr[mtr], Xte[mte],
                np.array([remap[v] for v in ytr[mtr]]),
                np.array([remap[v] for v in yte[mte]]),
                ytr[mtr], yte[mte])
        if include_raw:
            task += (raw_train[mtr], raw_test[mte])
        tasks.append(task)
    return tasks
