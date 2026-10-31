"""PAW v3.0 demo: the completed system on the hard task.

Sequential class-incremental CIFAR-10, same 5 tasks of 2
classes, same grayscale 16x16 data, final score 10-class
accuracy with NO task ID. Tasks arrive one at a time; after
each task its TRAIN arrays are replaced with None, so old
training data cannot be used later. Everything is learned
through PAWSystem (paw/engine.py) and evaluated from models
REOPENED from the store.

Modes measured on the same stored system:
  - keyed: task known (continuity reference).
  - oracle: perfect routing (the ceiling).
  - centroid: no-feedback routing, least surprising net.
  - ensemble-maxprob / ensemble-energy: no-feedback routing
    from concatenated logits, training-free.
  - feedback: labeled probes (16 per draw, 50 draws per
    true task, drawn from that task's test images), scored
    by cross-entropy. Ungated argmin + gated (margin +
    agreement, budget inf). Feedback joint is computed
    exactly: per draw, the selected net classifies that
    task's full test set (a wrong selection scores 0 on it);
    gated abstentions fall back to centroid routing for
    that task's images. No number is estimated from
    selection accuracy alone.
  - health: dead conv2 channel fraction + mean |activation|
    per task, diagnostic only.

MEASUREMENT STATEMENT: v3.0 is the completed system, not a
claim that routing is solved. The no-feedback gap found in
the standalone run is carried in honestly: centroid and
ensemble are measured here, and if they land near chance
again, that stands. The remaining routing fix on the table
is a feature bank (stored real feature vectors as
architectural rehearsal material) — it touches the raw-data
line and is NOT built here. Feedback selection is the
working selector; keyed pull is exact.

Run: python demo.py [--quick]
Writes paw_v3_results.json next to this script.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from paw.data import TASKS, ensure_data, prepare
from paw.engine import (PAWSystem, cnn_logits, joint_from_task_pred,
                        per_task_accuracy, task_of_class,
                        to_tensor_1ch)
from paw.models import SmallCNN

STORE_PATH = os.path.join(_HERE, "paw_v3_store.rflx")
EPOCHS = 5
BATCH = 128
LR = 0.001
DRAWS_PER_TASK = 50
PROBES_PER_DRAW = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smoke check: 1 epoch, subsampled tasks")
    ap.add_argument("--prototypes", action="store_true",
                    help="experimental raw-cosine class summaries; "
                         "compare to original routes on the same stored CNNs")
    ap.add_argument("--store-path", default=STORE_PATH,
                    help="path to the benchmark store (default: paw_v3_store.rflx)")
    ap.add_argument("--reset-store", action="store_true",
                    help="explicitly DELETE existing benchmark store and its "
                         "backup before training (destructive)")
    args = ap.parse_args()
    store_path = os.path.abspath(args.store_path)
    existing = [store_path + suffix for suffix in ("", ".bak", ".compact")
                if os.path.exists(store_path + suffix)]
    if existing and not args.reset_store:
        ap.error("existing PAW store found. Refusing to overwrite learned "
                 "knowledge. Supply a new --store-path or explicitly use "
                 "--reset-store (DELETES the existing store and backup).")
    if args.reset_store:
        for file_path in existing:
            os.remove(file_path)
    epochs = 1 if args.quick else EPOCHS
    print(f"epochs/model={epochs} quick={args.quick} "
          f"draws/task={DRAWS_PER_TASK} "
          f"probes/draw={PROBES_PER_DRAW}", flush=True)

    ensure_data()
    tasks = prepare(include_raw=args.prototypes)
    if args.quick:
        # All arrays remain index-aligned, including optional raw views.
        tasks = [tuple(a[:2000] if i % 2 == 0 else a[:500]
                       for i, a in enumerate(task))
                 for task in tasks]
    print(f"task sizes: {[(len(t[0]), len(t[1])) for t in tasks]}",
          flush=True)

    paw = PAWSystem(store_path)
    t0 = time.perf_counter()
    for ti, task in enumerate(tasks):
        Xtr, Xte, ytr, yte, ytrg, yteg = task[:6]
        print(f"=== task T{ti} arrives (sequential window) ===",
              flush=True)
        fit = paw.learn_task(
            f"T{ti}", Xtr, ytr, Xte, yte, epochs,
            batch=BATCH, lr=LR,
            router_raw_Xtr=task[6] if args.prototypes else None)
        paw.store.put_manifest({
            "paw_cnn_tasks": [f"T{k}" for k in range(ti + 1)]})
        paw.store.flush()
        print(f"cnn T{ti} fit (torch): {fit:.4f} -> stored in PAW",
              flush=True)
        # Enforce the sequential rule: this task's TRAIN data is
        # dropped now. Test arrays stay for final scoring.
        tasks[ti] = ((None, Xte, None, yte, None, yteg,
                      None, task[7]) if args.prototypes
                     else (None, Xte, None, yte, None, yteg))
    train_time = time.perf_counter() - t0
    paw.close()

    # ---------- evaluation from REOPENED store models ----------
    paw = PAWSystem(store_path)
    print(f"manifest: {paw.store.get_manifest()}", flush=True)
    models, centroids = paw.load_all()

    Xte_all = np.vstack([t[1] for t in tasks])
    yte_glob_all = np.concatenate([t[5] for t in tasks])
    yte_task_true = np.array([task_of_class(c) for c in yte_glob_all])
    Xte_1ch = to_tensor_1ch(Xte_all)

    # Keyed binary (continuity) + oracle (perfect routing).
    keyed_binary = []
    for ti, classes in enumerate(TASKS):
        mask = np.isin(yte_glob_all, classes)
        logits = cnn_logits(models[ti], Xte_1ch[mask])
        bin_pred = logits.argmax(axis=1)
        pred_glob = np.array([classes[b] for b in bin_pred])
        keyed_binary.append(
            float((pred_glob == yte_glob_all[mask]).mean()))
    keyed_mean = float(np.mean(keyed_binary))
    pred_oracle = joint_from_task_pred(models, yte_task_true, Xte_1ch,
                                       yte_glob_all)
    joint_oracle = float((pred_oracle == yte_glob_all).mean())

    # No-feedback modes.
    arms = {}
    for name, task_pred in (
            ("centroid", paw.route_centroid(models, centroids, Xte_1ch)),
            ("ensemble_maxprob",
             paw.route_ensemble(models, Xte_1ch, mode="maxprob")),
            ("ensemble_energy",
             paw.route_ensemble(models, Xte_1ch, mode="energy"))):
        pred = joint_from_task_pred(models, task_pred, Xte_1ch,
                                    yte_glob_all)
        arms[name] = {
            "routing_accuracy": float(
                (task_pred == yte_task_true).mean()),
            "routed_joint": float((pred == yte_glob_all).mean()),
            "routed_joint_per_task": per_task_accuracy(
                pred, yte_glob_all),
        }
        print(f"{name}: routing {arms[name]['routing_accuracy']:.4f} "
              f"joint {arms[name]['routed_joint']:.4f}", flush=True)
    if args.prototypes:
        profiles = paw.load_prototypes()  # reopening exercises persisted store
        raw_test_all = np.vstack([t[7] for t in tasks])
        task_pred = paw.route_prototypes(profiles, raw_test_all)
        pred = joint_from_task_pred(models, task_pred, Xte_1ch, yte_glob_all)
        arms["prototype_raw_cos_k12"] = {
            "routing_accuracy": float((task_pred == yte_task_true).mean()),
            "routed_joint": float((pred == yte_glob_all).mean()),
            "routed_joint_per_task": per_task_accuracy(pred, yte_glob_all),
            "prototype_bytes": int(sum(
                p["centers"].nbytes for p in profiles.values())),
        }
        print("prototype_raw_cos_k12: routing "
              f"{arms['prototype_raw_cos_k12']['routing_accuracy']:.4f} "
              f"joint {arms['prototype_raw_cos_k12']['routed_joint']:.4f}",
              flush=True)
    centroid_per_task = arms["centroid"]["routed_joint_per_task"]

    # Feedback mode: draws of labeled probes per true task.
    # Per-draw selections are retained so the feedback joints
    # below are computed EXACTLY from these same draws: per
    # draw, the selected net classifies that task's full test
    # set (a wrong selection scores 0 on it); a gated
    # abstention falls back to centroid routing for that
    # task's images. Nothing is estimated from selection
    # accuracy alone.
    rng = np.random.default_rng(0)
    fb_ung_correct = 0
    fb_gated_decided = 0
    fb_gated_correct = 0
    fb_total = 0
    ung_sel = {}
    gated_sel = {}
    for ti in range(len(tasks)):
        Xte_t, yte_t = tasks[ti][1], tasks[ti][3]
        ung_list, gated_list = [], []
        for _ in range(DRAWS_PER_TASK):
            idx = rng.choice(len(Xte_t), size=PROBES_PER_DRAW,
                             replace=False)
            probes = to_tensor_1ch(Xte_t[idx])
            one_hot = np.zeros((PROBES_PER_DRAW, 2))
            one_hot[np.arange(PROBES_PER_DRAW), yte_t[idx]] = 1.0
            logits_by_task = {
                tj: {"logits": cnn_logits(m, probes)}
                for tj, m in models.items()}
            sel_ung = paw.feedback_ungated(logits_by_task, one_hot)
            sel_g, _ = paw.feedback_gated(logits_by_task, one_hot,
                                          max_ce=float("inf"))
            ung_list.append(sel_ung)
            gated_list.append(sel_g)
            fb_total += 1
            if sel_ung == ti:
                fb_ung_correct += 1
            if sel_g is not None:
                fb_gated_decided += 1
                if sel_g == ti:
                    fb_gated_correct += 1
        ung_sel[ti] = ung_list
        gated_sel[ti] = gated_list
    ung_acc = fb_ung_correct / fb_total
    gated_cov = fb_gated_decided / fb_total
    gated_acc = (fb_gated_correct / fb_gated_decided
                 if fb_gated_decided else 0.0)
    ung_joint_tasks = []
    gated_joint_tasks = []
    for ti in range(len(tasks)):
        ung_joint_tasks.append(float(np.mean([
            keyed_binary[ti] if s == ti else 0.0
            for s in ung_sel[ti]])))
        gated_joint_tasks.append(float(np.mean([
            keyed_binary[ti] if s == ti else
            (centroid_per_task[f"T{ti}"] if s is None else 0.0)
            for s in gated_sel[ti]])))
    feedback_joint_ungated = float(np.mean(ung_joint_tasks))
    feedback_joint_gated_fallback = float(np.mean(gated_joint_tasks))
    print(f"feedback ungated selection {ung_acc:.4f} | gated "
          f"coverage {gated_cov:.4f} decided accuracy {gated_acc:.4f} "
          f"| feedback joint ungated {feedback_joint_ungated:.4f} "
          f"gated+centroid-fallback {feedback_joint_gated_fallback:.4f}",
          flush=True)

    health = {}
    for ti, model in models.items():
        Xte_t = tasks[ti][1]
        health[f"T{ti}"] = paw.health(model, to_tensor_1ch(Xte_t))
    paw.close()

    result = {
        "system": "PAW v3.0 completed version: sequential learning "
                  "into ChunkStore, routing from reopened models, "
                  "all helpers architectural",
        "protocol": "CIFAR-10 official splits, grayscale 16x16 "
                    "standardized, 5 binary tasks T0={0,1} T1={2,3} "
                    "T2={4,5} T3={6,7} T4={8,9}; FULLY SEQUENTIAL; "
                    "final score is 10-CLASS accuracy with NO task ID",
        "seed": 0, "epochs_per_model": epochs, "batch": BATCH, "lr": LR,
        "quick": args.quick, "prototype_router_enabled": args.prototypes,
        "draws_per_task": DRAWS_PER_TASK,
        "probes_per_draw": PROBES_PER_DRAW,
        "paw_cnn_params": sum(p.numel() for p in
                              SmallCNN(2).parameters()),
        "paw_cnn_keyed_binary": keyed_binary,
        "paw_cnn_keyed_binary_mean": keyed_mean,
        "paw_joint_oracle": joint_oracle,
        "paw_joint_oracle_per_task": per_task_accuracy(pred_oracle,
                                                       yte_glob_all),
        "paw_no_feedback_arms": arms,
        "paw_feedback": {
            "ungated_selection_accuracy": ung_acc,
            "gated_coverage": gated_cov,
            "gated_decided_accuracy": gated_acc,
            "feedback_joint_ungated": feedback_joint_ungated,
            "feedback_joint_gated_centroid_fallback":
                feedback_joint_gated_fallback,
        },
        "paw_health": health,
        "paw_time_s": train_time,
        "history_for_orientation_only": {
            "scaffolded_feature32_routed_joint": 0.5930,
            "scaffolded_feature128_routed_joint": 0.6180,
            "standalone_centroid_routed_joint": 0.2114,
            "standalone_dream_router_routed_joint": 0.2142,
            "previous_resnet_sequential_10way_joint": 0.1861,
            "previous_resnet_fresh_10way_joint": 0.7167,
        },
    }
    print(json.dumps(result, indent=2, default=float))
    out = os.path.join(
        _HERE, "paw_v3_results_prototypes.json" if args.prototypes
        else "paw_v3_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
