"""ChunkStore: PAW's disk layer on top of Christopher's ReflexStore.

The ReflexStore implementation itself (paw_minimal/reflex_optimized.py)
is his file, copied byte-identical, not modified. This wrapper adds
PAW's chunk semantics: dicts serialized with pickle, keys are u64
content hashes (identical payloads dedup), task snapshots are single
records keyed by hash(("task", task_id)). Same wrapper as the optimized
build; the minimal build uses it for per-step offload slots.

Pickle is used only for PAW's own data in its own store files.

Checksum behavior (added 2026-10-30 after the break battery found
silent corruption): put_task_snapshot also writes a companion record
holding SHA-256 of the snapshot payload; get_task_snapshot verifies
it and raises ValueError on mismatch instead of returning wrong
weights silently. Records written before this fix have no companion
record and load unverified (backward compatible). The manifest (task
list) is stored the same way and verified on load. Christopher's
ReflexStore file itself is untouched — all of this lives in this
wrapper.
"""

import hashlib
import io
import pickle
import shutil

from .reflex_optimized import ReflexStore


class _RestrictedUnpickler(pickle.Unpickler):
    """Allowlist unpickler for store payloads.

    Snapshots and manifests legitimately contain only plain containers,
    numbers, strings, bytes, and numpy arrays/scalars. Any other global
    (arbitrary classes, functions, os/system-style callables) is refused
    instead of executed. This closes the arbitrary-code path that plain
    pickle.loads leaves open; it does not make a store 'trusted' on its
    own, and the allowlist must stay this tight.
    """

    _ALLOWED = {
        ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"),
        ("builtins", "set"), ("builtins", "frozenset"),
        ("builtins", "str"), ("builtins", "int"), ("builtins", "float"),
        ("builtins", "bool"), ("builtins", "complex"), ("builtins", "bytes"),
        ("builtins", "bytearray"), ("builtins", "slice"), ("builtins", "range"),
        ("copyreg", "__newobj__"), ("copyreg", "__newobj_ex__"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy", "float64"), ("numpy", "float32"), ("numpy", "float16"),
        ("numpy", "int64"), ("numpy", "int32"), ("numpy", "int16"),
        ("numpy", "int8"),
        ("numpy", "uint64"), ("numpy", "uint32"), ("numpy", "uint16"),
        ("numpy", "uint8"),
        ("numpy", "bool_"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"forbidden global in store payload: {module}.{name}")


def _safe_loads(raw):
    return _RestrictedUnpickler(io.BytesIO(raw)).load()

_TASK_PREFIX = b"task:"


def _key_for(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _task_key(task_id) -> int:
    return _key_for(_TASK_PREFIX + str(task_id).encode())


_SHA_PREFIX = b"task-sha:"
_MANIFEST_PREFIX = b"manifest"


def _sha_key(task_id) -> int:
    return _key_for(_SHA_PREFIX + str(task_id).encode())


def _manifest_key() -> int:
    return _key_for(_MANIFEST_PREFIX)


def _manifest_sha_key() -> int:
    return _key_for(_MANIFEST_PREFIX + b":sha")


class ChunkStore:
    def __init__(self, path):
        self.store = ReflexStore(path)
        self.path = path

    def put_task_snapshot(self, task_id, snapshot: dict) -> int:
        payload = pickle.dumps(snapshot, protocol=4)
        key = _task_key(task_id)
        self.store.put(key, payload)
        self.store.put(_sha_key(task_id),
                       hashlib.sha256(payload).digest())
        return key

    def get_task_snapshot(self, task_id):
        raw = self.store.get(_task_key(task_id))
        if raw is None:
            return None
        sha = self.store.get(_sha_key(task_id))
        if sha is not None and hashlib.sha256(raw).digest() != sha:
            raise ValueError(
                f"checksum mismatch for task {task_id}: stored "
                f"snapshot corrupted, refusing to load it")
        return _safe_loads(raw)

    def put_manifest(self, manifest: dict) -> int:
        payload = pickle.dumps(manifest, protocol=4)
        key = _manifest_key()
        self.store.put(key, payload)
        self.store.put(_manifest_sha_key(),
                       hashlib.sha256(payload).digest())
        return key

    def get_manifest(self):
        raw = self.store.get(_manifest_key())
        if raw is None:
            return None
        sha = self.store.get(_manifest_sha_key())
        if sha is not None and hashlib.sha256(raw).digest() != sha:
            raise ValueError(
                "manifest checksum mismatch: stored manifest "
                "corrupted, refusing to load it")
        return _safe_loads(raw)

    def backup(self):
        """Atomically refresh a verified one-generation backup.

        PAW 3.0 uses 'paw_cnn_tasks' in its manifest. Verify every listed
        snapshot (including checksum companions) before replacing the last
        good backup. Missing snapshots and corrupt manifests block backup.
        Existing legacy task families remain supported. Returns False
        when there is no manifest yet (fresh unused store), True on copy.
        """
        import os

        # Callers must flush buffered records before backing up.
        if not os.path.exists(self.path):
            return False
        manifest = self.get_manifest()
        if manifest is None:
            # An empty/new store has nothing to preserve yet.
            return False
        if not isinstance(manifest, dict):
            raise ValueError("invalid task manifest: expected dict")
        tasks = set()
        for key in ("paw_cnn_tasks", "unified_tasks", "vessel_tasks"):
            ids = manifest.get(key, [])
            if not isinstance(ids, (list, tuple)):
                raise ValueError(f"invalid manifest list: {key}")
            tasks.update(ids)
        for task in tasks:
            if self.get_task_snapshot(task) is None:
                raise ValueError(
                    f"missing snapshot for listed task {task}; "
                    "refusing to overwrite a valid backup")
        tmp = self.path + ".bak.tmp"
        try:
            shutil.copyfile(self.path, tmp)
            os.replace(tmp, self.path + ".bak")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return True

    def flush(self):
        self.store.flush()

    def finalize(self):
        self.store.finalize()

    def close(self):
        self.store.close()

    def compact(self, task_ids):
        """Rewrite the store keeping only the latest record per task.

        ReflexStore is append-only: every offload step appends a record
        and the index keeps the latest, so a 9000-write run leaves 9000
        records in the file for 3 live slots. Compaction reads the live
        snapshots, writes them to a fresh file, and atomically replaces
        the old file. Returns (old_bytes, new_bytes, n_live_slots).
        Post-compaction pulls are identical: the snapshots are the same
        dicts, only the dead history is gone.
        """
        import os

        if not os.path.exists(self.path):
            return 0, 0, 0
        snapshots = {}
        for task in task_ids:
            snap = self.get_task_snapshot(task)
            if snap is not None:
                snapshots[task] = snap
        manifest = self.get_manifest()
        old_bytes = os.path.getsize(self.path)
        if not snapshots:
            return old_bytes, old_bytes, 0
        tmp_path = self.path + ".compact"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        fresh = ChunkStore(tmp_path)
        for task, snap in snapshots.items():
            fresh.put_task_snapshot(task, snap)
        if manifest is not None:
            fresh.put_manifest(manifest)
        fresh.flush()
        fresh.close()
        self.store.close()
        # Back up the pre-compact file (full history) before replacing.
        shutil.copy(self.path, self.path + ".bak")
        os.replace(tmp_path, self.path)
        self.store = ReflexStore(self.path)
        new_bytes = os.path.getsize(self.path)
        return old_bytes, new_bytes, len(snapshots)
