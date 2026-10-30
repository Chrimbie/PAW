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
import pickle
import shutil

from .reflex_optimized import ReflexStore

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
        return pickle.loads(raw)

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
        return pickle.loads(raw)

    def backup(self):
        """Copy the live store file to path + '.bak' (one generation).

        Guards against later corruption or loss of the live file.
        Same disk, so this is not protection against disk failure —
        it protects against a bad write, a bad compact, or an
        accidental delete of the live file.
        """
        import os

        if os.path.exists(self.path):
            shutil.copy(self.path, self.path + ".bak")

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
