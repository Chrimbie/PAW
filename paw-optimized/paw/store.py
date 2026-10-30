"""ChunkStore: PAW's disk layer on top of Christopher's ReflexStore.

The ReflexStore implementation itself (paw/reflex_optimized.py) is his
file, copied byte-identical, not modified. This wrapper adds PAW's chunk
semantics on top of it:

- A chunk is a dict (numpy arrays + metadata) serialized with pickle.
- Keys are u64 content hashes (first 8 bytes of SHA-256 of the payload).
  Identical payloads map to the same key, so re-storing unchanged material
  is a no-op instead of store growth (change-only snapshots).
- Task snapshots (a whole task's hidden weights, biases, head, centroid,
  classes) are single records keyed by hash(("task", task_id)). Pull loads
  a task snapshot by task id — no separate manifest file needed.
- Intervention chunks (single units reset during training) are stored
  under their content-hash keys as preserved material. In this build they
  are archival: pull uses task snapshots. Per-neuron reload from
  intervention chunks is the documented next step, not built here.

Pickle is used only for PAW's own data in its own store files.
"""

import hashlib
import pickle

from .reflex_optimized import ReflexStore

_TASK_PREFIX = b"task:"


def _key_for(payload: bytes) -> int:
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _task_key(task_id) -> int:
    return _key_for(_TASK_PREFIX + str(task_id).encode())


class ChunkStore:
    def __init__(self, path):
        self.store = ReflexStore(path)
        self.path = path

    def put_chunk(self, chunk: dict) -> int:
        payload = pickle.dumps(chunk, protocol=4)
        key = _key_for(payload)
        self.store.put(key, payload)
        return key

    def get_chunk(self, key: int):
        raw = self.store.get(key)
        if raw is None:
            return None
        return pickle.loads(raw)

    def put_task_snapshot(self, task_id, snapshot: dict) -> int:
        payload = pickle.dumps(snapshot, protocol=4)
        key = _task_key(task_id)
        self.store.put(key, payload)
        return key

    def get_task_snapshot(self, task_id):
        raw = self.store.get(_task_key(task_id))
        if raw is None:
            return None
        return pickle.loads(raw)

    def put_batch_chunks(self, chunks):
        items = {}
        for chunk in chunks:
            payload = pickle.dumps(chunk, protocol=4)
            items[_key_for(payload)] = payload
        if items:
            self.store.put_batch(items)
        return list(items.keys())

    def flush(self):
        self.store.flush()

    def finalize(self):
        self.store.finalize()

    def close(self):
        self.store.close()
