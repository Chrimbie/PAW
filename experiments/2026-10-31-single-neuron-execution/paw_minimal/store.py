"""ChunkStore from PAW paw-minimal/paw_minimal/store.py — subset unchanged for vessel."""
import hashlib
import pickle
from .reflex_optimized import ReflexStore
_TASK_PREFIX=b'task:'
def _key_for(payload): return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')
def _task_key(task_id): return _key_for(_TASK_PREFIX+str(task_id).encode())
def _sha_key(task_id): return _key_for(b'task-sha:'+str(task_id).encode())
class ChunkStore:
    def __init__(self,path):
        self.store=ReflexStore(path);self.path=path
    def put_task_snapshot(self,task_id,snapshot):
        payload=pickle.dumps(snapshot,protocol=4);key=_task_key(task_id)
        self.store.put(key,payload)
        self.store.put(_sha_key(task_id),hashlib.sha256(payload).digest());return key
    def get_task_snapshot(self,task_id):
        raw=self.store.get(_task_key(task_id))
        if raw is None:return None
        sha=self.store.get(_sha_key(task_id))
        if sha is not None and hashlib.sha256(raw).digest()!=sha:
            raise ValueError(f'checksum mismatch for task {task_id}')
        return pickle.loads(raw)
    def flush(self):self.store.flush()
    def close(self):self.store.close()
