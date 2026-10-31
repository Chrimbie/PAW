import tempfile
from pathlib import Path
import sys
import numpy as np
import torch

from task_gate_rgb import TaskGateRGB, _images, _check_disjoint, _crop_flip, train_rgb, predict_tasks, save, load

class MemStore:
    def __init__(self):self.d={}
    def put_task_snapshot(self,k,v):self.d[k]=v
    def get_task_snapshot(self,k):return self.d.get(k)
    def flush(self):pass


def test_all():
    rng=np.random.default_rng(18)
    # Image color is task-defining in this synthetic test. Trains quickly on CPU.
    x=rng.integers(0,256,size=(80,3,32,32),dtype=np.uint8)
    y=np.repeat(np.arange(5),16)
    x[:,0,:,:] = np.minimum(255, y[:,None,None]*48+10).astype(np.uint8)
    xval=rng.integers(0,256,size=(30,3,32,32),dtype=np.uint8)
    vy=np.repeat(np.arange(5),6)
    xval[:,0,:,:]=np.minimum(255,vy[:,None,None]*48+10).astype(np.uint8)
    assert _images(x.reshape(80,3072)).shape==(80,3,32,32)
    assert _crop_flip(torch.rand(4,3,32,32),crop=True,flip=True).shape==(4,3,32,32)
    try:_check_disjoint(x,np.concatenate([xval,x[:1]]))
    except ValueError:pass
    else:raise AssertionError('overlap detection missing')
    model,mean,scale,diag=train_rgb(x,y,xval,vy,epochs=1,device='cpu',batch=32,seed=18)
    before=predict_tasks(model,xval,mean=mean,scale=scale)
    store=MemStore()
    save(store,model,mean,scale,selection=True,diagnostic=diag)
    restored,m,s=load(store,expert_fingerprints={})
    try:load(store,expert_fingerprints={'T0':'changed'})
    except ValueError:pass
    else:raise AssertionError('expert fingerprint mismatch accepted')
    assert np.array_equal(before,predict_tasks(restored,xval,mean=m,scale=s))
    store.d['task-gate-rgb32-v1']['weights']['head.4.bias'][0]+=1
    try:load(store)
    except ValueError:pass
    else:raise AssertionError('weight corruption accepted')
    print('PASS RGB shapes, crop, persistence, and corruption rejection')

if __name__=='__main__':test_all()