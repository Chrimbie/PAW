"""CPU-only smoke tests and strict failure cases; no CIFAR-10 claim."""
import sys
from pathlib import Path
import tempfile
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from task_gate_v2 import (TaskGateV2, train_gate, predict_gate, state_numpy,
                          _fingerprint, choose_gate, save_gate, load_gate)


class MemoryStore:
    def __init__(self): self.records={}
    def put_task_snapshot(self,k,v): self.records[k]=v
    def get_task_snapshot(self,k):return self.records.get(k)
    def flush(self):pass


def run():
    raw,y=load_digits(return_X_y=True)
    a=(raw/16).reshape(-1,8,8).astype(np.float32)
    # 16x16 input expected by PAW; digits only serve as a local integration check.
    a=np.repeat(np.repeat(a,2,axis=1),2,axis=2).reshape(-1,256)
    tr,te=train_test_split(np.arange(len(a)),random_state=419,test_size=.3,stratify=y)
    mu=a[tr].mean(axis=0)
    sd=a[tr].std(axis=0)+.05
    xt, xv=(a[tr]-mu)/sd,(a[te]-mu)/sd
    yt, yv=(y[tr]//2).astype(np.int64),(y[te]//2).astype(np.int64)
    net,diag=train_gate(xt,yt,xv,yv,epochs=5,batch_size=128,
                        device='cpu',augment=False)
    before=predict_gate(net,xv,device='cpu')
    acc=float(np.mean(before==yv))
    assert acc>0.70,(acc,diag)
    store=MemoryStore()
    assert _fingerprint(state_numpy(net))
    save_gate(store,net,selected='taskgate_v2',validation_gate_joint=acc,
              validation_existing_joint=0.3,diagnostic=diag,
              expert_fingerprints={'T0':'x'})
    loaded=load_gate(store,expert_fingerprints={'T0':'x'})
    after=predict_gate(loaded,xv,device='cpu')
    assert np.array_equal(before,after)
    try:
        load_gate(store,expert_fingerprints={'T0':'different'})
        raise AssertionError('should refuse changed expert weights')
    except ValueError: pass
    state=store.records['task-gate-v2']
    state['weights']['head.4.weight'][0,0]+=0.1
    try:
        load_gate(store,expert_fingerprints={'T0':'x'})
        raise AssertionError('should detect mutated state')
    except ValueError: pass
    try:
        train_gate(xt[:50],yt[:50],xt[:50],yt[:50],epochs=1,device='cpu')
        raise AssertionError('should reject overlap')
    except ValueError: pass
    classes=np.arange(10).reshape(5,2)
    l=np.zeros((10,5,2),dtype=np.float64)
    true=np.arange(10)
    right=np.arange(10)//2
    for i in range(10): l[i,right[i],i%2]=3
    scores=choose_gate(right,np.zeros(10,dtype=int),l,true,task_classes=classes)
    assert scores['selected']=='taskgate_v2' and scores['validation_gate_joint']==1.
    scores=choose_gate(np.zeros(10,dtype=int),right,l,true,task_classes=classes)
    assert scores['selected']=='existing'
    print('TASKGATE_CPU_SMOKE_PASS',{'digits_router_accuracy':round(acc,4),
         'test_samples':len(xv),'trained_parameters':diag['params'],
         'restored_predictions_identical':True,'fails_closed_on_weight_change':True,
         'rejects_fit_validation_overlap':True,'validation_gating':True},flush=True)

if __name__=='__main__':run()