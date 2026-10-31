"""One-multiply-accumulate-unit replay of a million-parameter trained dense model.
Neurons' original unchanged parameters are stored individually in PAW's
checksummed ChunkStore-compatible adapter backed by unmodified ReflexStore.
"""
import os, json, tempfile, gc, time
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from paw_minimal.store import ChunkStore
from benchmark import single_linear

torch.set_num_threads(2);torch.manual_seed(77)
X,y=load_digits(return_X_y=True)
X=X.astype('float32')/16.0
tr,te=train_test_split(np.arange(len(X)),stratify=y,test_size=.2,random_state=42)
a,b=X[tr],X[te];ya,yb=y[tr],y[te]
net=nn.Sequential(nn.Linear(64,1024),nn.ReLU(),nn.Linear(1024,1024),nn.ReLU(),nn.Linear(1024,10))
params=sum(p.numel() for p in net.parameters())
optim=torch.optim.Adam(net.parameters(),lr=.0015)
start=time.perf_counter()
for epoch in range(15):
    for ids in np.array_split(np.random.default_rng(2026+epoch).permutation(len(a)),12):
        optim.zero_grad(set_to_none=True)
        loss=F.cross_entropy(net(torch.as_tensor(a[ids])),torch.as_tensor(ya[ids],dtype=torch.long))
        loss.backward();optim.step()
train_s=time.perf_counter()-start
net.eval()
with torch.no_grad():
    t=time.perf_counter();original=net(torch.as_tensor(b)).detach().numpy(); ref_s=time.perf_counter()-t
with tempfile.TemporaryDirectory(prefix='paw_million_') as tmp:
    p=os.path.join(tmp,'million.rflx');store=ChunkStore(p)
    dims=[];written=0
    for li in (0,2,4):
        layer=net[li]
        dims.append(layer.out_features)
        for j in range(layer.out_features):
            w=layer.weight[j].detach().cpu().numpy().copy()
            bias=np.float32(layer.bias[j].detach().item())
            store.put_task_snapshot(f'MLP/L{li}/N{j}',{'weight':w,'bias':bias})
            written+=w.size+1
    store.put_task_snapshot('MLP/plan',{'dims':dims,'version':'single-unit-MLP-v1'})
    assert written==params,(written,params)
    store.flush();size=os.path.getsize(p);store.close()
    del net;gc.collect()
    store=ChunkStore(p)
    # Run every example, sequentially loading one output neuron's weights.
    # Preserve intermediate layer activations; no original model object.
    predictions=[]
    start=time.perf_counter()
    for sample in b:
        h=sample.copy()
        for li,units in zip((0,2,4),dims):
            next_h=np.empty(units,dtype=np.float32)
            for j in range(units):
                rec=store.get_task_snapshot(f'MLP/L{li}/N{j}')
                next_h[j]=single_linear(h,rec['weight'],rec['bias'])
                del rec
            h=np.maximum(next_h,0) if li<4 else next_h
        predictions.append(h)
    seq_s=time.perf_counter()-start
    store.close()
    seq=np.vstack(predictions)
    agree=float(np.mean(original.argmax(1)==seq.argmax(1)))
    mx=float(np.max(np.abs(original-seq)))
    result={'architecture':'64-1024-1024-10','train_n':len(tr),'test_n':len(te),'trained_params':params,
      'stored_params':written,'stored_neuron_chunks':sum(dims),
      'original_accuracy':float(np.mean(original.argmax(1)==yb)),
      'receiver_accuracy':float(np.mean(seq.argmax(1)==yb)),
      'prediction_agreement':agree,'max_abs_logit_error':mx,
      'one_scalar_accumulator_reused':True,'original_model_deleted_before_inference':True,
      'storage_bytes':size,'largest_weight_chunk_bytes':(1024+1)*4,
      'training_seconds':train_s,'source_forward_seconds_360':ref_s,'one_unit_forward_seconds_360':seq_s,'slowdown':seq_s/ref_s}
    assert agree==1. and mx<0.002
    with open(os.path.join(os.path.dirname(__file__),'million_dense_results.json'),'w') as f:json.dump(result,f,indent=2)
    print('MILLION_PASS',json.dumps(result),flush=True)
