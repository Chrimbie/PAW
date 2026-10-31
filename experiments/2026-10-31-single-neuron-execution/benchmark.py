"""PAW: trained CNNs re-evaluated by one reusable logical neuron.

Real handwritten-digit data. Every original conv/dense destination neuron's
weights are persisted as separate checksummed PAW ChunkStore records and
replayed using a single scalar multiply-accumulate loop at a time. Graph
operations (batch normalization, ReLU, maxpool, residual addition) remain.
No source model is retained during receiver inference. No receiver training.
"""
import gc, hashlib, json, os, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from numba import njit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from paw_minimal.store import ChunkStore

# Suppress inadvertent multithreaded CPU execution in the reference harness.
torch.set_num_threads(2)
np.random.seed(42)
torch.manual_seed(42)

@njit(cache=True)
def single_conv(x,w,bias,pad):
    # One scalar weighted-sum accumulator, reused at every output location.
    ci,hi,wi=x.shape
    channels,kh,kw=w.shape
    oh=hi+2*pad-kh+1; ow=wi+2*pad-kw+1
    out=np.zeros((oh,ow),dtype=np.float32)
    for row in range(oh):
        for col in range(ow):
            accum=np.float32(bias)
            for i in range(ci):
                for kr in range(kh):
                    for kc in range(kw):
                        r=row+kr-pad; c=col+kc-pad
                        if 0<=r<hi and 0<=c<wi:
                            accum=np.float32(accum+x[i,r,c]*w[i,kr,kc])
            out[row,col]=accum
    return out

@njit(cache=True)
def single_linear(x,w,bias):
    accum=np.float32(bias)
    for i in range(len(x)):
        accum=np.float32(accum+x[i]*w[i])
    return accum

@njit(cache=True)
def apply_bn(x,gamma,beta,mean,var,eps):
    scale=np.float32(gamma/np.sqrt(var+eps))
    out=np.empty(x.shape,dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i,j]=np.float32(np.float32(x[i,j]-mean)*scale+beta)
    return out

@njit(cache=True)
def do_relu(x):
    out=np.empty(x.shape,dtype=np.float32)
    for c in range(x.shape[0]):
        for r in range(x.shape[1]):
            for j in range(x.shape[2]):
                out[c,r,j]=max(np.float32(0),x[c,r,j])
    return out

@njit(cache=True)
def do_pool(x):
    nc,h,w=x.shape
    out=np.empty((nc,h//2,w//2),dtype=np.float32)
    for c in range(nc):
        for r in range(h//2):
            for col in range(w//2):
                m=np.float32(-1.e30)
                for dr in range(2):
                    for dc in range(2):
                        m=max(m,x[c,r*2+dr,col*2+dc])
                out[c,r,col]=m
    return out

@njit(cache=True)
def do_add(a,b):
    out=np.empty(a.shape,dtype=np.float32)
    for c in range(a.shape[0]):
        for r in range(a.shape[1]):
            for col in range(a.shape[2]):
                out[c,r,col]=np.float32(a[c,r,col]+b[c,r,col])
    return out

class Simple(nn.Module):
    def __init__(self,w1=32,w2=64,extra=None):
        super().__init__()
        self.conv1=nn.Conv2d(1,w1,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(w1)
        self.conv2=nn.Conv2d(w1,w2,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(w2)
        self.has_extra=extra is not None
        if self.has_extra:
            self.conv3=nn.Conv2d(w2,extra,3,padding=1,bias=False)
            self.bn3=nn.BatchNorm2d(extra)
        self.fc=nn.Linear((extra or w2)*4*4,10)
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.bn1(self.conv1(x))),2)
        x=F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2)
        if self.has_extra:
            x=F.relu(self.bn3(self.conv3(x)))
        return self.fc(x.flatten(1))

class Residual(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.conv1=nn.Conv2d(1,width,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(width)
        self.conv2=nn.Conv2d(width,width,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(width)
        self.conv3=nn.Conv2d(width,width,3,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(width)
        self.fc=nn.Linear(width*4*4,10)
    def forward(self,x):
        identity=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(identity)))
        x=self.bn3(self.conv3(x))
        x=F.relu(x+identity)
        x=F.max_pool2d(x,2)
        x=F.max_pool2d(x,2)
        return self.fc(x.flatten(1))

def tape_for(name):
    if name=='PAW_SmallCNN':
        return [('conv','conv1'),('bn','bn1'),('relu',),('pool',),
                ('conv','conv2'),('bn','bn2'),('relu',),('pool',),('flat',),('linear','fc')]
    if name in ('WideCNN','LargeCNN'):
        return [('conv','conv1'),('bn','bn1'),('relu',),('pool',),
                ('conv','conv2'),('bn','bn2'),('relu',),('pool',),
                ('conv','conv3'),('bn','bn3'),('relu',),('flat',),('linear','fc')]
    if name=='ResidualCNN':
        return [('conv','conv1'),('bn','bn1'),('relu',),('remember',),
                ('conv','conv2'),('bn','bn2'),('relu',),
                ('conv','conv3'),('bn','bn3'),('add_saved',),('relu',),
                ('pool',),('pool',),('flat',),('linear','fc')]
    raise ValueError(name)

def sha_arr(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()

def save_model(store, key, model, tape):
    """Persist one output channel/neuron per checksum-protected record."""
    nodes=[]; cnt=0; params=0; largest=0
    for i,node in enumerate(tape):
        if node[0] not in ('conv','bn','linear'):
            nodes.append(node);continue
        module=getattr(model,node[1]);n=len(module.weight)
        for out in range(n):
            items={}
            for field in ('weight','bias','running_mean','running_var'):
                v=getattr(module,field,None)
                if v is not None:
                    items[field]=v[out].detach().cpu().numpy().astype('float32').copy()
            if node[0]=='bn':
                items['eps']=float(module.eps)
            store.put_task_snapshot(f'{key}/layer{i}/neuron{out}',items)
            largest=max(largest,sum(np.asarray(x).nbytes for x in items.values() if isinstance(x,np.ndarray)))
            params+=sum(np.asarray(x).size for fld,x in items.items() if fld in ('weight','bias'))
            cnt+=1
        nodes.append((node[0],i,n))
    store.put_task_snapshot(f'{key}/plan',{'version':'single-unit-cnn-v1','steps':nodes,'source':key,
                       'n_chunks':cnt,'learned_parameters':params,'largest_weight_chunk_bytes':largest})
    store.flush()
    return cnt,params,largest

def serial_forward(store,key,img):
    """Uses only ONE neuron's weights and ONE scalar MAC at a time.

    Other information resident: the graph tape and intermediate activations.
    Earlier weight chunks are released immediately after processing.
    """
    steps=store.get_task_snapshot(f'{key}/plan')['steps']
    x=img.astype('float32').copy();shortcut=None
    for item in steps:
        operation=item[0]
        if operation=='conv':
            _,i,n=item
            sample=store.get_task_snapshot(f'{key}/layer{i}/neuron0')
            # Output size is fixed by kernel size and same padding.
            pad=1
            result=np.empty((n,x.shape[1],x.shape[2]),dtype=np.float32)
            for j in range(n):
                chunk=store.get_task_snapshot(f'{key}/layer{i}/neuron{j}')
                bias=np.float32(chunk.get('bias',0.0))
                result[j]=single_conv(x,chunk['weight'],bias,pad)
                del chunk
            x=result
        elif operation=='bn':
            _,i,n=item
            result=np.empty_like(x)
            for j in range(n):
                chunk=store.get_task_snapshot(f'{key}/layer{i}/neuron{j}')
                result[j]=apply_bn(x[j],np.float32(chunk['weight']),np.float32(chunk['bias']),
                      np.float32(chunk['running_mean']),np.float32(chunk['running_var']),np.float32(chunk['eps']))
                del chunk
            x=result
        elif operation=='relu':x=do_relu(x)
        elif operation=='pool':x=do_pool(x)
        elif operation=='remember':shortcut=x.copy()
        elif operation=='add_saved':x=do_add(x,shortcut);shortcut=None
        elif operation=='flat':x=x.ravel()
        elif operation=='linear':
            _,i,n=item
            result=np.empty(n,dtype='float32')
            for j in range(n):
                chunk=store.get_task_snapshot(f'{key}/layer{i}/neuron{j}')
                result[j]=single_linear(x,chunk['weight'],np.float32(chunk['bias']))
                del chunk
            x=result
        else:raise ValueError(operation)
    return x

def train(model,X,y,epochs=8):
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    data=torch.as_tensor(X,dtype=torch.float32)
    labels=torch.as_tensor(y,dtype=torch.long)
    n=len(X)
    for ep in range(epochs):
        model.train(); rng=np.random.default_rng(51+ep)
        for ids in np.array_split(rng.permutation(n),max(1,int(np.ceil(n/128)))):
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(data[ids]),labels[ids])
            loss.backward();optimizer.step()
    model.eval()

def run():
    X,y=load_digits(return_X_y=True)
    images=(X.astype('float32').reshape(-1,8,8)/16.0)
    images=np.repeat(np.repeat(images,2,axis=1),2,axis=2)[:,None,:,:]
    i_train,i_test=train_test_split(np.arange(len(images)),test_size=0.2,random_state=42,stratify=y)
    Xtr,ytr=images[i_train],y[i_train]
    Xte,yte=images[i_test],y[i_test]
    # Run every sample through source network but only selected subset through
    # slow single-computational-unit receiver, including difficult examples.
    results=[]
    with tempfile.TemporaryDirectory(prefix='paw_single_unit_cnn_') as folder:
      for name,build in [('PAW_SmallCNN',lambda:Simple(32,64)),
                         ('WideCNN',lambda:Simple(64,128,128)),
                         ('ResidualCNN',lambda:Residual(32))]:
        torch.manual_seed(100)
        source=build()
        start=time.perf_counter();train(source,Xtr,ytr);train_sec=time.perf_counter()-start
        with torch.no_grad():
            t=time.perf_counter(); logits=source(torch.as_tensor(Xte,dtype=torch.float32)).numpy();
            vector_time=time.perf_counter()-t
        pred=logits.argmax(axis=1)
        # Select all 360 examples: accuracy and agreement are directly comparable.
        classes=int((pred==yte).sum())
        stored_n=sum(p.numel() for p in source.parameters())
        path=os.path.join(folder,name+'.rflx')
        store=ChunkStore(path)
        chunks,counted,largest=save_model(store,name,source,tape_for(name))
        assert counted==stored_n,(name,counted,stored_n)
        store.close();del source;gc.collect()
        store=ChunkStore(path)
        t=time.perf_counter()
        # Warm up JIT compilation before timing.
        warm=serial_forward(store,name,Xte[0]);
        t=time.perf_counter()
        serial=np.vstack([serial_forward(store,name,row) for row in Xte])
        elapsed=time.perf_counter()-t
        store.close()
        dpred=serial.argmax(axis=1)
        agree=float(np.mean(dpred==pred))
        discrepancy=float(np.max(np.abs(serial-logits)))
        reldiff=float(np.max(np.abs(serial-logits)/(1+np.abs(logits))))
        result={'architecture':name,'dataset':'sklearn digits enlarged to 16x16',
            'train_n':len(Xtr),'test_n':len(Xte),'source_params':stored_n,
            'original_test_accuracy':round(classes/len(Xte),6),
            'one_unit_test_accuracy':round(float(np.mean(dpred==yte)),6),
            'prediction_agreement':round(agree,6),
            'max_abs_logit_error':discrepancy,'max_relative_like_logit_error':reldiff,
            'neurons_serialized':chunks,'active_dot_product_units':1,
            'largest_weight_chunk_bytes':largest,
            'PAW_store_bytes':os.path.getsize(path),
            'source_forward_seconds_360':round(vector_time,5),
            'receiver_forward_seconds_360':round(elapsed,5),
            'slowdown_ratio':round(elapsed/vector_time,2),
            'source_train_seconds':round(train_sec,3),
            'source_deleted_before_replay':True,'stored_all_trainable_params':True}
        assert agree==1.,f'{name}: {agree} agreement: max error {discrepancy}'
        assert discrepancy<0.002,(name,discrepancy)
        results.append(result)
        print('RESULT',json.dumps(result),flush=True)
    payload={'execution':'single sequential multiply-accumulate neuron; source graph preserved as stored execution tape',
        'storage':'PAW compatible ChunkStore+ReflexStore; checksummed per-neuron records',
        'simultaneous_model_weights_in_receiver':'only one neuron chunk; graph and intermediate activations also resident',
        'limitation':'the physical CPU executes general instructions and intermediate activations still require memory; this is not literally a single biological/artificial hardware neuron or model compression',
        'results':results}
    out=Path(__file__).with_name('results.json')
    out.write_text(json.dumps(payload,indent=2))
    print('PASS_ALL',len(results),flush=True)

if __name__=='__main__':run()
