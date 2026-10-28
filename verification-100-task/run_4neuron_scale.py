import os,time,json,io,tracemalloc
import numpy as np
from paw_core_placeholder import TinyCore
from paw_health import HealthGuard
from paw_scaleton import serialize,deserialize,pressure_total
from reflex_optimized_10_mfi4 import ReflexStore
ROOT=os.path.dirname(__file__)

def tasks_make(n,seed=11,steps=350):
 r=np.random.default_rng(seed); tasks=[]
 for _ in range(n):
  M=r.normal(size=(2,4));b=r.normal(0,.5,2)
  X=r.normal(size=(steps,4));Y=X@M.T+b+r.normal(0,.05,(steps,2))
  T=r.normal(size=(150,4));V=T@M.T+b
  tasks.append((X,Y,T,V))
 return tasks

def mse(model,X,Y):
 return float(np.mean([(model.forward(x)[0]-y)**2 for x,y in zip(X,Y)]))
def train_paw(core,X,Y):
 for x,y in zip(X,Y):
  core.set_param_items(list(core.propose_update(x,y,lr=.01).values()))
def train_linear(X,Y,lr=.01):
 W=np.zeros((4,2));b=np.zeros(2)
 for x,y in zip(X,Y):
  err=(x@W+b-y);W-=lr*np.outer(x,err);b-=lr*err
 return W,b
def linear_mse(wb,X,Y):
 W,b=wb;return float(np.mean((X@W+b-Y)**2))

out={'note':'Placeholder linear one-scalar-bottleneck PAW; tasks are independent random linear mappings; no MNIST or original PAW core.'}
for n in [10,30,100]:
 tasks=tasks_make(n);path=os.path.join(ROOT,'benchmark_temp.rflx')
 if os.path.exists(path):os.remove(path)
 start=time.perf_counter();store=ReflexStore(path);writes=0;pressure=[];linear_models=[];paw_models=[]
 for i,(X,Y,T,V) in enumerate(tasks):
  core=TinyCore(seed=2000+i);guard=HealthGuard(width=3,window=100);last=0
  for x,y in zip(X,Y):
   _,cache=core.forward(x);guard.observe(cache['h']);core.set_param_items(list(core.propose_update(x,y,lr=.01).values()))
   if pressure_total(guard)-last>=150:
    store.put(i,serialize(core));writes+=1;last=pressure_total(guard)
  store.put(i,serialize(core));writes+=1;pressure.append(pressure_total(guard))
  paw_models.append(core);linear_models.append(train_linear(X,Y))
 store.flush();store.close();elapsed=time.perf_counter()-start
 disk=os.path.getsize(path);store=ReflexStore(path);probe=TinyCore(seed=0)
 paw=[];lin=[]
 for i,(_,_,T,V) in enumerate(tasks):
  deserialize(store.get(i),probe);paw.append(mse(probe,T,V));lin.append(linear_mse(linear_models[i],T,V))
 store.close();os.remove(path)
 out[f'scale_{n}']={'paw_mean_mse':float(np.mean(paw)),'linear_sgd_mean_mse':float(np.mean(lin)),'paw_mse_per_task':paw if n==3 else None,'linear_mse_per_task':lin if n==3 else None,'writes':writes,'pressure_events_total':sum(pressure),'disk_bytes':disk,'elapsed_seconds_paw_including_storage':elapsed,'parameters_per_paw_core':sum(a.size for _,a in probe.param_items())}
 print('scale',n,out[f'scale_{n}'],flush=True)
# Continual learning: single shared model vs per-task stored weights; evaluate old tasks after each new task
T=tasks_make(5,seed=37,steps=450);shared=TinyCore(seed=2);linear_shared=(np.zeros((4,2)),np.zeros(2));history=[]
for i,(X,Y,_,_) in enumerate(T):
 train_paw(shared,X,Y)
 W,b=linear_shared
 for x,y in zip(X,Y):
  e=x@W+b-y;W-=.01*np.outer(x,e);b-=.01*e
 history.append({'after_task':i+1,'shared_paw_mse':[mse(shared,t[2],t[3]) for t in T[:i+1]],'shared_linear_sgd_mse':[linear_mse((W,b),t[2],t[3]) for t in T[:i+1]]})
out['continual_shared_weights']=history
# Fair matched-steps single task and parameter size
out['fairness']={'linear_sgd_train_steps':'same as PAW','linear_parameters':10,'paw_parameters':sum(a.size for _,a in shared.param_items()),'caveat':'PAW has rank-1 linear bottleneck; linear baseline can express rank-2 map. Distinct per-task stored weights avoid interference but consume O(tasks) disk.'}
with open(os.path.join(ROOT,'four_neuron_results.json'),'w') as f:json.dump(out,f,indent=2)
print('CONTINUAL',history,flush=True)
