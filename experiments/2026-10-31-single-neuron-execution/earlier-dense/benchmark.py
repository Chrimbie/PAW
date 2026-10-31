"""Real-data one-neuron PAW weight-transfer comparison; reproducible locally.
Tests source MLP, PAW's stored one-dot-product vessel, and a truly single
standard neuron (binary sigmoid) on a nonlinearly separable XOR control.
"""
import json, os, warnings, time, tempfile
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from paw_minimal.weight_vessel import PAWWeightVessel

np.set_printoptions(precision=8)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
X,y=load_digits(return_X_y=True)
X=X.astype(np.float64)/16.
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
results=[]
with tempfile.TemporaryDirectory(prefix='paw_one_neuron_') as tmp:
  path=os.path.join(tmp,'transfer.rflx')
  for hidden in ((16,), (64,), (256,), (32,16)):
    source=MLPClassifier(hidden_layer_sizes=hidden,activation='relu',solver='lbfgs',random_state=12,max_iter=400,alpha=0.001)
    start=time.perf_counter();source.fit(Xtr,ytr);train_time=time.perf_counter()-start
    # Direct source layers, not sklearn public predict: to compare identical raw logits
    h=Xte.copy()
    for ww, bb in zip(source.coefs_[:-1],source.intercepts_[:-1]):
      h=np.maximum(h@ww+bb,0.)
    direct=h@source.coefs_[-1]+source.intercepts_[-1]
    ref_pred=source.classes_[direct.argmax(axis=1)]
    # Time the source computation before deleting the original model.
    t=time.perf_counter()
    source_h=Xte.copy()
    for ww,bb in zip(source.coefs_[:-1],source.intercepts_[:-1]):
      source_h=np.maximum(source_h@ww+bb,0.)
    _=source_h@source.coefs_[-1]+source.intercepts_[-1]
    direct_elapsed=time.perf_counter()-t
    classes=source.classes_.copy()
    vessel=PAWWeightVessel(path)
    vessel.save('digits_'+str(hidden),source.coefs_,source.intercepts_,hidden_activation='relu')
    vessel.close()
    fresh=PAWWeightVessel(path)
    snap=fresh.store.get_task_snapshot('digits_'+str(hidden))
    preserve_n=sum(len(weights)+1 for layer in snap['layers'] for weights,bias in layer)
    expected_n=sum(w.size+b.size for w,b in zip(source.coefs_,source.intercepts_))
    assert preserve_n==expected_n
    del source, source_h
    # Reset receiver by reopening disk. No training.
    t=time.perf_counter();receiver=fresh.predict_logits('digits_'+str(hidden),Xte);vessel_elapsed=time.perf_counter()-t
    fresh.close()
    pred=classes[receiver.argmax(1)]
    agreement=float(np.mean(pred==ref_pred));maxdiff=float(np.max(np.abs(direct-receiver)))
    r={'source_architecture':'64-'+'-'.join(map(str,hidden))+'-10','source_params':expected_n,'receiver_reusable_dot_product_units':1,
       'receiver_stored_params':preserve_n,'receiver_stored_neuron_records':sum(hidden)+10,
       'source_accuracy':float(np.mean(ref_pred==yte)),'receiver_accuracy':float(np.mean(pred==yte)),
       'pred_agreement':agreement,'max_abs_logit_error':maxdiff,'train_seconds':round(train_time,4),
       'receiver_seconds_360':round(vessel_elapsed,5),'source_vector_seconds_360':round(direct_elapsed,6),
       'store_bytes_after':os.path.getsize(path)}
    assert agreement==1. and maxdiff<1e-9
    results.append(r)
    print('RESULT',json.dumps(r),flush=True)
  # Distinguish single reusable compute unit from a conventional single sigmoid neuron.
  # XOR on clean heldout corners, repeated with light noise. Test nonlinear function.
  rng=np.random.default_rng(2026)
  def xor_data(n):
    bits=rng.integers(0,2,size=(n,2)); x=bits.astype(float)+rng.normal(0,0.08,size=(n,2))
    return x,np.bitwise_xor(bits[:,0],bits[:,1])
  a,ay=xor_data(3000);b,by=xor_data(1000)
  one=MLPClassifier(hidden_layer_sizes=(),activation='logistic',solver='lbfgs',alpha=0.0001,random_state=1,max_iter=500)
  many=MLPClassifier(hidden_layer_sizes=(16,),activation='tanh',solver='lbfgs',alpha=0.0001,random_state=1,max_iter=500)
  one.fit(a,ay);many.fit(a,ay)
  # Literal zero-retraining parameter transfer to an ordinary 2-input,
  # one-output PyTorch neuron: source layer dimensions do not fit.
  import torch
  target=torch.nn.Linear(2,1)
  try:
    target.load_state_dict({
      'weight':torch.tensor(many.coefs_[0].T,dtype=torch.float32),
      'bias':torch.tensor(many.intercepts_[0],dtype=torch.float32)
    },strict=True)
    literal_transfer='accepted'
  except RuntimeError as e:
    literal_transfer='rejected: '+str(e).splitlines()[0]
  assert literal_transfer.startswith('rejected:')
  xor_r={'problem':'XOR with held-out input noise','ordinary_one_neuron_acc':float(one.score(b,by)),
         'multi_neuron_acc':float(many.score(b,by)),'ordinary_one_neuron_params':sum(z.size for z in one.coefs_+one.intercepts_),
         'multi_neuron_params':sum(z.size for z in many.coefs_+many.intercepts_),
         'literal_direct_weight_transfer_to_one_neuron':literal_transfer}
  print('XOR_CONTROL',json.dumps(xor_r),flush=True)
  payload={'real_data':'sklearn digits 8x8; 80/20 stratified, random_state 42','train_samples':len(Xtr),'test_samples':len(Xte),
           'weights_unchanged':True,'source_deleted_before_replay':True,'actual_reflex_store':True,'experimental_receiver_kind':'one reusable computational dot product, sequential source topology',
           'scaling':results,'single_fixed_neuron_control':xor_r,'caveat':'single reusable compute unit is NOT an ordinary single-neuron architecture; all source neuron values and graph retained'}
  (Path(__file__).resolve().parent/'results.json').write_text(json.dumps(payload,indent=2))
