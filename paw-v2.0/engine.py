"""PAW 2: fixed four-neuron engine plus addressable non-gradient Reflex evidence.

Every task retains independent core/Adam state, immutable observations and an
externally sampled active index. Input-only retrieval ABSTAINS on ambiguity;
feedback can disambiguate. No performance/semantic guarantee is implied.
"""
import hashlib
import json
import math
import random
import statistics
from collections import OrderedDict
import config
from core import Core,Adam,PAWError,finite,new_params,project
from encoding import encode,projection,digest,distance,squared,ENCODER_VERSION
from store import Store


def task_name(s):
    if not isinstance(s,str) or not (1<=len(s)<=100) or not s.strip() or any(ord(c)<32 or ord(c)==127 for c in s):
        raise PAWError('Invalid task name')
    return s

def percentile(vals,f):
    a=sorted(vals)
    if not a:return 0
    return a[min(len(a)-1,int((len(a)-1)*f))]

def mean(values):return sum(values)/len(values)

def normalized_input(m,raw_projection):
    return max(-config.CORE_INPUT_CLIP,min(config.CORE_INPUT_CLIP,
               (raw_projection-m['x_mean'])/m['x_scale']))

def class_index(m,y):
    return m['classes'].index(y)

def y_to_norm(m,y):
    value=class_index(m,y) if m['mode']=='classification' else finite(y,'target')
    return (value-m['y_mean'])/m['y_scale']

def core_raw(m,p):
    y=Core(m['params']).forward(normalized_input(m,p))[0]*m['y_scale']+m['y_mean']
    return finite(y,'neural output')

def target_prediction(m,p):
    v=core_raw(m,p)
    if m['mode']=='classification':
        return m['classes'][max(0,min(len(m['classes'])-1,round(v)))]
    return v

def memory_predict(m,vec,anchors,k=5):
    if not anchors:return None,None
    near=sorted(((squared(vec,a[0]),i) for i,a in enumerate(anchors)))[:k]
    close=math.sqrt(near[0][0])
    if m['mode']=='classification':
        votes={}
        for d,i in near:
            label=anchors[i][1]
            votes[label]=votes.get(label,0.0)+1/(math.sqrt(d)+0.03)
        # Stable tie-breaking, independent of insertion and index order.
        label=min(votes,key=lambda l:(-votes[l],str(l)))
        return label,close
    n=d=0.0
    for q,i in near:
        w=1/(math.sqrt(q)+0.03)
        n+=w*float(anchors[i][1]);d+=w
    return n/d,close

def predict_from(m,vec,anchors):
    neural=target_prediction(m,projection(vec))
    memory,closest=memory_predict(m,vec,anchors)
    selected=m.get('method','core')
    if selected=='memory' and memory is not None:answer=memory;source='reflex'
    elif selected=='baseline':
        answer=m['classes'][0] if m['mode']=='classification' else m['y_mean'];source='baseline'
    else:answer=neural;source='core'
    return {'answer':answer,'source':source,'closest_distance':closest,'core':neural,'memory':memory}

def objective(m,pred,y):
    if m['mode']=='classification':return 0.0 if pred==y else 1.0
    return (pred-y)**2

class Engine:
    def __init__(self,store=None):
        self.store=store or Store()
        self._model_cache=None
        self._anchor_cache=OrderedDict()
        self._external_version=None
    def _invalidate_cache(self):
        self._model_cache=None
        self._anchor_cache.clear()
        self._external_version=None
    def _check_cache(self):
        # SQLite data_version advances for commits by OTHER connections.
        # Changes by this engine are invalidated by Engine.train.
        v=self.store.db.execute('PRAGMA data_version').fetchone()[0]
        if self._external_version!=v:
            self._invalidate_cache()
            self._external_version=v
    def _anchors(self,task):
        self._check_cache()
        if task in self._anchor_cache:
            self._anchor_cache.move_to_end(task)
            return self._anchor_cache[task]
        anchors=self.store.anchors(task)
        self._anchor_cache[task]=anchors
        while len(self._anchor_cache)>config.MAX_CACHED_TASKS:
            self._anchor_cache.popitem(last=False)
        return anchors
    def _models(self):
        self._check_cache()
        if self._model_cache is None:self._model_cache=self.store.items()
        return self._model_cache
    def close(self):self.store.close()
    def _new_model(self,task,mode,encoded):
        projs=[p for _,_,p in encoded]
        xm=mean(projs);xs=math.sqrt(mean([(p-xm)**2 for p in projs]))
        ys=[v for _,v,_ in encoded]
        labels=sorted(set(ys)) if mode=='classification' else []
        values=[labels.index(y) if mode=='classification' else y for y in ys]
        ym=mean(values);sd=math.sqrt(mean([(y-ym)**2 for y in values]))
        return {'format':'PAW2','encoder_version':ENCODER_VERSION,'feature_dim':config.FEATURE_DIM,'task':task,'mode':mode,'params':new_params(),
           'adam':Adam.fresh().to_dict(),'x_mean':xm,'x_scale':max(xs,0.1),
           'y_mean':ym,'y_scale':max(sd,0.5),'classes':labels,'method':'core',
           'seen':0,'trained_steps':0,'retention_rejections':0,'repairs':0,'train_calls':0,
           'validation_score':None,'novelty_radius':config.MIN_NOVELTY_RADIUS,'fit_job':None,
           'center':[0.0]*config.FEATURE_DIM,'spread':0.0}
    def _check_task_mode(self,task,mode):
        old=self.store.load(task)
        if old is not None and old['mode']!=mode:
            raise PAWError('Cannot change task type between regression and classification')
        return old
    def _retained_examples(self,task,limit=64):
        return [(json.loads(r['features']),json.loads(r['y']),r['projection']) for r in
                self.store.db.execute('SELECT features,y,projection FROM evidence WHERE task=? ORDER BY stamp LIMIT ?', (task,limit))]
    def _core_error(self,m,records):
        if not records:return 0.0
        return mean([objective(m,target_prediction(m,p),y) for _,y,p in records])
    def _health_repair(self,m,xs):
        if len(xs)<8 or max(xs)-min(xs)<1e-3:return
        health=Core(m['params']).health(xs)
        if not health['dead_hidden_neurons']:return
        p=m['params'][:]
        # Revive only when the incoming scalar signal actually varies.
        if health['hidden_variances'][0]<config.DEAD_VARIANCE:
            p[0]=max(abs(p[0]),0.12)*(1 if p[0]>=0 else -1)
            p[1]=max(abs(p[1]),0.12)*(1 if p[1]>=0 else -1)
        if health['hidden_variances'][1]<config.DEAD_VARIANCE:
            p[2]=max(abs(p[2]),0.12)*(1 if p[2]>=0 else -1)
        m['params']=project(p)
        m['repairs']+=1
    def _recompute_radius(self,task):
        anchors=self.store.anchors(task)
        if len(anchors)<3:return config.MIN_NOVELTY_RADIUS
        vectors=[a[0] for a in anchors]
        # Bounded O(64*min(384, count)) at batch boundaries, not inference.
        sample=vectors[::max(1,len(vectors)//64)][:64]
        nn=[]
        for v in sample:
            ds=sorted(squared(v,w) for w in vectors)
            # self distance is first; duplicates may produce zero.
            if len(ds)>1:nn.append(math.sqrt(ds[1]))
        return max(config.MIN_NOVELTY_RADIUS,min(3.0,
                   percentile(nn,0.90)*config.NOVELTY_MULTIPLIER))
    def _fit(self,task,training,mode,steps_override=None):
        """Restartable, bounded core optimizer over new data + replay anchors.

        Progress and Adam moments are saved in the same DB transaction. A
        rerun of the same data resumes the original objective, not a new one.
        """
        ids=sorted(row[3] for row in training)
        signature=digest(ids)
        with self.store.transaction():
            m=self.store.load(task)
            if m is None:raise PAWError('Task not found during optimization')
            pending=m['fit_job']
            if pending is None:
                goal=steps_override if steps_override is not None else (config.CORE_FIT_STEPS_FIRST if m['train_calls']==0 else config.CORE_FIT_STEPS_CONTINUE)
                pending={'signature':signature,'goal':goal,'done':0,
                         'old_params':m['params'][:], 'old_adam':json.loads(json.dumps(m['adam'])),
                         'old_steps':m['trained_steps']}
                m['fit_job']=pending
                self.store.save(m)
            elif pending['signature']!=signature:
                raise PAWError('Incomplete optimization requires the SAME dataset for restart')
        # Use all new examples in deterministic sample order; no validation examples.
        # A fixed-size active replay set protects prior input regions.
        while True:
            with self.store.transaction():
                m=self.store.load(task)
                job=m['fit_job']
                if job is None:break
                start=job['done']
                if start>=job['goal']:
                    m['fit_job']=None
                    self.store.save(m)
                    break
                anchors=self.store.anchors(task)
                # Enduring records are accessible, while only a bounded sample
                # is loaded into resident working memory.
                replay=anchors[:config.REPLAY_COUNT] if m['train_calls']>0 else []
                pools=[(p,y) for _,y,p,_,_ in training]
                pools.extend((p,y) for _,y,p in replay)
                if not pools:raise PAWError('No examples for optimizer')
                p=m['params'][:]; opt=Adam.from_dict(m['adam'])
                done=min(job['goal'],start+config.FIT_CHECKPOINT_EVERY)
                for i in range(start,done):
                    r=random.Random(config.SEED+i+len(training)*37)
                    samples=[pools[r.randrange(len(pools))] for _ in range(min(config.BATCH_SIZE,len(pools)))]
                    pairs=[(normalized_input(m,pr),y_to_norm(m,y)) for pr,y in samples]
                    _,g=Core(p).loss_grad(pairs)
                    p=opt.update(p,g)
                m['params']=p;m['adam']=opt.to_dict();m['trained_steps']=opt.step_count
                # Hold the original task's performance within the configured
                # guard band. A rejected update leaves external samples intact.
                if m['train_calls']>0:
                    protected=self._retained_examples(task)
                    reference=dict(m)
                    reference['params']=job['old_params'];reference['adam']=job['old_adam']
                    old_loss=self._core_error(reference,protected)
                    new_loss=self._core_error(m,protected)
                    if protected and new_loss>old_loss*(1+config.ADAPTATION_RETENTION_TOLERANCE)+0.001:
                        m['params']=job['old_params'];m['adam']=job['old_adam']
                        m['trained_steps']=job['old_steps']
                        m['retention_rejections']+=1
                        job['done']=job['goal']
                    else:job['done']=done
                else:job['done']=done
                m['fit_job']=job
                self.store.save(m)

    def train(self,task,rows,mode,progress=None,streaming=False):
        task=task_name(task)
        if len(rows)<4:raise PAWError('At least four examples required')
        if len(rows)>config.MAX_ROWS_PER_IMPORT:raise PAWError('Too many examples')
        if mode not in ('regression','classification'):raise PAWError('Invalid learning mode')
        self._invalidate_cache()
        records=[]
        for x,y in rows:
            if mode=='regression':y=finite(y,'target')
            else:
                if not isinstance(y,str) or len(y)>config.MAX_LABEL_CHARS:raise PAWError('Invalid label')
            v=encode(x);records.append((v,y,projection(v),digest([x,y]),digest(x)))
        rng=random.Random(config.SEED)
        rng.shuffle(records)
        # Split each call before any evidence goes to memory. Only train rows are archived.
        validation_n=max(1,round(len(records)*0.2))
        val=records[:validation_n];training=records[validation_n:]
        starting_model=self.store.load(task)
        fresh=self._check_task_mode(task,mode) is None
        processed=skipped=0
        for offset in range(0,len(training),config.BATCH_SIZE):
            batch=training[offset:offset+config.BATCH_SIZE]
            with self.store.transaction():
                m=self.store.load(task)
                if m is None:
                    if self.store.db.execute('SELECT count(*) FROM models').fetchone()[0]>=config.MAX_TASKS:
                        raise PAWError('Task count limit exceeded')
                    m=self._new_model(task,mode,[(a,b,c) for a,b,c,_,_ in training])
                    self.store.save(m)
                # Only distinct observations cause new learning.
                newly=[]
                stamp=m['seen']
                for vec,y,p,uid,xhash in batch:
                    if self.store.append(task,uid,xhash,vec,y,p,stamp+1):
                        stamp+=1
                        newly.append((vec,y,p))
                    else:skipped+=1
                if not newly:continue
                old_guard=self._retained_examples(task)[:64]
                # Prior model snapshot remains available for rejection.
                before=dict(m);before['params']=m['params'][:]
                before['adam']=json.loads(json.dumps(m['adam']))
                if mode=='classification':
                    for _,y,_ in newly:
                        if y not in m['classes']:m['classes'].append(y)
                # Bounded old-task rehearsal; external evidence remains authoritative.
                previous=self.store.anchors(task)
                rehearsal=previous[:config.REPLAY_COUNT]
                updates=[(p,y) for _,y,p in newly]
                for _,y,p in rehearsal:
                    if mode=='classification' and y not in m['classes']:continue
                    updates.append((p,y))
                pairs=[(normalized_input(m,p),y_to_norm(m,y)) for p,y in updates]
                params=m['params'][:];opt=Adam.from_dict(m['adam'])
                # Single bounded working batch; persisted only at transaction boundary.
                passes=3 if m['seen']==0 else 1
                for j in range(passes):
                    order=pairs[:]
                    random.Random(config.SEED+m['trained_steps']+j).shuffle(order)
                    _,grad=Core(params).loss_grad(order)
                    params=opt.update(params,grad)
                m['params']=params;m['adam']=opt.to_dict();m['trained_steps']=opt.step_count
                xs=[normalized_input(m,p) for _,_,p in newly]
                self._health_repair(m,xs)
                # Guard on FIRST unmodified observations, excluding current batch.
                # It prevents destructive updates to the old core; evidence is still retained.
                if before['seen']>0 and old_guard:
                    old_error=self._core_error(before,old_guard)
                    new_error=self._core_error(m,old_guard)
                    threshold=old_error*(1+config.ADAPTATION_RETENTION_TOLERANCE)+0.001
                    if new_error>threshold:
                        m['params']=before['params'];m['adam']=before['adam']
                        m['trained_steps']=before['trained_steps']
                        m['retention_rejections']+=1
                n=m['seen']
                for v,_,_ in newly:
                    n+=1
                    old_center=m['center']
                    m['center']=[c+(val-c)/n for c,val in zip(old_center,v)]
                m['seen']=stamp
                self.store.save(m)
                self.store.audit(task,'batch',{'new':len(newly),'skipped':len(batch)-len(newly),'steps':m['trained_steps'],'guard_rejects':m['retention_rejections']})
                processed+=len(newly)
            if progress:progress(min(offset+len(batch),len(training)),len(training))
        if streaming and processed==0 and (self.store.load(task) or {}).get('fit_job') is None:
            m=self.store.load(task)
            anchors=self.store.anchors(task)
            return {'task':task,'created':False,'new_evidence':0,'duplicates':skipped,
                    'total_evidence':m['seen'],'active_anchors':len(anchors),'core_neurons':4,
                    'trainable_core_parameters':8,'method':m['method'],
                    'validation_losses':{'last_saved':m['validation_score']},
                    'core_health':Core(m['params']).health([normalized_input(m,p) for _,_,p,_,_ in training]),
                    'retention_rejections':m['retention_rejections'],'novelty_radius':m['novelty_radius']}
        if not streaming or processed>0 or (self.store.load(task) or {}).get('fit_job') is not None:
            steps=(config.STREAM_FIRST_FIT_STEPS if fresh else config.STREAM_CORE_FIT_STEPS) if streaming else None
            self._fit(task,training,mode,steps_override=steps)
        with self.store.transaction():
            m=self.store.load(task)
            if m is None:raise PAWError('No nonduplicate training examples')
            anchors=self.store.anchors(task)
            m['novelty_radius']=self._recompute_radius(task)
            # Evaluate modes on unseen validation records; never archive them.
            methods=['core','memory','baseline']
            scores={}
            for method in methods:
                m['method']=method
                results=[predict_from(m,vec,anchors)['answer'] for vec,_,_,_,_ in val]
                scores[method]=mean([objective(m,pred,y) for pred,(_,y,_,_,_) in zip(results,val)])
            # Preference for neural core on ties to avoid unneeded memory reliance.
            chosen=min(methods,key=lambda k:(scores[k],methods.index(k)))
            # Never change to a new retrieval policy that degrades prior evidence
            # beyond the configured retention tolerance. It is a local measured
            # protection, not a theorem about unseen past distributions.
            if starting_model is not None:
                retained=self._retained_examples(task)
                before_anchors=self.store.anchors(task)
                old_method=starting_model['method']
                old_predictions=[predict_from(starting_model,v,before_anchors)['answer'] for v,_,_ in retained]
                old_risk=mean([objective(starting_model,p,y) for p,(_,y,_) in zip(old_predictions,retained)]) if retained else 0
                m['method']=chosen
                new_predictions=[predict_from(m,v,before_anchors)['answer'] for v,_,_ in retained]
                new_risk=mean([objective(m,p,y) for p,(_,y,_) in zip(new_predictions,retained)]) if retained else 0
                if retained and new_risk>old_risk*(1+config.ADAPTATION_RETENTION_TOLERANCE)+0.001:
                    chosen=old_method
                    m['retention_rejections']+=1
            m['method']=chosen
            m['validation_score']=scores[m['method']]
            m['train_calls']+=1
            self.store.save(m)
            self.store.audit(task,'train',{'new_rows':processed,'duplicate_rows':skipped,'validation':scores,'method':m['method']})
        return {'task':task,'created':fresh,'new_evidence':processed,'duplicates':skipped,
                'total_evidence':m['seen'],'active_anchors':len(anchors),'core_neurons':4,
                'trainable_core_parameters':8,'method':m['method'],'validation_losses':scores,
                'core_health':Core(m['params']).health([normalized_input(m,p) for _,_,p,_,_ in training]),
                'retention_rejections':m['retention_rejections'],'novelty_radius':m['novelty_radius']}
    def train_stream(self,task,path,progress=None):
        from encoding import iter_dataset
        chunk=[];mode=None;total=0;added=0;duplicates=0;last=None
        for x,y,current_mode in iter_dataset(path):
            if mode is None:mode=current_mode
            if current_mode!=mode:raise PAWError('Mixed classification/regression targets in one dataset')
            chunk.append((x,y))
            if len(chunk)>=config.STREAM_CHUNK_ROWS:
                out=self.train(task,chunk,mode,streaming=True)
                total+=len(chunk);added+=out['new_evidence'];duplicates+=out['duplicates'];last=out
                chunk=[]
                if progress:progress(total)
        if chunk:
            # Preserve a minimally useful train/validation split on the tail.
            if len(chunk)<4:
                # These trailing examples must not disappear silently.
                if last is None:raise PAWError('Dataset too small')
                # Simple one-step streaming ingestion (no local holdout).
                tail_new=self._ingest_tail(task,chunk,mode)
                total+=len(chunk);added+=tail_new;duplicates+=len(chunk)-tail_new
            else:
                out=self.train(task,chunk,mode,streaming=True)
                total+=len(chunk);added+=out['new_evidence'];duplicates+=out['duplicates'];last=out
        if last is None:raise PAWError('No full training chunk')
        return {'task':task,'rows_processed':total,'new_evidence':added,
                'duplicates_in_training_splits':duplicates,
                'mode':mode,'last_chunk_validation':last['validation_losses'],
                'core_neurons':4,'store_evidence':self.store.count(task),
                'active_anchors':len(self.store.anchors(task))}

    def _ingest_tail(self,task,rows,mode):
        self._invalidate_cache()
        # Tail still becomes durable evidence but is not falsely counted as
        # having trained the core or undergone validation.
        with self.store.transaction():
            m=self.store.load(task)
            if m is None:raise PAWError('Cannot ingest tail into missing task')
            stamp=m['seen']; old_stamp=stamp
            for x,y in rows:
                v=encode(x);p=projection(v);uid=digest([x,y])
                if self.store.append(task,uid,digest(x),v,y,p,stamp+1):
                    stamp+=1
                    if mode=='classification' and y not in m['classes']:m['classes'].append(y)
                    m['center']=[c+(val-c)/stamp for c,val in zip(m['center'],v)]
            if stamp!=old_stamp:
                m['seen']=stamp
                self.store.save(m)
                self.store.audit(task,'tail-evidence-only',{'rows':len(rows),'new_total':stamp})
            return stamp-old_stamp

    def predict(self,task,x):
        task=task_name(task)
        m=self.store.load(task)
        if m is None:raise PAWError('Unknown task: '+task)
        vec=encode(x)
        result=predict_from(m,vec,self._anchors(task))
        # Every observed input remains retrievable even when its active anchor
        # has been evicted. Conflicting labels produce an explicit uncertainty.
        exact=self.store.exact(task,digest(x))
        if exact:
            if m['mode']=='regression':
                result['answer']=mean(exact);result['source']='exact_evidence'
            else:
                counts={v:exact.count(v) for v in set(exact)}
                best=max(counts.values())
                if sum(v==best for v in counts.values())==1:
                    result['answer']=next(k for k,v in counts.items() if v==best)
                    result['source']='exact_evidence'
                else:
                    result['conflict']=True;result['answer']=None
                    result['source']='conflicting_evidence' 
        result.update({'task':task,'mode':m['mode'],'out_of_domain':not bool(exact) and result['closest_distance'] is not None and
                       result['closest_distance']>m['novelty_radius'],'novelty_radius':m['novelty_radius']})
        return result
    def auto(self,x,feedback=None):
        vec=encode(x)
        tasks=self._models()
        if not tasks:return {'status':'unknown','reason':'no stored tasks'}
        pre=[]
        for name,m in tasks:
            if feedback is not None:
                if m['mode']=='classification' and not isinstance(feedback,str):continue
                if m['mode']=='regression' and (isinstance(feedback,str) or isinstance(feedback,bool)):continue
            center=m.get('center')
            # Full archive is addressed only after task metadata filtering.
            # Group identity is checked by nearest real evidence, not centroid alone.
            pre.append((name,m))
        candidates=[]
        # The model count is bounded, but task routing is O(task_count); document it.
        # Conservative metadata shortlist keeps runtime bounded in model count;
        # may miss distributions with poor centroid summaries (reported limitation).
        if len(pre)>config.MAX_SELECTION_CANDIDATES*8:
            pre=sorted(pre,key=lambda it:squared(vec,it[1].get('center',[0]*config.FEATURE_DIM)))[:config.MAX_SELECTION_CANDIDATES*8]
        for name,m in pre:
            anchors=self._anchors(name)
            if not anchors:continue
            nearest=min(distance(vec,a[0]) for a in anchors)
            radius=max(config.MIN_NOVELTY_RADIUS,m['novelty_radius'])
            if nearest>radius:continue
            pred=predict_from(m,vec,anchors)
            if feedback is not None:
                exact=self.store.exact(name,digest(x))
                if exact and m['mode']=='regression':pred['answer']=mean(exact)
                if exact and m['mode']=='classification' and len(set(exact))==1:pred['answer']=exact[0]
            score=nearest/radius
            if feedback is not None:
                if m['mode']=='classification':
                    if pred['answer']!=feedback:continue
                else:
                    try:
                        residual=abs(float(pred['answer'])-finite(feedback,'feedback'))/max(.05,m['y_scale'])
                        if residual>2.0:continue
                        score+=residual
                    except (ValueError,TypeError):continue
            candidates.append((score,name,pred,nearest,radius))
        candidates.sort(key=lambda v:(v[0],v[1]))
        if not candidates:return {'status':'unknown','reason':'no matching task distribution'}
        best=candidates[0]
        if len(candidates)>1 and candidates[1][0]-best[0]<config.REQUIRE_SELECTION_MARGIN:
            return {'status':'ambiguous','reason':'multiple tasks fit the observed input/feedback',
                    'candidates':[{'task':q[1],'score':round(q[0],5)} for q in candidates[:config.MAX_SELECTION_CANDIDATES]]}
        selected_prediction=self.predict(best[1],x)
        return {'status':'selected','task':best[1],'prediction':selected_prediction['answer'],'source':selected_prediction['source'],
                'distance':best[3],'novelty_radius':best[4],
                'with_feedback':feedback is not None,'candidate_count':len(candidates)}
    def evaluate(self,task,rows):
        m=self.store.load(task_name(task))
        if m is None:raise PAWError('Unknown task')
        anchors=self._anchors(task)
        values=[]
        for x,y in rows:
            pred=predict_from(m,encode(x),anchors)['answer']
            values.append(objective(m,pred,y))
        score=mean(values)
        return {'task':task,'rows':len(rows),'method':m['method'],
                'accuracy':1-score if m['mode']=='classification' else None,
                'mse':score if m['mode']=='regression' else None}
