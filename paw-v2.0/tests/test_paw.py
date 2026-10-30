"""Real end-to-end regression tests, failure injection, concurrency and memory controls."""
import csv
import json
import math
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import config
from core import PAWError,Core,Adam,new_params
from encoding import encode,projection,read_dataset,digest
from engine import Engine
from store import Store


def line(n=64,lo=-2,hi=2,kind='first'):
    out=[]
    for i in range(n):
        x=lo+(hi-lo)*i/(n-1)
        y=2*x-1 if kind=='first' else -x+3
        out.append(({'x':x},y))
    return out

def textrows(n=60):
    return [({'word':'up' if i%2 else 'down','v':i%3},'yes' if i%2 else 'no') for i in range(n)]

class PAWTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.folder=Path(t.name)
        self.store=Store(self.folder/'paw.db');self.addCleanup(self.store.close)
        self.engine=Engine(self.store)
    def test_01_fixed_four_neurons(self):
        core=Core(new_params())
        self.assertEqual(len(core.params),8)
        self.assertEqual(len(core.forward(0.2)[1]),6)
        with self.assertRaises(PAWError):Core([0]*10)
    def test_02_analytic_gradient(self):
        c=Core([.6,-.5,.7,1.,.1,-.05,.04,.01])
        rows=[(-1.,.8),(.6,-.9)]
        _,grad=c.loss_grad(rows)
        for i in range(8):
            w=c.params[:];w[i]+=1e-6
            a=Core(w).loss_grad(rows)[0]
            w[i]-=2e-6
            b=Core(w).loss_grad(rows)[0]
            self.assertAlmostEqual(grad[i],(a-b)/2e-6,delta=1e-8)
    def test_03_bound_saturation_large_inputs(self):
        p=new_params()
        a=Adam.fresh()
        for i in range(300):
            c=Core(p)
            _,g=c.loss_grad([(float(10000*(-1)**i),float(i%10))])
            p=a.update(p,g)
            h=Core(p).health([-1e100,-100,0,100,1e100])
            self.assertEqual(h['saturation_fraction'],0)
            self.assertGreater(h['worst_hidden_derivative'],0.27)
            self.assertLessEqual(h['maximum_preactivation'],config.CORE_Z_LIMIT+1e-10)
    def test_04_learns_and_reloads(self):
        out=self.engine.train('line',line(), 'regression')
        self.assertEqual(out['core_neurons'],4)
        self.assertLess(out['validation_losses']['core'],2.0)
        x={'x':.5};ans=self.engine.predict('line',x)
        with Store(self.folder/'paw.db') as second:
            self.assertEqual(Engine(second).predict('line',x),ans)
    def test_05_no_cross_task_overwrite(self):
        self.engine.train('first',line(), 'regression')
        original=self.store.load('first')
        self.engine.train('second',line(lo=10,hi=14,kind='other'),'regression')
        self.assertEqual(original,self.store.load('first'))
        self.assertGreater(self.store.count('second'),0)
    def test_06_auto_routes_distinct(self):
        self.engine.train('first',line(), 'regression')
        self.engine.train('second',line(lo=10,hi=14,kind='other'),'regression')
        self.assertEqual(self.engine.auto({'x':0.1})['task'],'first')
        self.assertEqual(self.engine.auto({'x':12})['task'],'second')
    def test_07_auto_rejects_unseen(self):
        self.engine.train('first',line(), 'regression')
        self.assertEqual(self.engine.auto({'color':'impossible long distinct token'})['status'],'unknown')
    def test_08_ambiguous_overlapping_and_feedback(self):
        common=[{'x':float(i/12-2)} for i in range(50)]
        a=[(x,0.0) for x in common]
        b=[(x,10.0) for x in common]
        self.engine.train('A',a,'regression')
        self.engine.train('B',b,'regression')
        query=common[22]
        self.assertEqual(self.engine.auto(query)['status'],'ambiguous')
        self.assertEqual(self.engine.auto(query,0.0)['task'],'A')
        self.assertEqual(self.engine.auto(query,10.0)['task'],'B')
    def test_09_unknown_feedback_rejected(self):
        self.engine.train('A',line(),'regression')
        self.assertEqual(self.engine.auto({'x':0.4},10000)['status'],'unknown')
    def test_10_classification_mixed_text_numeric(self):
        self.engine.train('text',textrows(),'classification')
        self.assertEqual(self.engine.predict('text',{'word':'up','v':1})['answer'],'yes')
        self.assertEqual(self.engine.predict('text',{'word':'down','v':2})['answer'],'no')
    def test_11_csv_arbitrary_columns(self):
        p=self.folder/'mixed.csv'
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['age','city','score','label'])
            for i in range(40):w.writerow([i,'X' if i%2 else 'Y',i/20,'yes' if i%2 else 'no'])
        records,mode=read_dataset(p)
        self.assertEqual(mode,'classification')
        self.assertEqual(len(records),40)
        self.engine.train('tabular',records,mode)
        self.assertEqual(self.store.count('tabular'),32)
    def test_12_jsonl_nested_vectors(self):
        p=self.folder/'nested.jsonl'
        with p.open('w') as f:
            for i in range(40):f.write(json.dumps({'x':{'v':[i%5,i%2],'txt':'hello'},'y':float(i%5)})+'\n')
        records,mode=read_dataset(p)
        self.assertEqual(mode,'regression')
        self.engine.train('nested',records,mode)
        self.assertIsInstance(self.engine.predict('nested',{'v':[1,1],'txt':'hello'})['answer'],float)
    def test_13_duplicate_training_idempotence_evidence(self):
        rows=line()
        self.engine.train('A',rows,'regression')
        n=self.store.count('A')
        out=self.engine.train('A',rows,'regression')
        self.assertEqual(n,self.store.count('A'))
        self.assertEqual(out['new_evidence'],0)
    def test_14_full_provenance_after_anchor_eviction(self):
        with patch.object(config,'MAX_ANCHORS_PER_TASK',8):
            rows=line(80)
            self.engine.train('A',rows,'regression')
            anchors={r['uid'] for r in self.store.db.execute("SELECT uid FROM anchors WHERE task='A'")}
            original=[(x,y) for x,y in rows if digest([x,y]) not in anchors]
            self.assertTrue(original)
            x,y=original[0]
            result=self.engine.predict('A',x)
            # Verify query actually belongs to archived training and was evicted.
            if self.store.exact('A',digest(x)):
                self.assertAlmostEqual(result['answer'],y)
                self.assertEqual(result['source'],'exact_evidence')
            else:
                self.fail('No evicted training example found')
    def test_15_transaction_rollback(self):
        self.engine.train('A',line(),'regression')
        before=self.store.load('A');n=self.store.count('A')
        try:
            with self.store.transaction():
                temp=self.store.load('A');temp['params'][0]+=1
                self.store.save(temp)
                self.store.append('A','fake_uid','fake_x',[0.1]*config.FEATURE_DIM,8.5,0.5,n+1)
                raise RuntimeError('simulated process failure')
        except RuntimeError:pass
        self.assertEqual(before,self.store.load('A'))
        self.assertEqual(n,self.store.count('A'))
    def test_16_database_backup_and_integrity(self):
        self.engine.train('A',line(),'regression')
        self.assertEqual(self.store.integrity()['integrity'],'ok')
        target=self.folder/'backup.db'
        self.store.backup(target)
        with Store(target) as backup:
            self.assertEqual(backup.load('A'),self.store.load('A'))
            self.assertEqual(backup.integrity()['integrity'],'ok')
    def test_17_corrupt_checksum_detected(self):
        self.engine.train('A',line(),'regression')
        self.store.db.execute("UPDATE models SET state='{}' WHERE task='A'")
        with self.assertRaisesRegex(PAWError,'checksum'):
            self.store.load('A')
    def test_18_task_string_is_not_sql(self):
        self.engine.train("name';DROP TABLE models;--",line(),'regression')
        self.assertIsNotNone(self.store.load("name';DROP TABLE models;--"))
        self.assertEqual(self.store.integrity()['tasks'],1)
    def test_19_bounded_active_memory(self):
        with patch.object(config,'MAX_ANCHORS_PER_TASK',12):
            self.engine.train('A',line(300),'regression')
            self.assertEqual(self.store.count('A'),240)
            self.assertLessEqual(len(self.store.anchors('A')),12)
    def test_20_interrupted_fit_resume(self):
        rows=line(50)
        with tempfile.TemporaryDirectory() as t:
            with Store(Path(t)/'other.db') as other:
                Engine(other).train('A',rows,'regression')
                expected=other.load('A')
        real_save=self.store.save
        hit=[0]
        def crash(m):
            job=m.get('fit_job')
            if job and job['done']>=config.FIT_CHECKPOINT_EVERY:
                hit[0]+=1
                if hit[0]==1:raise RuntimeError('crash before transaction commit')
            return real_save(m)
        with patch.object(self.store,'save',side_effect=crash):
            with self.assertRaises(RuntimeError):self.engine.train('A',rows,'regression')
        self.engine.train('A',rows,'regression')
        self.assertEqual(expected,self.store.load('A'))
    def test_21_concurrent_task_writes(self):
        faults=[]
        def run(name,lo):
            try:
                with Store(self.folder/'paw.db') as db:
                    Engine(db).train(name,line(70,lo,lo+4),'regression')
            except Exception as exc:faults.append(str(exc))
        a=threading.Thread(target=run,args=('A',-5))
        b=threading.Thread(target=run,args=('B',10))
        a.start();b.start();a.join();b.join()
        self.assertEqual(faults,[])
        self.assertEqual(self.store.integrity()['tasks'],2)
    def test_22_reject_invalid_inputs(self):
        for x in [{'a':float('nan')},{'a':[float('inf')]},{}]:
            with self.assertRaises(PAWError):encode(x)
        with self.assertRaises(PAWError):self.engine.train('bad',[], 'regression')

if __name__=='__main__':unittest.main()

# Additional bounded-memory and unbounded-history regression tests.
class StreamingTests(unittest.TestCase):
    def test_streaming_chunks_and_tail_no_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            p=folder/'stream.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['x','y'])
                for i in range(1047):w.writerow([i/30, i/30*1.3])
            with patch.object(config,'STREAM_CHUNK_ROWS',128):
                with Store(folder/'model.db') as store:
                    e=Engine(store)
                    result=e.train_stream('stream',p)
                    self.assertEqual(result['rows_processed'],1047)
                    # 20% held out in each full chunk; no silent dropped tail.
                    self.assertEqual(result['store_evidence'], 8*102+18)
                    self.assertLessEqual(len(store.anchors('stream')),config.MAX_ANCHORS_PER_TASK)
                    self.assertEqual(store.integrity()['integrity'],'ok')

    def test_replay_stream_no_duplicate_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            p=folder/'stream.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['x','y'])
                for i in range(257):w.writerow([i,2*i])
            with patch.object(config,'STREAM_CHUNK_ROWS',128):
                with Store(folder/'model.db') as db:
                    e=Engine(db)
                    a=e.train_stream('stream',p)
                    old=db.load('stream')
                    b=e.train_stream('stream',p)
                    self.assertEqual(old,db.load('stream'))
                    self.assertEqual(a['store_evidence'],b['store_evidence'])
                    self.assertEqual(b['new_evidence'],0)

    def test_large_unseen_neuron_input_abstains(self):
        with tempfile.TemporaryDirectory() as folder:
            with Store(Path(folder)/'p.db') as s:
                e=Engine(s);e.train('x',line(),'regression')
                unknown=e.auto({'x':1e100})
                self.assertEqual(unknown['status'],'unknown')

class ReliabilityTests(unittest.TestCase):
    def test_other_connection_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'shared.db'
            with Store(path) as a, Store(path) as b:
                ea,eb=Engine(a),Engine(b)
                ea.train('one',line(),'regression')
                self.assertEqual(ea.auto({'x':12})['status'],'unknown')
                eb.train('two',line(lo=10,hi=14,kind='other'),'regression')
                self.assertEqual(ea.auto({'x':12})['task'],'two')

    def test_integrity_detects_corrupt_active_index(self):
        with tempfile.TemporaryDirectory() as d:
            with Store(Path(d)/'paw.db') as s:
                e=Engine(s);e.train('task',line(),'regression')
                s.db.execute("UPDATE anchors SET y='100000' WHERE task='task' AND slot=0")
                with self.assertRaisesRegex(PAWError,'anchor'):
                    s.integrity()

    def test_retention_after_conflicting_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            with Store(Path(d)/'paw.db') as s:
                e=Engine(s)
                old=line(65,-3,-1,kind='first')
                e.train('task',old,'regression')
                original=e.predict('task',old[22][0])
                newer=line(65,1,3,kind='other')
                e.train('task',newer,'regression')
                after=e.predict('task',old[22][0])
                # Exact evidence is immutable; a new domain cannot erase it.
                if original['source']=='exact_evidence':
                    self.assertEqual(after['answer'],original['answer'])
                    self.assertEqual(after['source'],'exact_evidence')
                else:
                    self.assertEqual(after['source'],'reflex')
