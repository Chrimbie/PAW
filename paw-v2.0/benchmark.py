"""Reproducible local stress smoke test. No external datasets required."""
import json
import random
import statistics
import tempfile
import time
from pathlib import Path
import config
from engine import Engine
from store import Store

def run():
    # 5,000 distinct observations + 24 other independent task memories.
    t0=time.perf_counter();rng=random.Random(9187)
    with tempfile.TemporaryDirectory() as tmp:
        with Store(Path(tmp)/'stress.db') as st:
            e=Engine(st)
            base=[({'x':-5+i*10/4999,'c':i%7}, 1.5*(-5+i*10/4999)+0.2*(i%7)) for i in range(5000)]
            t1=time.perf_counter()
            large=e.train_stream('large', _write(tmp,'large.csv',base))
            train_time=time.perf_counter()-t1
            first_pred=e.predict('large',{'x':1.2,'c':2})
            first_before=st.load('large')
            for task in range(24):
                rows=[({'identity':f'id_{task}', 'v':j/40},'yes' if j%2 else 'no') for j in range(55)]
                e.train(f'task_{task:02}',rows,'classification',streaming=True)
            first_after=st.load('large')
            a=time.perf_counter()
            probes=[{'identity':'id_'+str(i%24),'v':(i%55)/40} for i in range(300)]
            outcomes=[e.auto(p) for p in probes]
            route_seconds=time.perf_counter()-a
            correct=sum(r['status']=='selected' and r['task']==f'task_{i%24:02}' for i,r in enumerate(outcomes))
            unknown=[e.auto({'identity':'a completely unseen nonsensical identifier '+str(i),'v':i/4}) for i in range(40)]
            a=time.perf_counter()
            queries=[e.predict('large',{'x':rng.uniform(-4,4),'c':rng.randrange(7)}) for _ in range(400)]
            predict_seconds=time.perf_counter()-a
            audit=st.integrity()
            backup_path=Path(tmp)/'backup.sqlite3'
            st.backup(backup_path)
            with Store(backup_path) as copy:
                backup=copy.integrity()
                assert copy.load('large')==st.load('large')
            result={'version':'2.0','stress_type':'local_smoke_not_production_certification',
                    'total_tasks':audit['tasks'],'large_input_rows':5000,
                    'large_new_training_evidence':large['new_evidence'],
                    'large_train_seconds':round(train_time,3),
                    'large_input_rows_per_second':round(5000/train_time,1),
                    'large_active_anchors':large['active_anchors'],
                    'large_archive_evidence':st.count('large'),
                    'route_queries':len(probes), 'route_queries_per_second':round(len(probes)/route_seconds,1),
                    'route_selected':sum(x['status']=='selected' for x in outcomes),
                    'route_correct':correct,'unseen_inputs_tested':40,
                    'unseen_inputs_rejected':sum(x['status']=='unknown' for x in unknown),
                    'route_unknown':sum(x['status']=='unknown' for x in outcomes),
                    'route_ambiguous':sum(x['status']=='ambiguous' for x in outcomes),
                    'explicit_predictions_per_second':round(len(queries)/predict_seconds,1),
                    'first_task_unchanged_when_adding_24_tasks':first_before==first_after
                           and first_pred==e.predict('large',{'x':1.2,'c':2}),
                    'database_integrity':audit['integrity'],'backup_integrity':backup['integrity'],
                    'sqlite_size_bytes':audit['bytes'],
                    'total_wall_seconds':round(time.perf_counter()-t0,3)}
    (config.ROOT/'BENCHMARK_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

def _write(tmp,name,rows):
    import csv
    path=Path(tmp)/name
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['x','c','y'])
        for x,y in rows:w.writerow([x['x'],x['c'],y])
    return path

if __name__=='__main__':run()
