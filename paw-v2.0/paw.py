#!/usr/bin/env python3
"""PAW 2.0: fixed four-neuron core + validated external knowledge archive.
No flags: settings are in config.py. No third-party pip dependencies.
"""
import csv
import json
import math
import os
import shlex
import sys
from pathlib import Path
import config
from core import PAWError,Core
from encoding import read_dataset
from engine import Engine,task_name
from store import Store

def parse_x(s):
    try:
        value=json.loads(s)
    except json.JSONDecodeError:
        value=s
    # For convenience the numeric scalar command maps onto the CSV x column.
    return {'x':value} if isinstance(value,(int,float)) and not isinstance(value,bool) else value

def usage():
    print('''PAW 2.0  [four scalar neurons + Reflex]
  python paw.py demo
  python paw.py train TASK DATA.csv|DATA.jsonl
  python paw.py train-stream TASK DATA.csv|DATA.jsonl
  python paw.py predict TASK INPUT_JSON
  python paw.py auto INPUT_JSON [FEEDBACK_JSON]
  python paw.py evaluate TASK DATA.csv|DATA.jsonl
  python paw.py tasks
  python paw.py health TASK
  python paw.py audit
  python paw.py backup BACKUP.sqlite3
  python paw.py test
  python paw.py benchmark
  python paw.py legacy-info
Input example: '{"x":0.75}' or numeric shorthand: 0.75
JSONL lines: {"x":{"age":42,"city":"Dallas"},"y":"yes"}
Settings: config.py. No command-line flags.
''')

def make_demo():
    examples=config.ROOT/'examples';examples.mkdir(exist_ok=True)
    with (examples/'linear.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['x','y'])
        for i in range(160):
            x=-2+4*i/159
            w.writerow([x,2*x-1])
    with (examples/'remote.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['x','y'])
        for i in range(160):
            x=10+4*i/159
            w.writerow([x,-x+3])
    with (examples/'mixed.jsonl').open('w',encoding='utf-8') as f:
        for i in range(160):
            y='accept' if i%2 else 'deny'
            f.write(json.dumps({'x':{'token':'on' if i%2 else 'off','measure':float(i%4)},'y':y})+'\n')
    print('Created example CSV and JSONL datasets.')
    with Store() as store:
        e=Engine(store)
        for name,filename in [('linear','linear.csv'),('remote','remote.csv'),('mixed','mixed.jsonl')]:
            rows,mode=read_dataset(examples/filename)
            print(json.dumps(e.train(name,rows,mode),indent=2))
        print('Auto on x=-1:',json.dumps(e.auto({'x':-1}),indent=2))
        print('Auto on x=12:',json.dumps(e.auto({'x':12}),indent=2))

def run(argv):
    if not argv:
        usage()
        while True:
            try:s=input('PAW> ').strip()
            except (EOFError,KeyboardInterrupt):print();return 0
            if s.lower() in ('q','quit','exit'):return 0
            if s:
                try:run(shlex.split(s,posix=os.name!='nt'))
                except PAWError as exc:print('Error:',exc)
        return 0
    command=argv[0].lower()
    if command in ('help','?'):usage();return 0
    if command=='legacy-info':
        print('Previous v1 source is preserved in legacy_v1/. Its SQLite-incompatible JSON models are NOT silently migrated.')
        return 0
    if command=='demo' and len(argv)==1:make_demo();return 0
    if command=='test' and len(argv)==1:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(config.ROOT/'tests'),pattern='test_*.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if command=='benchmark' and len(argv)==1:
        import benchmark
        benchmark.run();return 0
    with Store() as store:
        e=Engine(store)
        if command=='train-stream' and len(argv)==3:
            out=e.train_stream(argv[1],argv[2])
        elif command=='train' and len(argv)==3:
            if Path(argv[2]).stat().st_size>config.STREAM_AUTO_FILE_BYTES:
                out=e.train_stream(argv[1],argv[2])
            else:
                rows,mode=read_dataset(argv[2])
                out=e.train(argv[1],rows,mode)
        elif command=='predict' and len(argv)==3:
            out=e.predict(argv[1],parse_x(argv[2]))
        elif command=='auto' and len(argv) in (2,3):
            feedback=json.loads(argv[2]) if len(argv)==3 else None
            out=e.auto(parse_x(argv[1]),feedback)
        elif command=='evaluate' and len(argv)==3:
            rows,mode=read_dataset(argv[2])
            m=store.load(argv[1])
            if m is None or m['mode']!=mode:raise PAWError('Unknown task or incompatible data type')
            out=e.evaluate(argv[1],rows)
        elif command=='tasks' and len(argv)==1:
            out=[{'task':name,'mode':m['mode'],'evidence':m['seen'],'method':m['method'],
                  'params':len(m['params']),'train_steps':m['trained_steps']} for name,m in store.items()]
        elif command=='health' and len(argv)==2:
            m=store.load(argv[1])
            if m is None:raise PAWError('Unknown task')
            anchors=store.anchors(argv[1])
            from engine import normalized_input
            out={'task':argv[1],'health':Core(m['params']).health([normalized_input(m,a[2]) for a in anchors])
                 if anchors else None,'retention_guard_rejections':m['retention_rejections'],
                 'repair_attempts':m['repairs'],'evidence':m['seen'],'method':m['method']}
        elif command=='audit' and len(argv)==1:out=store.integrity()
        elif command=='backup' and len(argv)==2:out={'backup':store.backup(argv[1])}
        else:
            usage();raise PAWError('Unknown command / incorrect argument count')
        print(json.dumps(out,indent=2,allow_nan=False,ensure_ascii=False))
    return 0

if __name__=='__main__':
    try:sys.exit(run(sys.argv[1:]))
    except (PAWError,ValueError,OverflowError,RecursionError) as exc:
        print('PAW error:',exc,file=sys.stderr);sys.exit(2)
    except KeyboardInterrupt:
        print('Stopped; last committed transaction is durable.',file=sys.stderr);sys.exit(130)
