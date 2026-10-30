"""Deterministic nontrainable, fixed-width raw-data translator.

No pretrained embeddings, encoder weights, or learned extra neurons. It accepts
JSON numbers, booleans, strings, lists and dictionaries and is deliberately
lossy. It is not a visual, audio or semantic-language model.
"""
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import config
from core import PAWError,finite

_WORDS = re.compile(r"[\w']+",re.UNICODE)
ENCODER_VERSION = 2

def jsonbytes(x):
    return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')

def digest(x):return hashlib.sha256(jsonbytes(x)).hexdigest()

def _insert(vec,key,value):
    h=hashlib.blake2b(key.encode('utf8'),digest_size=16).digest()
    index=int.from_bytes(h[:4],'little')%config.FEATURE_DIM
    sign=1 if h[4]&1 else -1
    if ':token:' in key or ':string:' in key:
        other=int.from_bytes(h[6:10],'little')%config.FEATURE_DIM
        if other==index:other=(other+1)%config.FEATURE_DIM
        sign2=1 if h[10]&1 else -1
        vec[index]+=sign*value/math.sqrt(2)
        vec[other]+=sign2*value/math.sqrt(2)
    else:
        vec[index]+=sign*value

def encode(x):
    vec=[0.0]*config.FEATURE_DIM
    entries=0
    def visit(value,key,depth):
        nonlocal entries
        if depth>12:raise PAWError('Feature nesting exceeds 12 levels')
        if entries>=config.MAX_FEATURES_PER_SAMPLE:raise PAWError('Too many feature entries')
        if value is None:
            _insert(vec,key+':null',.4);entries+=1
        elif isinstance(value,bool):
            _insert(vec,key+':bool',.75 if value else -.75);entries+=1
        elif isinstance(value,(int,float)):
            z=finite(value,'input feature')
            _insert(vec,key+':number',4*math.tanh(z/8))
            entries+=1
        elif isinstance(value,str):
            if len(value)>100000:raise PAWError('Text feature exceeds 100000 characters')
            tokens=_WORDS.findall(value.lower())
            if not tokens and value:_insert(vec,key+':string:'+value[:64],.5);entries+=1
            if len(tokens)>config.MAX_FEATURES_PER_SAMPLE-entries:
                raise PAWError('Too many text tokens')
            for token in tokens:
                _insert(vec,key+':token:'+token,1.0/math.sqrt(max(len(tokens),1)))
                entries+=1
        elif isinstance(value,(list,tuple)):
            if len(value)>config.MAX_FEATURES_PER_SAMPLE:raise PAWError('Array too large')
            for i,v in enumerate(value):visit(v,key+':'+str(i),depth+1)
        elif isinstance(value,dict):
            if len(value)>config.MAX_FEATURES_PER_SAMPLE:raise PAWError('Dictionary too large')
            if not all(isinstance(k,str) for k in value):raise PAWError('JSON feature keys must be strings')
            for k in sorted(value):visit(value[k],key+'/'+k,depth+1)
        else:raise PAWError(f'Unsupported input value type: {type(value).__name__}')
    try:visit(x,'input',0)
    except (OverflowError,RecursionError) as exc:raise PAWError('Input cannot be encoded') from exc
    if entries==0:raise PAWError('Input has no features')
    if not all(math.isfinite(v) for v in vec):raise PAWError('Feature projection is nonfinite')
    return vec

def projection(vec):
    if len(vec)!=config.FEATURE_DIM:raise PAWError('Incorrect feature dimension')
    total=0.0
    for i,v in enumerate(vec):
        h=hashlib.blake2b(f'paw-proj-{i}'.encode(),digest_size=2).digest()
        total+=(1 if h[0]&1 else -1)*v
    return finite(total/max(1,math.sqrt(sum(abs(v)>1e-14 for v in vec))),'scalar projection')

def squared(a,b):return sum((x-y)*(x-y) for x,y in zip(a,b))

def distance(a,b):return math.sqrt(squared(a,b))

def parse_number_or_string(s):
    if not isinstance(s,str) or not s.strip():raise PAWError('Missing target')
    try:return finite(float(s),'target')
    except (ValueError,OverflowError):return s[:config.MAX_LABEL_CHARS]

def parse_csv_feature(s):
    if not isinstance(s,str):raise PAWError('Malformed CSV field')
    if len(s)>100000:raise PAWError('CSV text feature exceeds 100000 characters')
    try:return finite(float(s),'input feature')
    except (ValueError,OverflowError):return s

def read_dataset(path):
    path=Path(path)
    if not path.is_file():raise PAWError(f'Dataset not found: {path}')
    rows=[]
    try:
        if path.suffix.lower()=='.jsonl':
            with path.open('r',encoding='utf-8-sig') as f:
                for number,line in enumerate(f,1):
                    if not line.strip():continue
                    if len(line)>config.MAX_EXAMPLE_BYTES:raise PAWError(f'JSONL line {number} too large')
                    obj=json.loads(line)
                    if not isinstance(obj,dict) or 'x' not in obj or 'y' not in obj:
                        raise PAWError(f'JSONL line {number} must include x and y')
                    target=obj['y']
                    if isinstance(target,(float,int)) and not isinstance(target,bool):
                        target=finite(target,'target')
                    elif isinstance(target,(str,bool)):
                        target=str(target)
                    else:raise PAWError('Target must be a scalar number or class label')
                    rows.append((obj['x'],target))
                    if len(rows)>config.MAX_ROWS_PER_IMPORT:raise PAWError('Dataset row limit exceeded')
        elif path.suffix.lower()=='.csv':
            with path.open('r',newline='',encoding='utf-8-sig') as f:
                reader=csv.DictReader(f)
                if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):
                    raise PAWError('CSV needs unique column headers')
                target=next((k for k in ('target','y','label') if k in reader.fieldnames),reader.fieldnames[-1])
                fields=[f for f in reader.fieldnames if f!=target]
                if not fields:raise PAWError('CSV must contain at least one input feature column')
                force_class=(target=='label')
                for number,row in enumerate(reader,2):
                    if None in row or any(v is None for v in row.values()):
                        raise PAWError(f'Malformed CSV row {number}')
                    x={k:parse_csv_feature(row[k]) for k in fields}
                    y=parse_number_or_string(row[target]);y=str(row[target]) if force_class else y
                    rows.append((x,y))
                    if len(rows)>config.MAX_ROWS_PER_IMPORT:raise PAWError('Dataset row limit exceeded')
        else:raise PAWError('Use .csv or .jsonl (text, numbers, categorical fields, vectors)')
    except (OSError,UnicodeError,json.JSONDecodeError,csv.Error) as exc:
        raise PAWError(f'Invalid dataset: {exc}') from exc
    if len(rows)<4:raise PAWError('Dataset requires at least four labeled examples')
    mode='classification' if any(isinstance(y,str) for _,y in rows) else 'regression'
    if mode=='classification':
        rows=[(x,str(y)) for x,y in rows]
        if any(len(y)>config.MAX_LABEL_CHARS for _,y in rows):raise PAWError('Class label too long')
    for x,y in rows:encode(x)
    return rows,mode

def iter_dataset(path):
    """Bounded-memory streaming reader. Yields validated (x,y,mode) rows.

    Mode is derived from the target column / each JSONL target. Mixed modes
    within one file are refused by the caller; only a small chunk is resident.
    """
    path=Path(path)
    if not path.is_file():raise PAWError(f'Dataset not found: {path}')
    if path.suffix.lower()=='.jsonl':
        with path.open('r',encoding='utf-8-sig') as f:
            for number,line in enumerate(f,1):
                if not line.strip():continue
                if len(line)>config.MAX_EXAMPLE_BYTES:raise PAWError('JSONL row too large')
                try:obj=json.loads(line)
                except ValueError as e:raise PAWError(f'Invalid JSONL row {number}: {e}') from e
                if not isinstance(obj,dict) or not {'x','y'}<=set(obj):raise PAWError('JSONL needs x and y')
                y=obj['y']
                if isinstance(y,bool) or isinstance(y,str):
                    mode='classification';y=str(y)
                elif isinstance(y,(int,float)):
                    mode='regression';y=finite(y,'target')
                else:raise PAWError('Unsupported target type')
                encode(obj['x'])
                yield obj['x'],y,mode
    elif path.suffix.lower()=='.csv':
        with path.open('r',newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):raise PAWError('Duplicate or missing CSV columns')
            target=next((k for k in ('target','y','label') if k in reader.fieldnames),reader.fieldnames[-1])
            fields=[f for f in reader.fieldnames if f!=target]
            if not fields:raise PAWError('Missing feature columns')
            for line,row in enumerate(reader,2):
                if None in row or any(v is None for v in row.values()):raise PAWError(f'Malformed CSV row {line}')
                x={k:parse_csv_feature(row[k]) for k in fields}
                y=str(row[target]) if target=='label' else parse_number_or_string(row[target])
                mode='classification' if isinstance(y,str) else 'regression'
                encode(x)
                yield x,y,mode
    else:raise PAWError('Expected .csv or .jsonl')
