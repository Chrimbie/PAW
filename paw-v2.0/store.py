"""Transactional SQLite Reflex memory, with checksummed core state.

SQLite is the single source of truth. Each complete batch (model + evidence)
is an atomic commit; no separate JSON/Adam consistency window exists.
"""
import hashlib
import json
import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
import config
from core import PAWError,Core,Adam
from encoding import jsonbytes,digest,ENCODER_VERSION

SCHEMA_VERSION=2

def unpack_model(row):
    raw=row['state']
    if not isinstance(raw,str) or len(raw)>config.MAX_MODEL_BYTES:raise PAWError('Invalid model payload length')
    if hashlib.sha256(raw.encode('utf8')).hexdigest()!=row['checksum']:
        raise PAWError('Stored model checksum mismatch')
    try:m=json.loads(raw)
    except ValueError as exc:raise PAWError('Malformed model JSON') from exc
    if m.get('format')!='PAW2' or m.get('task')!=row['task'] or m.get('mode') not in ('regression','classification'):
        raise PAWError('Invalid model metadata')
    if m.get('encoder_version')!=ENCODER_VERSION or m.get('feature_dim')!=config.FEATURE_DIM:
        raise PAWError('Incompatible encoder or feature-width version')
    if not isinstance(m.get('classes'),list):raise PAWError('Invalid label vocabulary')
    Core(m['params']);Adam.from_dict(m['adam'])
    if m.get('trained_steps')!=m['adam']['step_count']:
        raise PAWError('Core/Adam step counter mismatch')
    if not 0< m['y_scale'] or not 0<m['x_scale']:
        raise PAWError('Invalid stored normalization')
    if m['seen']<0 or m['trained_steps']<0:raise PAWError('Invalid counters')
    return m

class Store:
    def __init__(self,path=config.STORE_PATH):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise PAWError('SQLite store path must not be a symlink')
        self.db=sqlite3.connect(str(self.path), timeout=config.SQLITE_BUSY_TIMEOUT_MS/1000, isolation_level=None)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA busy_timeout='+str(config.SQLITE_BUSY_TIMEOUT_MS))
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS models(task TEXT PRIMARY KEY,state TEXT NOT NULL,checksum TEXT NOT NULL,version INTEGER NOT NULL CHECK(version>=0));
        CREATE TABLE IF NOT EXISTS evidence(task TEXT NOT NULL,uid TEXT NOT NULL,xhash TEXT NOT NULL,features TEXT NOT NULL,y TEXT NOT NULL,projection REAL NOT NULL,
          stamp INTEGER NOT NULL,PRIMARY KEY(task,uid),FOREIGN KEY(task) REFERENCES models(task) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS evidence_stamp ON evidence(task,stamp);
        CREATE INDEX IF NOT EXISTS evidence_exact ON evidence(task,xhash);
        CREATE TABLE IF NOT EXISTS anchors(task TEXT NOT NULL,slot INTEGER NOT NULL,uid TEXT NOT NULL,features TEXT NOT NULL,y TEXT NOT NULL,projection REAL NOT NULL,
          PRIMARY KEY(task,slot),FOREIGN KEY(task) REFERENCES models(task) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS anchor_task ON anchors(task);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT,task TEXT NOT NULL,kind TEXT NOT NULL,details TEXT NOT NULL,utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        ''')
        v=self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
        if v is None:self.db.execute("INSERT INTO metadata(key,value) VALUES('schema',?)",(str(SCHEMA_VERSION),))
        elif int(v[0])!=SCHEMA_VERSION:raise PAWError('Unsupported database schema; do not open with this version')
    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield self.db
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
    def load(self,task):
        row=self.db.execute('SELECT * FROM models WHERE task=?',(task,)).fetchone()
        return unpack_model(row) if row else None
    def save(self,model):
        raw=jsonbytes(model).decode('utf8')
        if len(raw)>config.MAX_MODEL_BYTES:raise PAWError('Model too large')
        checksum=hashlib.sha256(raw.encode('utf8')).hexdigest()
        self.db.execute('INSERT INTO models(task,state,checksum,version) VALUES(?,?,?,1) ON CONFLICT(task) DO UPDATE SET state=excluded.state, checksum=excluded.checksum, version=models.version+1',
                        (model['task'],raw,checksum))
    def count(self,task):
        return self.db.execute('SELECT count(*) FROM evidence WHERE task=?',(task,)).fetchone()[0]
    def append(self,task,uid,xhash,vec,y,proj,stamp):
        """Returns False if identical evidence already committed; bounded anchors + full provenance."""
        n=stamp-1
        c=self.db.execute('INSERT OR IGNORE INTO evidence(task,uid,xhash,features,y,projection,stamp) VALUES(?,?,?,?,?,?,?)',
             (task,uid,xhash,jsonbytes(vec).decode('utf8'),jsonbytes(y).decode('utf8'),proj,stamp)).rowcount
        if not c:return False
        if n<config.MAX_ANCHORS_PER_TASK:slot=n
        else:
            # Deterministic uniform-ish reservoir index per distinct evidence identity.
            k=int(hashlib.blake2b((uid+':slot').encode(),digest_size=8).hexdigest(),16)%(n+1)
            slot=k if k<config.MAX_ANCHORS_PER_TASK else None
        if slot is not None:
            self.db.execute('INSERT INTO anchors(task,slot,uid,features,y,projection) VALUES(?,?,?,?,?,?) ON CONFLICT(task,slot) DO UPDATE SET uid=excluded.uid,features=excluded.features,y=excluded.y,projection=excluded.projection',
                            (task,slot,uid,jsonbytes(vec).decode('utf8'),jsonbytes(y).decode('utf8'),proj))
        return True
    def exact(self,task,xhash):
        return [json.loads(r[0]) for r in self.db.execute('SELECT y FROM evidence WHERE task=? AND xhash=? LIMIT 128',(task,xhash))]
    def anchors(self,task):
        return [(json.loads(r['features']),json.loads(r['y']),r['projection']) for r in self.db.execute('SELECT features,y,projection FROM anchors WHERE task=? ORDER BY slot',(task,))]
    def items(self):
        return [(r['task'],unpack_model(r)) for r in self.db.execute('SELECT * FROM models ORDER BY task')]
    def audit(self,task,kind,details):
        self.db.execute('INSERT INTO audit(task,kind,details) VALUES(?,?,?)',(task,kind,jsonbytes(details).decode('utf8')))
    def integrity(self):
        status=self.db.execute('PRAGMA integrity_check').fetchone()[0]
        if status!='ok':raise PAWError('SQLite integrity check failed: '+str(status))
        issues=[]
        for row in self.db.execute('SELECT * FROM models'):
            try:
                m=unpack_model(row)
                actual=self.count(m['task'])
                if actual!=m['seen']:issues.append(f"{m['task']}: counter {m['seen']} != {actual}")
                ac=self.db.execute('SELECT count(*) FROM anchors WHERE task=?',(m['task'],)).fetchone()[0]
                if ac>config.MAX_ANCHORS_PER_TASK or ac>actual:issues.append(f"{m['task']}: invalid anchor count")
            except Exception as exc:issues.append(f"{row['task']}: {exc}")
        mismatched=self.db.execute('''SELECT count(*) FROM anchors a LEFT JOIN evidence e
          ON a.task=e.task AND a.uid=e.uid
          WHERE e.uid IS NULL OR a.features!=e.features OR a.y!=e.y OR a.projection!=e.projection''').fetchone()[0]
        if mismatched:issues.append(f'{mismatched} active anchors disagree with archived observations')
        if issues:raise PAWError('Store verification failed: '+ '; '.join(issues[:10]))
        return {'integrity':'ok','tasks':self.db.execute('SELECT count(*) FROM models').fetchone()[0],
                'evidence':self.db.execute('SELECT count(*) FROM evidence').fetchone()[0],
                'bytes':self.path.stat().st_size}
    def backup(self,destination):
        dest=Path(destination)
        if dest.resolve()==self.path.resolve():raise PAWError('Backup destination equals source')
        if dest.exists():raise PAWError('Refusing to overwrite existing backup')
        dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            with sqlite3.connect(str(dest)) as other:self.db.backup(other)
            with sqlite3.connect(str(dest)) as check:
                if check.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise PAWError('Backup integrity check failed')
        except BaseException:
            if dest.exists():dest.unlink()
            raise
        return str(dest)
