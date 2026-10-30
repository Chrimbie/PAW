"""Fixed 1 -> 1 -> 1 -> 1 scalar neuron chain, with two tanh hidden neurons.

No gates, additional trainable layers, hidden classifiers, or implicit neural
encoders. The first neuron is affine. Eight trainable scalar parameters total.
For normalized/clipped x in [-4,4], weight projection keeps |hidden z| <=1.25,
so tanh derivatives stay >=0.2804. This does NOT guarantee nonzero variance or
learning quality, and OOD inputs are clipped/flagged by the caller.
"""
import math
import random
from dataclasses import dataclass
import config

class PAWError(Exception):
    pass

def finite(x, context='number'):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise PAWError(f'{context} must be a finite real number')
    return float(x)

def new_params(seed=config.SEED):
    rng = random.Random(seed)
    return project([rng.uniform(0.4,0.75) for _ in range(4)] + [0.0]*4)

def project(p):
    if len(p)!=8:
        raise PAWError('Four-neuron core requires exactly eight parameters')
    p=[finite(v,'core weight') for v in p]
    # Maintain an analytically bounded preactivation on the full input range.
    bound0=abs(p[0])*config.CORE_INPUT_CLIP+abs(p[4])
    bound1=abs(p[1])*bound0+abs(p[5])
    if bound1>config.CORE_Z_LIMIT:
        k=config.CORE_Z_LIMIT/bound1
        p[1]*=k; p[5]*=k
    bound2=abs(p[2])*math.tanh(config.CORE_Z_LIMIT)+abs(p[6])
    if bound2>config.CORE_Z_LIMIT:
        k=config.CORE_Z_LIMIT/bound2
        p[2]*=k; p[6]*=k
    return p

@dataclass
class Core:
    params:list
    def __post_init__(self):
        if len(self.params)!=8:raise PAWError('Expected exactly eight parameters')
        self.params=[finite(v,'parameter') for v in self.params]
    def forward(self,x):
        x=finite(x,'input')
        x=max(-config.CORE_INPUT_CLIP,min(config.CORE_INPUT_CLIP,x))
        w0,w1,w2,w3,b0,b1,b2,b3=self.params
        a=w0*x+b0
        z1=w1*a+b1; h1=math.tanh(z1)
        z2=w2*h1+b2; h2=math.tanh(z2)
        y=finite(w3*h2+b3,'prediction')
        return y,(x,a,z1,h1,z2,h2)
    def loss_grad(self,rows):
        if not rows:raise PAWError('Empty batch')
        n=len(rows); grad=[0.0]*8;loss=0.0
        w0,w1,w2,w3=self.params[:4]
        for x,y in rows:
            pred,(xi,a,z1,h1,z2,h2)=self.forward(x)
            err=pred-finite(y,'target')
            loss+=err*err/n
            d=2*err/n
            grad[3]+=d*h2;grad[7]+=d
            d2=d*w3*(1-h2*h2)
            grad[2]+=d2*h1;grad[6]+=d2
            d1=d2*w2*(1-h1*h1)
            grad[1]+=d1*a;grad[5]+=d1
            d0=d1*w1
            grad[0]+=d0*xi;grad[4]+=d0
        finite(loss,'loss')
        if not all(math.isfinite(v) for v in grad):raise PAWError('Nonfinite gradient')
        return loss,grad
    def health(self,xs):
        if not xs:raise PAWError('No health inputs')
        samples=[self.forward(x)[1] for x in xs]
        h=[[s[3] for s in samples],[s[5] for s in samples]]
        variances=[sum((v-sum(a)/len(a))**2 for v in a)/len(a) for a in h]
        sat=sum(abs(v)>=config.SATURATION_ABS for a in h for v in a)/(2*len(xs))
        maxz=max(max(abs(s[2]),abs(s[4])) for s in samples)
        derivative_floor=min(1-math.tanh(s[2])**2 for s in samples)
        derivative_floor=min(derivative_floor,min(1-math.tanh(s[4])**2 for s in samples))
        return {'saturation_fraction':sat,'hidden_variances':variances,
                'dead_hidden_neurons':sum(v<config.DEAD_VARIANCE for v in variances) if len(xs)>7 else None,
                'worst_hidden_derivative':derivative_floor,'maximum_preactivation':maxz}

@dataclass
class Adam:
    m:list
    v:list
    step_count:int=0
    @classmethod
    def fresh(cls):return cls([0.0]*8,[0.0]*8,0)
    @classmethod
    def from_dict(cls,x):
        if not isinstance(x,dict) or set(x)!={'m','v','step_count'} or type(x['step_count'])!=int or x['step_count']<0:
            raise PAWError('Invalid Adam state')
        if not isinstance(x['m'],list) or not isinstance(x['v'],list) or len(x['m'])!=8 or len(x['v'])!=8:
            raise PAWError('Invalid Adam moment shape')
        m=[finite(v) for v in x['m']];v=[finite(a) for a in x['v']]
        if any(a<0 for a in v):raise PAWError('Negative Adam second moment')
        return cls(m,v,x['step_count'])
    def to_dict(self):return {'m':self.m[:],'v':self.v[:],'step_count':self.step_count}
    def update(self,p,g,lr=config.LEARNING_RATE):
        norm=math.sqrt(sum(v*v for v in g))
        if norm>config.GRADIENT_CLIP:g=[v*config.GRADIENT_CLIP/norm for v in g]
        self.step_count+=1
        t=self.step_count
        out=[]
        for i in range(8):
            self.m[i]=.9*self.m[i]+.1*g[i]
            self.v[i]=.999*self.v[i]+.001*g[i]*g[i]
            out.append(finite(p[i]-lr*(self.m[i]/(1-.9**t))/(math.sqrt(self.v[i]/(1-.999**t))+1e-8)))
        return project(out)
