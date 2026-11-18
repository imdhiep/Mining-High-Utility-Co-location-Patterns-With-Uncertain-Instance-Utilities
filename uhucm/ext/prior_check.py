"""Coverage of the NPMLE posterior and of a smooth (Efron g-model) prior on simulated Poisson counts, with and without a
local fine grid. Root-cause check for the under-estimated posterior variance at high counts."""
import numpy as np, sys
from scipy.special import gammaln
from scipy.optimize import minimize
sys.path.insert(0,'.')
def grid(x,E,J):
    lo=np.log((x.min()+0.5)/E)-2; hi=np.log((x.max()+0.5)/E)+1; return np.linspace(lo,hi,J)
def lik(x,th,E):
    ll=x[:,None]*th[None,:]-E*np.exp(th)[None,:]-gammaln(x+1)[:,None]; return np.exp(ll-ll.max(1,keepdims=True))
def npmle(L,iters=300):
    w=np.full(L.shape[1],1/L.shape[1])
    for _ in range(iters):
        p=L*w; p/=p.sum(1,keepdims=True); w=np.maximum(p.mean(0),1e-300)
    return w
def gmodel(L,th,df=7,c0=1.0):
    t=(th-th.mean())/th.std(); Q=np.vstack([t**k for k in range(1,df+1)]).T; Q=(Q-Q.mean(0))/Q.std(0)
    def f(a):
        g=Q@a; g=g-g.max(); w=np.exp(g); w/=w.sum(); fx=L@w
        nll=-np.log(fx).sum()+c0*np.sqrt(a@a)
        # gradient
        P=(L*w)/fx[:,None]; gw=-(P.sum(0)-len(fx)*w)  # d nll / d g
        return nll, Q.T@gw + c0*a/max(np.sqrt(a@a),1e-12)
    r=minimize(f,np.zeros(df),jac=True,method='L-BFGS-B'); g=Q@r.x; g-=g.max(); w=np.exp(g); return w/w.sum()
def post(L,th,w):
    p=L*w; p/=p.sum(1,keepdims=True); m=p@th; v=np.maximum(p@th**2-m**2,1e-12); return m,v
rng=np.random.default_rng(1)
for law in ('lognormal','gamma','mixture'):
  for loc,scl in ((0.7,1.0),(3.0,1.5),(4.5,1.8)):
    n=1500; z0=rng.normal(size=n)
    if law=='gamma':
        from scipy.special import digamma
        k=1/scl**2; lt=np.log(rng.gamma(k,1,size=n))-digamma(k)+loc
    elif law=='mixture': lt=loc+scl*z0+1.5*scl*(rng.random(n)<0.3)
    else: lt=loc+scl*z0
    x=rng.poisson(np.exp(lt)).astype(float); th=grid(x,1.0,300); L=lik(x,th,1.0)
    out=[]
    for nm,w in (('npmle',npmle(L)),('gmodel',gmodel(L,th))):
        m,v=post(L,th,w); z=(lt-m)/np.sqrt(v); hi=x>=100
        out.append(f"{nm}: cover {(np.abs(z)<1.96).mean():.3f} cover(x>=100) {(np.abs(z[hi])<1.96).mean() if hi.any() else float('nan'):.3f} rmse {np.sqrt(((lt-m)**2).mean()):.3f}")
    print(law,loc,scl,'median x',np.median(x),'|',' | '.join(out))
print('--- g-model + local fine grid for sharp likelihoods')
def post_local(x,E,th,w):
    lg=np.log(np.maximum(w,1e-300)); h=th[1]-th[0]; m=np.zeros(len(x)); v=np.zeros(len(x))
    L=lik(x,th,E); p=L*w; p/=p.sum(1,keepdims=True); m[:]=p@th; v[:]=np.maximum(p@th**2-m**2,1e-12)
    for i,xi in enumerate(x):
        sd=1/np.sqrt(xi+0.5)
        if sd<5*h:
            c=np.log((xi+0.5)/E); t=np.linspace(c-7*sd,c+7*sd,201); g=np.interp(t,th,lg)
            ll=xi*t-E*np.exp(t)+g; q=np.exp(ll-ll.max()); q/=q.sum(); m[i]=q@t; v[i]=max(q@t**2-m[i]**2,1e-12)
    return m,v
rng=np.random.default_rng(1)
for law in ('lognormal','gamma','mixture'):
  for loc,scl in ((0.7,1.0),(3.0,1.5),(4.5,1.8)):
    n=1500; z0=rng.normal(size=n)
    if law=='gamma':
        from scipy.special import digamma
        k=1/scl**2; lt=np.log(rng.gamma(k,1,size=n))-digamma(k)+loc
    elif law=='mixture': lt=loc+scl*z0+1.5*scl*(rng.random(n)<0.3)
    else: lt=loc+scl*z0
    x=rng.poisson(np.exp(lt)).astype(float); th=grid(x,1.0,300); L=lik(x,th,1.0)
    out=[]
    for nm,w in (('npmle',npmle(L)),('gmodel',gmodel(L,th))):
        m,v=post_local(x,1.0,th,w); z=(lt-m)/np.sqrt(v); hi=x>=100
        out.append(f"{nm}+local: cover {(np.abs(z)<1.96).mean():.3f} cover(x>=100) {(np.abs(z[hi])<1.96).mean() if hi.any() else float('nan'):.3f} rmse {np.sqrt(((lt-m)**2).mean()):.3f}")
    print(law,loc,scl,'|',' | '.join(out))
