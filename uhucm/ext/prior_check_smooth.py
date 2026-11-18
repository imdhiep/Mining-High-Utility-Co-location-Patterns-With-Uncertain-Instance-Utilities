"""Coverage of the posterior under the NPMLE prior, the convolution-smoothed NPMLE (Gu and Koenker 2023, Sec. 6.3,
Silverman bandwidth) and the Efron g-model, all with a local fine grid for sharp likelihoods. Reuses prior_check.py."""
import numpy as np
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"prior_check.py")).read().split("rng=np.random.default_rng(1)")[0])
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"prior_check.py")).read().split("print('--- g-model + local fine grid for sharp likelihoods')")[1].split("rng=np.random.default_rng(1)")[0])
def smooth(th,w,n):
    m=w@th; sd=np.sqrt(max(w@th**2-m*m,1e-12)); h=0.9*sd*n**(-0.2)
    K=np.exp(-0.5*((th[:,None]-th[None,:])/h)**2); ws=K@w; return ws/ws.sum()
rng=np.random.default_rng(1)
from scipy.special import digamma
for law in ('lognormal','gamma','mixture'):
  for loc,scl in ((0.7,1.0),(3.0,1.5),(4.5,1.8)):
    n=1500; z0=rng.normal(size=n)
    if law=='gamma': k=1/scl**2; lt=np.log(rng.gamma(k,1,size=n))-digamma(k)+loc
    elif law=='mixture': lt=loc+scl*z0+1.5*scl*(rng.random(n)<0.3)
    else: lt=loc+scl*z0
    x=rng.poisson(np.exp(lt)).astype(float); th=grid(x,1.0,300); L=lik(x,th,1.0); w0=npmle(L)
    out=[]
    for nm,w in (('npmle',w0),('smoothed npmle',smooth(th,w0,n)),('gmodel',gmodel(L,th))):
        m,v=post_local(x,1.0,th,w); z=(lt-m)/np.sqrt(v); hi=x>=100
        out.append(f"{nm}: {(np.abs(z)<1.96).mean():.3f}/{(np.abs(z[hi])<1.96).mean() if hi.any() else float('nan'):.3f}")
    print(law,loc,scl,'|',' | '.join(out))
