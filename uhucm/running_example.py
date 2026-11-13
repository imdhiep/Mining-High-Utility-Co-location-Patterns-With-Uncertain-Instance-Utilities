"""Running example of Sections III-V (Tables I-III): patterns, scores, FUB removal, feature order with the
non-positive features last, prefix scan and suffix bounds. Usage: python running_example.py [gamma]  (default 0.395)."""
import itertools, math, sys
from scipy.stats import norm
I={'S1':(300,-1.6,.040),'R1':(800,5.8,.023),'B1':(200,3.4,.059),'G1':(30,-1.4,.140),'W1':(10,-1.4,.194),'M1':(500,3.4,.031),
   'R2':(100,-0.6,.082),'B2':(80,-1.6,.091),'M2':(2,-0.5,.500),'R3':(50,1.6,.106),'B3':(60,-2.6,.095),'V1':(40,0.5,.122),
   'S2':(20,-1.9,.170),'M3':(150,2.6,.071)}            # instance: (number of reviews, posterior mean, posterior variance), Table I
mu={k:v[1] for k,v in I.items()}; s2={k:v[2] for k,v in I.items()}
C=[['R1','S1','B1','G1'],['R1','G1','W1'],['R1','M1','W1'],['R2','B2','M2'],['R3','B3','V1'],['S2','M3']]
f=lambda o:o[0]; z=1.645; gam=float(sys.argv[1]) if len(sys.argv)>1 else 0.395
Z=sum(m*norm.cdf(m/math.sqrt(v))+math.sqrt(v)*norm.pdf(m/math.sqrt(v)) for m,v in zip(mu.values(),s2.values()))
print('Z',round(Z,3),'Zplus',sum(max(m,0) for m in mu.values()))
feats=sorted(set(map(f,I)))
def PI(P,cl=C):
    s=set()
    for c in cl:
        F={f(o):o for o in c}
        if set(P)<=set(F): s|={F[g] for g in P}
    return s
def score(A,zz): return (sum(mu[o] for o in A)-zz*math.sqrt(sum(s2[o] for o in A)))/Z
def SR(Iset,trace=False):
    pos=sorted([o for o in Iset if mu[o]>0], key=lambda o:s2[o]/mu[o]); best=0; sm=sv=0; rows=[]
    for o in pos:
        sm+=mu[o]; sv+=s2[o]; g=sm-z*math.sqrt(sv); best=max(best,g); rows.append((o,mu[o],s2[o],round(s2[o]/mu[o],4),round(sm,2),round(sv,3),round(g,3)))
    if trace:
        for r in rows: print('  ',r)
    return best
res={}
for k in range(2,6):
    for P in itertools.combinations(feats,k):
        A=PI(P)
        if A: res[P]=A
for P,A in sorted(res.items(), key=lambda x:-score(x[1],0)):
    M=sum(mu[o] for o in A); V=sum(s2[o] for o in A)
    if M>0: print(''.join(P), sorted(A), round(M,2), round(V,3), 'L0 %.3f L %.3f p %.4f'%(score(A,0),score(A,z),norm.cdf((M-gam*Z)/math.sqrt(V))))
print('point answer',[''.join(P) for P,A in res.items() if score(A,0)>=gam]); print('LB answer',[''.join(P) for P,A in res.items() if score(A,z)>=gam])
# FUB with iteration
Fs=set(feats); Ss=set(I)
while True:
    fub={}
    for g in Fs:
        N=set()
        for o in Ss:
            if f(o)!=g: continue
            N.add(o)
            for c in C:
                if o in c: N|={x for x in c if x in Ss}
        fub[g]=SR(N)/Z
    D={g for g in Fs if fub[g]<gam}
    print('FUB',{g:round(v,3) for g,v in sorted(fub.items())},'remove',D)
    if not D: break
    Fs-=D; Ss={o for o in Ss if f(o) in Fs}
npf={g for g in Fs if all(mu[o]<=0 for o in Ss if f(o)==g)}
order=sorted([g for g in Fs if g not in npf],key=lambda g:-fub[g])+sorted(npf,key=lambda g:-fub[g]); print('npf',npf,'order',order)
Cs=[[o for o in c if o in Ss] for c in C]; Cs=[c for c in Cs if len(c)>=2]
def cand(P,T):
    out=set()
    for c in Cs:
        F={f(o):o for o in c}
        if set(P)<=set(F) and set(F)&set(T): out|={F[g] for g in F if g in set(P)|set(T)}
    return out
for P in [('R',),('R','B'),('R','M'),('M',),('B',),('S',)]:
    last=max(order.index(g) for g in P); T=order[last+1:]
    print('node',P,'T',T); Ic=cand(P,T); b=SR(Ic,True); print(' PCB',round(b,3),round(b/Z,3),'mean',round(sum(max(mu[o],0) for o in Ic)/Z,3))
    for j,g in enumerate(T):
        Tg=T[j:]; Ig=cand(P,Tg); print('  SB',g,round(SR(Ig)/Z,3))
