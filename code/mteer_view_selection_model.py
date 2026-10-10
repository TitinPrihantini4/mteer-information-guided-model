from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.stats import qmc, rankdata, pearsonr
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / 'data'
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
(OUT/'figures').mkdir(exist_ok=True)
SEED=20260903
rng=np.random.default_rng(SEED)

# -----------------------------------------------------------------------------
# Literature-derived clinical anchors. These are not fitted model coefficients.
# -----------------------------------------------------------------------------
ANCHORS = [
    dict(parameter='EVEREST favorable coaptation length', value=2.0, unit='mm', role='Minimum historical coaptation-length anchor for graspability', status='literature-derived', source='MitraClip patient selection / EVEREST criteria', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6288214/'),
    dict(parameter='EVEREST unfavorable coaptation depth', value=11.0, unit='mm', role='Historical tethering-depth complexity anchor', status='literature-derived', source='MitraClip patient selection / EVEREST criteria', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6288214/'),
    dict(parameter='EVEREST flail gap boundary', value=10.0, unit='mm', role='Historical degenerative-MR anatomical complexity anchor', status='literature-derived', source='MitraClip patient selection / EVEREST criteria', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6288214/'),
    dict(parameter='EVEREST flail width boundary', value=15.0, unit='mm', role='Historical degenerative-MR anatomical complexity anchor', status='literature-derived', source='MitraClip patient selection / EVEREST criteria', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6288214/'),
    dict(parameter='Mobile/posterior leaflet length favorable anchor', value=10.0, unit='mm', role='Leaflet tissue/graspability anchor', status='literature-derived', source='TEER echocardiographic eligibility reviews', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC7190777/'),
    dict(parameter='Mitral valve area historical anchor', value=4.0, unit='cm^2', role='Historical stenosis-risk anchor', status='literature-derived', source='EVEREST/TEER selection literature', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC9146624/'),
    dict(parameter='Transmitral mean-gradient reference boundary', value=5.0, unit='mmHg', role='Common post-TEER hemodynamic stratification anchor', status='literature-derived', source='CLASP IID / EXPANDed / registry analyses', url='https://pubmed.ncbi.nlm.nih.gov/42233921/'),
    dict(parameter='Residual 3D-VCA trace/mild boundary', value=0.10, unit='cm^2', role='Intraprocedural residual-MR quantitative anchor', status='literature-derived', source='MITRA-PRO registry', url='https://pubmed.ncbi.nlm.nih.gov/39652118/'),
    dict(parameter='Residual 3D-VCA relevant-MR boundary', value=0.30, unit='cm^2', role='Intraprocedural residual-MR quantitative anchor', status='literature-derived', source='MITRA-PRO registry', url='https://pubmed.ncbi.nlm.nih.gov/39652118/'),
    dict(parameter='Coaptation-gap prevalence in FMR TEER cohort', value=10.0, unit='%', role='Prevalence/context anchor, not a model probability', status='literature-derived', source='Coaptation gap study, n=2140', url='https://pubmed.ncbi.nlm.nih.gov/40921461/'),
    dict(parameter='AFMR 3D-TEE procedural failure residual MR count', value=31, unit='of 130', role='Outcome-context anchor', status='literature-derived', source='Quantitative 3D TEE AFMR study', url='https://pubmed.ncbi.nlm.nih.gov/42336064/'),
    dict(parameter='AFMR 3D-TEE gradient >=5 count', value=35, unit='of 130', role='Outcome-context anchor', status='literature-derived', source='Quantitative 3D TEE AFMR study', url='https://pubmed.ncbi.nlm.nih.gov/42336064/'),
    dict(parameter='TLA/MAA AUC for single-device success', value=0.79, unit='AUC', role='3D geometric-information context anchor', status='literature-derived', source='Quantitative 3D TEE AFMR study', url='https://pubmed.ncbi.nlm.nih.gov/42336064/'),
    dict(parameter='CLASP IID optimal hemodynamic result', value=72.5, unit='%', role='Residual MR/gradient trade-off context anchor', status='literature-derived', source='CLASP IID 1-year analysis', url='https://pubmed.ncbi.nlm.nih.gov/42233921/'),
    dict(parameter='OCEAN-Mitral residual MR <=2+ at 1 year', value=94.1, unit='%', role='Asian registry durability context anchor', status='literature-derived', source='OCEAN-Mitral registry', url='https://pubmed.ncbi.nlm.nih.gov/37815039/'),
    dict(parameter='MITRA-PRO relevant residual VCA prevalence', value=16.8, unit='%', role='Residual-MR context anchor', status='literature-derived', source='MITRA-PRO registry', url='https://pubmed.ncbi.nlm.nih.gov/39652118/'),
]

# -----------------------------------------------------------------------------
# State: x=[coaptation gap (mm), coaptation length (mm), residual 3D-VCA (cm2),
#           transmitral mean gradient (mmHg)].
# These priors/noise values are provisional structural assumptions for a proof
# of concept; they are NOT empirical measurement-error estimates.
# -----------------------------------------------------------------------------
STATE_NAMES=['Coaptation gap','Coaptation length','Residual 3D-VCA','Transmitral gradient']
UNITS=['mm','mm','cm^2','mmHg']
MU0=np.array([6.0, 3.6, 0.22, 3.8])
SD0=np.array([2.6, 1.4, 0.14, 1.5])
P0=np.diag(SD0**2)
Q_BASE=np.diag(np.array([0.12,0.08,0.015,0.10])**2) # between-acquisition perturbation

# Candidate echo views and their information geometry. H maps the latent state
# to observables. R contains provisional measurement-noise variances.
VIEWS={
    'Intercommissural': dict(H=np.array([[1.0,0.15,0.0,0.0]]), R=np.diag([1.20**2]), cost=1.00, phase=np.array([1.3,1.1,0.4,0.2])),
    'LVOT long-axis': dict(H=np.array([[0.20,1.0,0.0,0.0]]), R=np.diag([0.75**2]), cost=1.05, phase=np.array([1.0,1.35,0.6,0.2])),
    'Biplane X-plane': dict(H=np.array([[0.90,0.25,0.0,0.0],[0.20,0.90,0.0,0.0]]), R=np.diag([0.85**2,0.60**2]), cost=1.25, phase=np.array([1.2,1.45,0.8,0.3])),
    '3D en-face': dict(H=np.array([[0.95,0.35,0.05,0.0],[0.10,0.65,0.0,0.0]]), R=np.diag([0.65**2,0.70**2]), cost=1.45, phase=np.array([1.45,1.2,0.9,0.3])),
    '3D MPR': dict(H=np.array([[1.0,0.20,0.0,0.0],[0.15,1.0,0.0,0.0]]), R=np.diag([0.48**2,0.45**2]), cost=1.70, phase=np.array([1.30,1.60,1.0,0.4])),
    '3D color VCA': dict(H=np.array([[0.0,0.0,1.0,0.05]]), R=np.diag([0.045**2]), cost=1.35, phase=np.array([0.15,0.25,1.15,1.65])),
    'CW Doppler': dict(H=np.array([[0.0,0.0,0.05,1.0]]), R=np.diag([0.40**2]), cost=0.90, phase=np.array([0.10,0.20,0.85,1.50])),
}
VIEW_NAMES=list(VIEWS)
PHASES=['Targeting','Alignment','Grasping','Result assessment']
CLIN_W=np.array([0.23,0.27,0.31,0.19])
SD_TARGET=np.array([0.65,0.50,0.055,0.55])
MAX_STEPS=8
COST_WEIGHT=0.18


def stable_logdet(A):
    sign,ld=np.linalg.slogdet(A)
    if sign<=0: raise ValueError('Non-positive definite covariance')
    return ld

def entropy_gaussian(P):
    d=P.shape[0]
    return 0.5*(d*(1+np.log(2*np.pi))+stable_logdet(P))

def predict_cov(P,q_scale=1.0):
    return P + q_scale*Q_BASE

def kalman_update(mu,P,H,R,y=None):
    S=H@P@H.T+R
    K=P@H.T@np.linalg.inv(S)
    Pn=(np.eye(len(mu))-K@H)@P
    Pn=(Pn+Pn.T)/2
    if y is None: return mu,Pn
    mun=mu+K@(y-H@mu)
    return mun,Pn

def information_gain(P,H,R):
    _,Pn=kalman_update(np.zeros(P.shape[0]),P,H,R,None)
    return entropy_gaussian(P)-entropy_gaussian(Pn)

def weighted_uncertainty(P):
    return float(np.dot(CLIN_W,np.sqrt(np.maximum(np.diag(P),0))/SD0))

def action_utility(P,view,phase,cost_weight=COST_WEIGHT,clinical_weights=CLIN_W,q_scale=1.0):
    vp=VIEWS[view]
    Pp=predict_cov(P,q_scale)
    _,Pn=kalman_update(np.zeros(4),Pp,vp['H'],vp['R'],None)
    before=np.sqrt(np.diag(Pp))/SD0
    after=np.sqrt(np.diag(Pn))/SD0
    weighted_gain=float(np.dot(clinical_weights,before-after))
    ig=information_gain(Pp,vp['H'],vp['R'])
    return vp['phase'][phase]*(0.58*ig+0.42*weighted_gain)-cost_weight*vp['cost']

def phase_from_cov(P,step):
    s=np.sqrt(np.diag(P))
    if step==0 or s[0]>1.0: return 0
    if s[1]>0.75: return 1
    if s[0]>0.70 or s[1]>0.50: return 2
    return 3

def stop_condition(P,step):
    s=np.sqrt(np.diag(P))
    return bool(step>=3 and np.all(s<=SD_TARGET))

def choose_view(P,phase,policy='full',cost_weight=COST_WEIGHT,clinical_weights=CLIN_W,q_scale=1.0):
    if policy=='fixed':
        seq=['3D en-face','Biplane X-plane','3D MPR','LVOT long-axis','3D color VCA','CW Doppler','3D color VCA','CW Doppler']
        return seq[min(choose_view.counter,len(seq)-1)]
    scores={}
    for v in VIEW_NAMES:
        vp=VIEWS[v]; Pp=predict_cov(P,q_scale)
        if policy=='entropy': sc=information_gain(Pp,vp['H'],vp['R'])-cost_weight*vp['cost']
        elif policy=='variance':
            _,Pn=kalman_update(np.zeros(4),Pp,vp['H'],vp['R'])
            sc=float(np.trace(Pp)-np.trace(Pn))-cost_weight*vp['cost']
        elif policy=='no_phase': sc=action_utility(P,v,3,cost_weight,clinical_weights,q_scale)
        else: sc=action_utility(P,v,phase,cost_weight,clinical_weights,q_scale)
        scores[v]=sc
    return max(scores,key=scores.get)
choose_view.counter=0

def run_sequence(true_state=None,policy='full',rng=None,cost_weight=COST_WEIGHT,clinical_weights=CLIN_W,q_scale=1.0,max_steps=MAX_STEPS):
    rng=np.random.default_rng(SEED) if rng is None else rng
    if true_state is None: true_state=MU0.copy()
    mu=MU0.copy(); P=P0.copy(); rows=[]
    for step in range(max_steps):
        phase=phase_from_cov(P,step)
        if stop_condition(P,step): break
        if policy=='fixed': choose_view.counter=step
        v=choose_view(P,phase,policy,cost_weight,clinical_weights,q_scale)
        vp=VIEWS[v]; Pp=predict_cov(P,q_scale)
        y=vp['H']@true_state+rng.multivariate_normal(np.zeros(vp['R'].shape[0]),vp['R'])
        h0=entropy_gaussian(Pp); mu,P=kalman_update(mu,Pp,vp['H'],vp['R'],y); h1=entropy_gaussian(P)
        rows.append(dict(step=step+1,phase=PHASES[phase],view=v,entropy_before=h0,entropy_after=h1,information_gain=h0-h1,
                         weighted_uncertainty=weighted_uncertainty(P),sd_gap=np.sqrt(P[0,0]),sd_coaptation=np.sqrt(P[1,1]),sd_vca=np.sqrt(P[2,2]),sd_gradient=np.sqrt(P[3,3]),
                         mu_gap=mu[0],mu_coaptation=mu[1],mu_vca=mu[2],mu_gradient=mu[3]))
    return pd.DataFrame(rows),mu,P

# Reference sequence
ref_df,ref_mu,ref_P=run_sequence(MU0,policy='full',rng=np.random.default_rng(SEED))
ref_df.to_csv(DATA/'Reference_Sequence.csv',index=False)

# View utility matrix at initial covariance by phase
u=[]
for pi,pn in enumerate(PHASES):
    for v in VIEW_NAMES:
        u.append(dict(phase=pn,view=v,utility=action_utility(P0,v,pi),information_gain=information_gain(predict_cov(P0),VIEWS[v]['H'],VIEWS[v]['R'])))
pd.DataFrame(u).to_csv(DATA/'View_Utility_by_Phase.csv',index=False)

# Synthetic population-like scenarios strictly for stress testing, not patients
N=600
truths=np.column_stack([
    np.clip(rng.normal(6.0,2.2,N),0.5,12.0),
    np.clip(rng.normal(3.6,1.0,N),1.0,8.0),
    np.clip(rng.gamma(2.2,0.09,N),0.01,0.65),
    np.clip(rng.normal(4.0,1.25,N),1.0,8.5)])
policies=['fixed','variance','entropy','no_phase','full']
res=[]
for pol in policies:
    for i,x in enumerate(truths):
        df,mu,P=run_sequence(x,policy=pol,rng=np.random.default_rng(SEED+10000*i+policies.index(pol)))
        err=np.abs(mu-x)
        res.append(dict(policy=pol,scenario=i,n_views=len(df),terminal_entropy=entropy_gaussian(P),weighted_uncertainty=weighted_uncertainty(P),
                        abs_gap_error=err[0],abs_coaptation_error=err[1],abs_vca_error=err[2],abs_gradient_error=err[3],
                        completed=stop_condition(P,max(3,len(df)))))
policy_results=pd.DataFrame(res)
policy_results.to_csv(DATA/'Policy_Stress_Test.csv',index=False)
policy_summary=policy_results.groupby('policy').agg(n=('scenario','size'),median_views=('n_views','median'),mean_views=('n_views','mean'),terminal_entropy=('terminal_entropy','mean'),weighted_uncertainty=('weighted_uncertainty','mean'),
    gap_MAE=('abs_gap_error','mean'),coaptation_MAE=('abs_coaptation_error','mean'),VCA_MAE=('abs_vca_error','mean'),gradient_MAE=('abs_gradient_error','mean'),completion_rate=('completed','mean')).reset_index()
policy_summary.to_csv(DATA/'Policy_Summary.csv',index=False)

# Internal model recovery via stochastic softmax action labels on synthetic decision states.
# Models differ in utility representation; log-likelihood is evaluated against model-generated actions.
M=1200
states=[]; actions=[]
for i in range(M):
    scale=np.exp(rng.normal(0,0.42,4)); P=np.diag((SD0*scale)**2)
    phase=int(rng.integers(0,4))
    full_scores=np.array([action_utility(P,v,phase) for v in VIEW_NAMES])
    temp=0.18; probs=np.exp((full_scores-full_scores.max())/temp); probs/=probs.sum()
    a=int(rng.choice(len(VIEW_NAMES),p=probs))
    states.append((P,phase)); actions.append(a)

def variant_scores(P,phase,variant):
    vals=[]
    for v in VIEW_NAMES:
        vp=VIEWS[v]; Pp=predict_cov(P)
        if variant=='Cost only': sc=-vp['cost']
        elif variant=='Entropy only': sc=information_gain(Pp,vp['H'],vp['R'])-COST_WEIGHT*vp['cost']
        elif variant=='Geometry weighted':
            w=np.array([0.50,0.50,0,0]); _,Pn=kalman_update(np.zeros(4),Pp,vp['H'],vp['R']); sc=np.dot(w,np.sqrt(np.diag(Pp))/SD0-np.sqrt(np.diag(Pn))/SD0)-COST_WEIGHT*vp['cost']
        elif variant=='Clinical weights no phase': sc=action_utility(P,v,3)
        else: sc=action_utility(P,v,phase)
        vals.append(sc)
    return np.array(vals)
variants=['Cost only','Entropy only','Geometry weighted','Clinical weights no phase','Full phase-aware']
mc=[]
for var in variants:
    ll=0.0; correct=0
    k={'Cost only':1,'Entropy only':2,'Geometry weighted':3,'Clinical weights no phase':5,'Full phase-aware':9}[var]
    for (P,phase),a in zip(states,actions):
        s=variant_scores(P,phase,var); temp=0.18; pr=np.exp((s-s.max())/temp); pr/=pr.sum(); ll+=np.log(max(pr[a],1e-15)); correct+=int(np.argmax(pr)==a)
    mc.append(dict(model=var,k=k,log_likelihood=ll,AIC=2*k-2*ll,BIC=k*np.log(M)-2*ll,internal_action_accuracy=correct/M))
mc=pd.DataFrame(mc).sort_values('AIC')
mc.to_csv(DATA/'Synthetic_Model_Comparison_AIC.csv',index=False)

# Uncertainty equilibrium / stability of repeated covariance maps with process noise.
def cov_map(P,view,q_scale=1.0):
    vp=VIEWS[view]; Pp=predict_cov(P,q_scale); _,Pn=kalman_update(np.zeros(4),Pp,vp['H'],vp['R']); return Pn

def vec_sym(P):
    idx=np.triu_indices(4); return P[idx]
def mat_sym(v):
    P=np.zeros((4,4)); idx=np.triu_indices(4); P[idx]=v; P[(idx[1],idx[0])]=v; return P

eq=[]; eigrows=[]
for v in VIEW_NAMES:
    P=P0.copy()
    for _ in range(2000):
        Pn=cov_map(P,v)
        if np.max(np.abs(Pn-P))<1e-12: break
        P=Pn
    eq.append(dict(view=v,iterations=_+1,trace_cov=np.trace(P),entropy=entropy_gaussian(P),sd_gap=np.sqrt(P[0,0]),sd_coaptation=np.sqrt(P[1,1]),sd_vca=np.sqrt(P[2,2]),sd_gradient=np.sqrt(P[3,3])))
    base=vec_sym(P); eps=1e-6; J=np.zeros((10,10))
    for j in range(10):
        vv=base.copy(); vv[j]+=eps; Pj=mat_sym(vv); out=vec_sym(cov_map(Pj,v)); J[:,j]=(out-vec_sym(cov_map(P,v)))/eps
    ev=np.linalg.eigvals(J); eigrows.append(dict(view=v,spectral_radius=max(abs(ev)),max_real_eigenvalue=max(ev.real)))
pd.DataFrame(eq).to_csv(DATA/'Covariance_Equilibria.csv',index=False)
pd.DataFrame(eigrows).to_csv(DATA/'Covariance_Map_Stability.csv',index=False)

# LHS global sensitivity: structural settings -> reference policy outputs
param_names=['cost_weight','q_scale','w_gap','w_coaptation','w_vca','w_gradient','target_gap','target_coaptation','target_vca','target_gradient']
ranges=np.array([[0.08,0.32],[0.50,1.80],[0.12,0.34],[0.16,0.38],[0.20,0.42],[0.10,0.30],[0.45,0.90],[0.32,0.65],[0.035,0.080],[0.35,0.80]])
L=500
sam=qmc.LatinHypercube(d=len(param_names),seed=SEED).random(L); vals=qmc.scale(sam,ranges[:,0],ranges[:,1])
lhs=[]
for i,row in enumerate(vals):
    cw,qs,*rest=row; w=np.array(rest[:4]); w=w/w.sum(); targets=np.array(rest[4:])
    # local variant of run_sequence with target override through explicit loop
    mu=MU0.copy(); P=P0.copy(); n=0; totig=0
    rr=np.random.default_rng(SEED+50000+i)
    for step in range(MAX_STEPS):
        ph=phase_from_cov(P,step)
        s=np.sqrt(np.diag(P))
        if step>=3 and np.all(s<=targets): break
        v=choose_view(P,ph,'full',cw,w,qs); vp=VIEWS[v]; Pp=predict_cov(P,qs); y=vp['H']@MU0+rr.multivariate_normal(np.zeros(vp['R'].shape[0]),vp['R'])
        h0=entropy_gaussian(Pp); mu,P=kalman_update(mu,Pp,vp['H'],vp['R'],y); totig+=h0-entropy_gaussian(P); n+=1
    lhs.append(dict(sample=i,**{k:row[j] for j,k in enumerate(param_names)},n_views=n,terminal_entropy=entropy_gaussian(P),weighted_uncertainty=float(np.dot(w,np.sqrt(np.diag(P))/SD0)),total_information_gain=totig))
lhs=pd.DataFrame(lhs); lhs.to_csv(DATA/'LHS_Parameter_Samples.csv',index=False)

# PRCC via residualization of ranks
def prcc(df,xcols,ycol):
    X=np.column_stack([rankdata(df[c]) for c in xcols]); y=rankdata(df[ycol]); out=[]
    for j,c in enumerate(xcols):
        others=np.delete(X,j,axis=1); A=np.column_stack([np.ones(len(df)),others]); bx=np.linalg.lstsq(A,X[:,j],rcond=None)[0]; by=np.linalg.lstsq(A,y,rcond=None)[0]
        rx=X[:,j]-A@bx; ry=y-A@by; r,p=pearsonr(rx,ry); out.append(dict(parameter=c,output=ycol,PRCC=r,p_value=p))
    return out
pr=[]
for y in ['n_views','terminal_entropy','weighted_uncertainty','total_information_gain']:
    pr.extend(prcc(lhs,param_names,y))
pd.DataFrame(pr).to_csv(DATA/'Global_Sensitivity_PRCC.csv',index=False)

# Local elasticity around reference settings; use deterministic truth and same RNG seed.
base_metrics=dict(n_views=len(ref_df),terminal_entropy=entropy_gaussian(ref_P),weighted_uncertainty=weighted_uncertainty(ref_P))
loc=[]
for pname in ['cost_weight','q_scale']:
    for sign in [-1,1]:
        factor=1+0.01*sign; cw=COST_WEIGHT*factor if pname=='cost_weight' else COST_WEIGHT; qs=1.0*factor if pname=='q_scale' else 1.0
        df,mu,P=run_sequence(MU0,'full',np.random.default_rng(SEED),cw,CLIN_W,qs)
        metrics=dict(n_views=len(df),terminal_entropy=entropy_gaussian(P),weighted_uncertainty=weighted_uncertainty(P))
        for y,val in metrics.items():
            elasticity=((val-base_metrics[y])/base_metrics[y])/(factor-1) if base_metrics[y]!=0 else np.nan
            loc.append(dict(parameter=pname,perturbation=sign,output=y,value=val,elasticity=elasticity))
pd.DataFrame(loc).to_csv(DATA/'Local_Elasticity.csv',index=False)

# Scenario stress tests
scenarios=[
 ('Reference',1.0,COST_WEIGHT,CLIN_W),('High image noise',1.8,COST_WEIGHT,CLIN_W),('Low image noise',0.55,COST_WEIGHT,CLIN_W),
 ('High acquisition penalty',1.0,0.30,CLIN_W),('Low acquisition penalty',1.0,0.08,CLIN_W),
 ('Residual-MR priority',1.0,COST_WEIGHT,np.array([.15,.20,.50,.15])),('Geometry priority',1.0,COST_WEIGHT,np.array([.40,.40,.12,.08])),
 ('Hemodynamic priority',1.0,COST_WEIGHT,np.array([.15,.15,.25,.45]))]
sc=[]
for name,qs,cw,w in scenarios:
    # q_scale here acts as latent process volatility, while image-noise labels are broad stress labels; preserve explicit caveat
    df,mu,P=run_sequence(MU0,'full',np.random.default_rng(SEED),cw,w/w.sum(),qs)
    sc.append(dict(scenario=name,n_views=len(df),terminal_entropy=entropy_gaussian(P),weighted_uncertainty=float(np.dot(w/w.sum(),np.sqrt(np.diag(P))/SD0)),last_view=df.view.iloc[-1] if len(df) else 'None',last_phase=df.phase.iloc[-1] if len(df) else 'None'))
pd.DataFrame(sc).to_csv(DATA/'Scenario_Results.csv',index=False)

# Policy surface: initial coaptation-gap SD vs residual-VCA SD and selected first view
surf=[]
for sg in np.linspace(0.5,4.0,36):
    for sv in np.linspace(0.02,0.25,36):
        P=P0.copy(); P[0,0]=sg**2; P[2,2]=sv**2; ph=0
        scores={v:action_utility(P,v,ph) for v in VIEW_NAMES}; vv=max(scores,key=scores.get)
        surf.append(dict(sd_gap=sg,sd_vca=sv,selected_view=vv,max_utility=scores[vv],entropy=entropy_gaussian(P)))
pd.DataFrame(surf).to_csv(DATA/'Sensitivity_Surface.csv',index=False)

# Literature source ledger (18 sources)
sources=[
('Little SH et al. Guidelines for Intraprocedural Imaging for M-TEER. JASE/ASE. 2026.','Defines required 2D, biplane, 3D volume and 3D-MPR procedural imaging content; notes absence of a standard directing algorithm.','https://www.asecho.org/guideline/intraprocedural-imaging-for-mitral-valve-transcatheter-edge-to-edge-repair/'),
('MitraClip patient selection inclusion and exclusion criteria for optimal outcomes. Ann Cardiothorac Surg. 2018.','EVEREST anchors for coaptation length, coaptation depth, flail gap/width and valve area.','https://pmc.ncbi.nlm.nih.gov/articles/PMC6288214/'),
('MitraClip a novel percutaneous approach to mitral valve repair. 2011.','Historical coaptation and flail anatomy anchors; role of online 3D TEE.','https://pmc.ncbi.nlm.nih.gov/articles/PMC3150716/'),
('Percutaneous mitral valve repair the MitraClip device. 2017.','Leaflet tissue, coaptation and valve-area considerations for grasping and stenosis avoidance.','https://pmc.ncbi.nlm.nih.gov/articles/PMC5642839/'),
('Transcatheter Treatment of Mitral Regurgitation. 2022.','Summarizes EVEREST and COAPT echocardiographic criteria including gradient and leaflet length.','https://pmc.ncbi.nlm.nih.gov/articles/PMC9146624/'),
('Mitral valve transcatheter edge-to-edge repair. EuroIntervention review. 2023.','Standard intercommissural/LVOT X-plane guidance and systematic residual MR/gradient assessment.','https://pmc.ncbi.nlm.nih.gov/articles/PMC9869401/'),
('Fan Y et al. Advances in Procedural Echocardiographic Imaging in TEER. Front Cardiovasc Med. 2022.','3D, unlimited X-plane, live 3D-MPR and advanced procedural guidance.','https://pmc.ncbi.nlm.nih.gov/articles/PMC8996060/'),
('Screening and Procedural Guidance for M-TEER. 2025.','Stepwise view use for transseptal puncture, navigation, grasping and outcome assessment.','https://pmc.ncbi.nlm.nih.gov/articles/PMC12295293/'),
('Prognostic value of leaflet coaptation gap in TEER for functional MR. Heart. 2026.','2140-patient cohort; coaptation gap present in 10%, supporting gap as a clinically meaningful geometry state.','https://pubmed.ncbi.nlm.nih.gov/40921461/'),
('Quantitative 3D TEE Analysis of Mitral Valve Geometry for M-TEER Strategy in AFMR. JASE. 2026.','130-patient study; TLA/MAA associated with residual MR, device count, gradient; AUC 0.79 for single-device success.','https://pubmed.ncbi.nlm.nih.gov/42336064/'),
('Intraprocedural 3D-VCA predicts survival after TEER MITRA-PRO registry. 2024.','Residual 3D-VCA categories <0.1, 0.1-<0.3, and >=0.3 cm2; quantitative residual-MR anchor.','https://pubmed.ncbi.nlm.nih.gov/39652118/'),
('Impact of Intraprocedural MR and Gradient Following TEER for Primary MR. JACC Cardiovasc Interv. 2024.','1509 patients; residual MR >=2+ associated with worse outcomes; gradient >5 mmHg not independently associated in primary MR.','https://pubmed.ncbi.nlm.nih.gov/38986655/'),
('Using Transmitral Pressure Gradients and Residual MR to Optimize Outcome After TEER. JACC. 2025/2026.','2360 FMR patients; joint MR/gradient outcome strata and continuous gradient risk relationship.','https://pubmed.ncbi.nlm.nih.gov/41193089/'),
('Impact of Residual MR and Gradient After M-TEER CLASP IID. JACC Cardiovasc Interv. 2026.','284 patients; 72.5% met MR <=1+ and MVG <=5 mmHg optimal-result definition.','https://pubmed.ncbi.nlm.nih.gov/42233921/'),
('Combined Impact of Residual MR and Gradient After M-TEER EXPANDed. 2024.','1723 subjects; four hemodynamic result groups based on MR and mean gradient.','https://pubmed.ncbi.nlm.nih.gov/39453373/'),
('One-Year Outcomes and Residual MR OCEAN-Mitral Registry. JAHA. 2023.','2150-patient Asian-Pacific cohort; residual MR groups associated with death/HF hospitalization.','https://pubmed.ncbi.nlm.nih.gov/37815039/'),
('MiCLASP 1-Year Outcomes. JACC Cardiovasc Interv. 2024.','544-patient post-market cohort; sustained MR reduction and clinical outcomes context.','https://pubmed.ncbi.nlm.nih.gov/38599692/'),
('Goldberger MI et al. Comparison of 2D and 3D Echocardiographic Mitral Leaflet Length Measurements. J Cardiothorac Vasc Anesth. 2026.','Supports importance of millimeter-scale leaflet measurement and 3D-MPR for quantitative anatomy.','https://pubmed.ncbi.nlm.nih.gov/42476856/'),
]
pd.DataFrame(sources,columns=['reference','modeling_role','url']).to_csv(DATA/'Literature_Parameter_Sources.csv',index=False)

# Model parameters provenance
params=[]
for a in ANCHORS: params.append(dict(name=a['parameter'],value=a['value'],unit=a['unit'],status=a['status'],role=a['role'],source=a['source'],url=a['url']))
for i,n in enumerate(STATE_NAMES): params.append(dict(name='Prior mean '+n,value=MU0[i],unit=UNITS[i],status='provisional structural assumption',role='Reference synthetic-state center, not cohort mean',source='Chosen for literature-consistent proof-of-concept stress testing',url=''))
for i,n in enumerate(STATE_NAMES): params.append(dict(name='Prior SD '+n,value=SD0[i],unit=UNITS[i],status='provisional structural assumption',role='Initial epistemic uncertainty scale',source='Not estimated from patients',url=''))
for i,n in enumerate(STATE_NAMES): params.append(dict(name='Stopping SD '+n,value=SD_TARGET[i],unit=UNITS[i],status='provisional decision guardrail',role='Proof-of-concept uncertainty tolerance',source='Not a clinical cutoff',url=''))
params.append(dict(name='Acquisition cost weight',value=COST_WEIGHT,unit='dimensionless',status='provisional structural assumption',role='Information-versus-acquisition penalty',source='Financial-engineering-inspired utility term',url=''))
pd.DataFrame(params).to_csv(DATA/'Model_Parameters.csv',index=False)

summary={
 'seed':SEED,
 'reference':{'n_views':int(len(ref_df)),'view_sequence':ref_df.view.tolist(),'terminal_entropy':float(entropy_gaussian(ref_P)),'weighted_uncertainty':float(weighted_uncertainty(ref_P)),
              'terminal_sd':{STATE_NAMES[i]:float(np.sqrt(ref_P[i,i])) for i in range(4)},'posterior_mean':{STATE_NAMES[i]:float(ref_mu[i]) for i in range(4)},'total_information_gain':float(ref_df.information_gain.sum())},
 'policy_summary':policy_summary.to_dict(orient='records'),
 'model_comparison':mc.to_dict(orient='records'),
 'top_prcc_terminal_entropy':pd.DataFrame(pr).query("output=='terminal_entropy'").assign(absPRCC=lambda d:d.PRCC.abs()).sort_values('absPRCC',ascending=False).head(6).drop(columns='absPRCC').to_dict(orient='records'),
 'covariance_stability':pd.read_csv(DATA/'Covariance_Map_Stability.csv').to_dict(orient='records')
}
json.dump(summary,open(OUT/'Analysis_Summary.json','w'),indent=2)

print(json.dumps(summary,indent=2))

# Complementary-cycle covariance stability (important because single views leave unobserved dimensions).
def _cycle_map(P,cycle):
    for vv in cycle:
        P=cov_map(P,vv)
    return P
_cycles=[('Geometry cycle',['3D MPR','Biplane X-plane']),('Outcome cycle',['3D color VCA','CW Doppler']),('Full complementary cycle',['3D MPR','3D color VCA','CW Doppler'])]
_cycle_rows=[]
for _name,_cyc in _cycles:
    _P=P0.copy()
    for _it in range(5000):
        _Pn=_cycle_map(_P,_cyc)
        if np.max(np.abs(_Pn-_P))<1e-12: break
        _P=_Pn
    _base=vec_sym(_P); _eps=1e-6; _J=np.zeros((10,10)); _f0=vec_sym(_cycle_map(_P,_cyc))
    for _j in range(10):
        _vv=_base.copy(); _vv[_j]+=_eps
        _J[:,_j]=(vec_sym(_cycle_map(mat_sym(_vv),_cyc))-_f0)/_eps
    _ev=np.linalg.eigvals(_J)
    _cycle_rows.append(dict(cycle=_name,views=' -> '.join(_cyc),iterations=_it+1,trace_cov=np.trace(_P),entropy=entropy_gaussian(_P),spectral_radius=max(abs(_ev))))
pd.DataFrame(_cycle_rows).to_csv(OUT/'Covariance_Cycle_Stability.csv',index=False)
