from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE=Path('/mnt/data/echo360_mteer_information_guided_package'); FIG=BASE/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9.5,'axes.titlesize':13,'axes.labelsize':10,'figure.dpi':160,'font.family':'DejaVu Sans'})

def save(name):
    plt.savefig(FIG/name,bbox_inches='tight',dpi=260); plt.close()

def table_figure(df,title,name,fontsize=7.3,figwidth=15,rowheight=.42):
    df=df.copy()
    h=max(2.8,1.35+len(df)*rowheight)
    fig,ax=plt.subplots(figsize=(figwidth,h)); ax.axis('off')
    tab=ax.table(cellText=df.values,colLabels=df.columns,loc='center',cellLoc='left',colLoc='left')
    tab.auto_set_font_size(False); tab.set_fontsize(fontsize); tab.scale(1,1.3)
    for (r,c),cell in tab.get_celld().items():
        if r==0:
            cell.set_facecolor('#17324d'); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
        elif r%2==0: cell.set_facecolor('#edf4f8')
        else: cell.set_facecolor('#ffffff')
        cell.set_edgecolor('#bdc9d4')
    ax.set_title(title,fontsize=15,weight='bold',pad=16)
    save(name)

# 1 architecture
fig,ax=plt.subplots(figsize=(15.5,8.5)); ax.axis('off')
boxes=[
(.02,.70,.20,.17,'Literature Anchors','Coaptation geometry\nLeaflet length\nResidual 3D-VCA\nTransmitral gradient'),
(.27,.70,.21,.17,'Latent Imaging State','Gap g\nCoaptation length c\nResidual VCA r\nGradient m'),
(.53,.70,.20,.17,'Belief-State Update','Bayesian posterior\nCovariance P(t)\nEntropy H(P)'),
(.78,.70,.20,.17,'View Allocation','Expected information gain\nClinical weighting\nAcquisition penalty'),
(.27,.39,.21,.18,'Procedural Regimes','Targeting\nAlignment\nGrasping\nResult assessment'),
(.53,.39,.20,.18,'Decision Layer','Bellman value\nContinue imaging\nversus stop'),
(.35,.08,.34,.19,'Robustness and Data Science','Five policy models | AIC/BIC recovery\nLHS-PRCC | stochastic stress testing\nRiccati covariance stability | sensitivity surface')]
colors=['#dceefe','#e3f5e8','#fff0c9','#ffe3df','#eee6ff','#f9e2f0','#dff4f3']
for (x,y,w,h,t,s),c in zip(boxes,colors):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.014,rounding_size=.018',fc=c,ec='#334155',lw=1.4); ax.add_patch(p)
    ax.text(x+w/2,y+h*.70,t,ha='center',va='center',weight='bold',fontsize=11); ax.text(x+w/2,y+h*.34,s,ha='center',va='center',fontsize=9)
for a,b in [((.22,.785),(.27,.785)),((.48,.785),(.53,.785)),((.73,.785),(.78,.785)),((.375,.70),(.375,.57)),((.63,.70),(.63,.57)),((.48,.48),(.53,.48)),((.375,.39),(.50,.27)),((.63,.39),(.56,.27))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.3,color='#475569'))
ax.text(.5,.95,'Information-Guided Intraprocedural M-TEER View-Selection Architecture',ha='center',weight='bold',fontsize=18)
ax.text(.5,.90,'Literature-parameterized computational proof-of-concept — no patient-level calibration',ha='center',fontsize=10.5,style='italic')
save('Figure_1_Model_Architecture.png')

# 2 reference sequence
r=pd.read_csv(BASE/'Reference_Sequence.csv')
fig,ax=plt.subplots(figsize=(12.8,7.2))
x=np.arange(1,len(r)+1); width=.16
series=[('Coaptation gap SD','sd_gap'),('Coaptation length SD','sd_coaptation'),('Residual VCA SD','sd_vca'),('Gradient SD','sd_gradient')]
# normalize to baseline SDs for comparable scale
base=np.array([2.6,1.4,.14,1.5])
for i,(lab,col) in enumerate(series): ax.plot(x,r[col]/base[i],marker='o',lw=2.2,label=lab)
# Keep acquisition-view labels readable and separated from the title.
for _, row in r.iterrows():
    ax.annotate(
        f'{int(row["step"])}. {row["view"]}',
        xy=(row.step, 1.0),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=8,
        rotation=0
    )
ax.set_xticks(x)
ax.set_ylim(0,1.28)
ax.set_xlabel('Sequential acquisition step')
ax.set_ylabel('Posterior SD / initial SD')
ax.set_title('Reference Information-Guided Acquisition Path', weight='bold', pad=18)
ax.grid(alpha=.18)
ax.legend(frameon=False,ncol=2,loc='lower left')
save('Figure_2_Reference_Sequence.png')

# 3 phase utility heatmap
u=pd.read_csv(BASE/'View_Utility_by_Phase.csv'); piv=u.pivot(index='view',columns='phase',values='utility').reindex(index=['Intercommissural','LVOT long-axis','Biplane X-plane','3D en-face','3D MPR','3D color VCA','CW Doppler'],columns=['Targeting','Alignment','Grasping','Result assessment'])
fig,ax=plt.subplots(figsize=(10.5,7)); im=ax.imshow(piv.values,aspect='auto',cmap='viridis'); ax.set_xticks(range(len(piv.columns)),piv.columns); ax.set_yticks(range(len(piv.index)),piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]): ax.text(j,i,f'{piv.iloc[i,j]:.2f}',ha='center',va='center',color='white' if piv.iloc[i,j]>.45*piv.values.max() else 'black',fontsize=8.5)
fig.colorbar(im,ax=ax,label='Model utility'); ax.set_title('Phase-Specific Expected Information Utility by Echocardiographic View',weight='bold'); save('Figure_3_View_Utility_Heatmap.png')

# 4 model comparison
mc=pd.read_csv(BASE/'Synthetic_Model_Comparison_AIC.csv').sort_values('AIC')
fig,ax=plt.subplots(figsize=(11.2,6.5))
bars=ax.barh(mc.model,mc.AIC)
ax.invert_yaxis()
max_aic = mc.AIC.max()
ax.set_xlim(0, max_aic * 1.22)
ax.set_xlabel('AIC on model-generated action labels (lower is better)')
ax.set_title('Internal Model-Recovery Comparison',weight='bold',pad=14)

# Place every AIC/accuracy annotation inside the plotting area.
for b,v,acc in zip(bars,mc.AIC,mc.internal_action_accuracy):
    ax.text(
        v + max_aic * .012,
        b.get_y() + b.get_height()/2,
        f'AIC {v:.0f} | acc {acc:.2f}',
        va='center',
        ha='left',
        fontsize=8.5,
        clip_on=True
    )
ax.grid(axis='x',alpha=.18)
save('Figure_4_Model_Comparison_AIC.png')

# 5 PRCC tornado for terminal entropy
pr=pd.read_csv(BASE/'Global_Sensitivity_PRCC.csv'); pr=pr[pr.output=='terminal_entropy'].copy(); pr['abs']=pr.PRCC.abs(); pr=pr.sort_values('abs').tail(10)
fig,ax=plt.subplots(figsize=(10.7,6.8)); ax.barh(pr.parameter,pr.PRCC); ax.axvline(0,lw=1,color='black'); ax.set_xlabel('PRCC with terminal posterior entropy'); ax.set_title('Global Sensitivity of Residual Imaging Uncertainty',weight='bold'); ax.grid(axis='x',alpha=.18); save('Figure_5_Global_Sensitivity_PRCC.png')

# 6 sensitivity surface
s=pd.read_csv(BASE/'Sensitivity_Surface.csv'); order=['Intercommissural','LVOT long-axis','Biplane X-plane','3D en-face','3D MPR','3D color VCA','CW Doppler']; code={v:i for i,v in enumerate(order)}; s['code']=s.selected_view.map(code)
X=s.pivot(index='sd_vca',columns='sd_gap',values='code'); U=s.pivot(index='sd_vca',columns='sd_gap',values='max_utility')
fig,ax=plt.subplots(figsize=(11,7.5)); im=ax.imshow(X.values,origin='lower',aspect='auto',extent=[X.columns.min(),X.columns.max(),X.index.min(),X.index.max()],cmap='tab20',vmin=0,vmax=len(order)-1)
cs=ax.contour(U.columns,U.index,U.values,levels=7,colors='black',linewidths=.6,alpha=.65); ax.clabel(cs,fontsize=7,fmt='%.2f'); ax.set_xlabel('Prior SD of coaptation gap (mm)'); ax.set_ylabel('Prior SD of residual 3D-VCA (cm²)'); ax.set_title('Sensitivity Surface for the First Information-Guided View',weight='bold')
handles=[plt.Line2D([0],[0],marker='s',linestyle='',label=v,markerfacecolor=plt.cm.tab20(code[v]/max(1,len(order)-1)),markersize=9) for v in order]; ax.legend(handles=handles,bbox_to_anchor=(1.03,1),loc='upper left',frameon=False,fontsize=8); save('Figure_6_View_Selection_Surface.png')

# 7 covariance cycle stability
cy=pd.read_csv(BASE/'Covariance_Cycle_Stability.csv')
fig,ax=plt.subplots(figsize=(10.5,6.2)); bars=ax.bar(cy.cycle,cy.spectral_radius); ax.axhline(1,ls='--',lw=1.2,color='black'); ax.set_ylabel('Spectral radius of linearized covariance map'); ax.set_title('Stability of Repeated Complementary Imaging Cycles',weight='bold'); ax.tick_params(axis='x',rotation=12)
for b,v in zip(bars,cy.spectral_radius): ax.text(b.get_x()+b.get_width()/2,v+0.015,f'{v:.3f}',ha='center'); save('Figure_7_Covariance_Stability.png')

# Table figures
anchors=pd.read_csv(BASE/'Model_Parameters.csv'); lit=anchors[anchors.status=='literature-derived'][['name','value','unit','role']].head(12).copy(); lit['value']=lit.value.map(lambda x:f'{x:g}')
table_figure(lit,'Table 1. Selected Literature-Derived Clinical Anchors','Table_1_Literature_Anchors.png',fontsize=7.0,figwidth=15,rowheight=.40)
par=anchors[anchors.status!='literature-derived'][['name','value','unit','status']].head(14).copy(); par['value']=par.value.map(lambda x:f'{x:.3g}')
table_figure(par,'Table 2. Provisional Structural Parameters and Decision Guardrails','Table_2_Model_Parameters.png',fontsize=7.0,figwidth=15,rowheight=.40)
mc2=mc[['model','k','AIC','BIC','internal_action_accuracy']].copy(); mc2[['AIC','BIC']]=mc2[['AIC','BIC']].round(1); mc2.internal_action_accuracy=mc2.internal_action_accuracy.round(3)
table_figure(mc2,'Table 3. Internal AIC/BIC Model-Recovery Experiment','Table_3_Model_Comparison.png',fontsize=7.5,figwidth=14,rowheight=.44)
ps=pd.read_csv(BASE/'Policy_Summary.csv')[['policy','median_views','terminal_entropy','weighted_uncertainty','VCA_MAE','gradient_MAE','completion_rate']].copy(); ps=ps.round(3)
table_figure(ps,'Table 4. Synthetic Policy Stress-Test Summary','Table_4_Policy_Summary.png',fontsize=7.5,figwidth=14,rowheight=.44)
sc=pd.read_csv(BASE/'Scenario_Results.csv').copy(); sc[['terminal_entropy','weighted_uncertainty']]=sc[['terminal_entropy','weighted_uncertainty']].round(3)
table_figure(sc,'Table 5. Prespecified Structural Sensitivity Scenarios','Table_5_Scenario_Results.png',fontsize=7.3,figwidth=15,rowheight=.43)

print('figures written',len(list(FIG.glob('*.png'))))
