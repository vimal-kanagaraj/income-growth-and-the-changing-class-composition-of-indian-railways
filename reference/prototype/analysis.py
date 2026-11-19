import sys; sys.path.insert(0,'/home/claude/rail/a3')
import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings, json; warnings.filterwarnings('ignore')
from data import *
OUT='/home/claude/rail/a3/out/'; import os; os.makedirs(OUT,exist_ok=True)
AC=CLASSES[:5]; ORD=['First Class (Ordinary)','Sleeper (Ordinary)','Second Class (Ordinary)']
rows=[dict(year=y,clas=c,pax_mn=PAX[y][i],pkm_mn=PKM[y][i],earn_cr=EARN[y][i]) for y in YEARS for i,c in enumerate(CLASSES)]
df=pd.DataFrame(rows)
df['segment']=np.where(df.clas.isin(AC),'AC',np.where(df.clas=='Suburban','Suburban','NonAC'))
df['t']=df.year.map({y:i for i,y in enumerate(YEARS)})
df['Y']=df.year.map(NNI_CON); df['defl']=100*df.year.map(NNI_CUR)/df.Y
df['pop_mn']=df.year.map(lambda y:NNI_TOT[y]*1e7/NNI_CON[y]/1e6)
df['pkm_pc']=df.pkm_mn/df.pop_mn
df['yield_p']=np.where(df.pkm_mn>0,df.earn_cr*1000/df.pkm_mn,np.nan); df['ryield']=df.yield_p*100/df.defl
df['D_SUSP']=(df.year=='2020-21')*1; df['D_RESTR']=(df.year=='2021-22')*1; df['POST']=df.year.isin(['2022-23','2023-24','2024-25'])*1
df['D_FARE']=np.where(df.clas.isin(ORD),df.year.map({'2021-22':1.0,'2022-23':1.0,'2023-24':0.9}).fillna(0.0),0.0)
df['lnq']=np.log(df.pkm_pc.where(df.pkm_mn>0)); df['lnY']=np.log(df.Y); df['lnP']=np.log(df.ryield.where(df.ryield>0))
df.to_csv(OUT+'panel.csv',index=False)
R={}
pop=df.groupby('year').pop_mn.first(); R['pop']=pop.round(1).to_dict()
# ---------- descriptives
seg=df.groupby(['year','segment']).agg(pax=('pax_mn','sum'),pkm=('pkm_mn','sum'),earn=('earn_cr','sum')).reset_index()
seg.loc[seg.year=='2015-16','pax']=np.nan
for m in ['pax','pkm','earn']: seg['sh_'+m]=seg.groupby('year')[m].transform(lambda s:100*s/s.sum())
seg['yield_p']=seg.earn*1000/seg.pkm
seg['ryield']=seg.yield_p*100/seg.year.map(lambda y:100*NNI_CUR[y]/NNI_CON[y])
seg['pkm_pc']=seg.pkm/seg.year.map(pop)
seg.to_csv(OUT+'segment_year.csv',index=False)
print(seg.pivot(index='year',columns='segment',values='sh_pkm').round(2))
print(seg.pivot(index='year',columns='segment',values='sh_earn').round(2))
print(seg.pivot(index='year',columns='segment',values='ryield').round(1))
print(seg.pivot(index='year',columns='segment',values='pkm_pc').round(1))
# ---------- decomposition of AC share change (Shapley)
def s(A,N): return 100*A/(A+N)
def decomp(y0,y1):
    g=seg.groupby(['year','segment']).pkm.sum().unstack()
    A0,A1=g.loc[y0,'AC'],g.loc[y1,'AC']; N0,N1=g.loc[y0,['NonAC','Suburban']].sum(),g.loc[y1,['NonAC','Suburban']].sum()
    tot=s(A1,N1)-s(A0,N0)
    a=0.5*((s(A1,N0)-s(A0,N0))+(s(A1,N1)-s(A0,N1))); n=tot-a
    return dict(window=f'{y0}->{y1}',total=tot,AC_growth=a,nonAC_change=n,A_growth_pct=100*(A1/A0-1),N_growth_pct=100*(N1/N0-1))
D=[decomp('2015-16','2019-20'),decomp('2019-20','2024-25'),decomp('2018-19','2024-25')]
print(pd.DataFrame(D).round(2)); R['decomp']=D
# ---------- M2 pre-pandemic per-capita elasticities, pooled with class FE
pre=df[(df.t<=4)]
def m2(d,form):
    return smf.ols(form,d).fit()
M2={}
for sname in ['AC','NonAC','Suburban']:
    d=pre[(pre.segment==sname)&pre.lnq.notna()]
    if sname=='NonAC': d=d[~d.clas.isin(['First Class (M/E)','First Class (Ordinary)'])]
    fe=' + C(clas)' if d.clas.nunique()>1 else ''
    r=m2(d,'lnq ~ lnY'+fe); rp=m2(d,'lnq ~ lnY + lnP'+fe)
    loo=[m2(d[d.year!=y],'lnq ~ lnY'+fe).params.lnY for y in YEARS[:5]]
    clean=m2(d[d.year!='2019-20'],'lnq ~ lnY'+fe).params.lnY
    fixed={th:m2(d.assign(q2=d.lnq-th*d.lnP),'q2 ~ lnY'+fe).params.lnY for th in [0,-0.5,-1.0]}
    M2[sname]=dict(n=int(r.nobs),beta=r.params.lnY,se=r.bse.lnY,r2=r.rsquared,beta_withP=rp.params.lnY,theta=rp.params.lnP,theta_se=rp.bse.lnP,
                  loo_min=min(loo),loo_max=max(loo),beta_FY16_19=clean,fixed_theta=fixed)
print(pd.DataFrame(M2).T.round(2))
# segment-aggregate per-capita fits (used for benchmark)
sg=seg.copy(); sg['lnq']=np.log(sg.pkm_pc); sg['lnY']=np.log(sg.year.map(NNI_CON))
B={}
for sname in ['AC','NonAC','Suburban']:
    d=sg[(sg.segment==sname)&(sg.year.isin(YEARS[:5]))]; r=smf.ols('lnq ~ lnY',d).fit()
    loo=[smf.ols('lnq ~ lnY',d[d.year!=y]).fit().params.lnY for y in YEARS[:5]]
    B[sname]=dict(a=r.params.Intercept,b=r.params.lnY,se=r.bse.lnY,r2=r.rsquared,loo_min=min(loo),loo_max=max(loo),resid=r.resid.values)
    print(f'segment agg pc {sname}: b={r.params.lnY:.2f} se={r.bse.lnY:.2f} R2={r.rsquared:.2f} loo[{min(loo):.2f},{max(loo):.2f}]')
# class-by-class exploratory, per capita, FY16-FY20
cl=[]
for c in CLASSES:
    d=pre[(pre.clas==c)&pre.lnq.notna()]
    r=smf.ols('lnq ~ lnY',d).fit(); loo=[smf.ols('lnq ~ lnY',d[d.year!=y]).fit().params.lnY for y in YEARS[:5]]
    cl.append(dict(clas=c,beta=r.params.lnY,se=r.bse.lnY,loo_min=min(loo),loo_max=max(loo),share_pkm_FY20=100*PKM['2019-20'][CLASSES.index(c)]/sum(PKM['2019-20'])))
cl=pd.DataFrame(cl); print(cl.round(2)); cl.to_csv(OUT+'class_elasticities.csv',index=False)
# combined second class and sleeper
for g,cs in [('Second combined',['Second Class (M/E)','Second Class (Ordinary)']),('Sleeper combined',['Sleeper (M/E)','Sleeper (Ordinary)'])]:
    d=df[df.clas.isin(cs)].groupby('year').agg(pkm=('pkm_mn','sum'),pop=('pop_mn','first')).reset_index()
    d['lnq']=np.log(d.pkm/d['pop']); d['lnY']=np.log(d.year.map(NNI_CON))
    r=smf.ols('lnq ~ lnY',d[d.year.isin(YEARS[:5])]).fit()
    pred=np.exp(r.params.Intercept+r.params.lnY*d.lnY)*d['pop']
    d['pred']=pred; d['gap_pct']=100*(d.pkm/pred-1)
    print(g,'beta',round(r.params.lnY,2),d[['year','pkm','pred','gap_pct']].round(1).to_string(index=False))
    R[g]=dict(beta=r.params.lnY,gap=d.set_index('year').gap_pct.round(1).to_dict())
# ---------- M3 benchmark: predicted pkm and shares
yr=pd.DataFrame({'year':YEARS}); yr['lnY']=np.log(yr.year.map(NNI_CON)); yr['pop']=yr.year.map(pop)
for sname in B: yr['pred_'+sname]=np.exp(B[sname]['a']+B[sname]['b']*yr.lnY)*yr['pop']
g=seg.pivot(index='year',columns='segment',values='pkm').reset_index()
yr=yr.merge(g,on='year')
yr['sh_act']=100*yr.AC/(yr.AC+yr.NonAC+yr.Suburban); yr['sh_pred']=100*yr.pred_AC/(yr.pred_AC+yr.pred_NonAC+yr.pred_Suburban)
yr['gap_pp']=yr.sh_act-yr.sh_pred
for sname in B: yr['gap_pct_'+sname]=100*(yr[sname]/yr['pred_'+sname]-1)
k,c=np.polyfit(np.arange(5),yr.sh_act[:5],1); yr['sh_trend']=c+k*np.arange(10)
band=np.abs(yr.gap_pp[:5]).max(); R['band_pp']=band
print(yr[['year','sh_act','sh_pred','sh_trend','gap_pp','gap_pct_AC','gap_pct_NonAC','gap_pct_Suburban']].round(2).to_string(index=False))
print('baseline max |share error| pp',round(band,2))
yr.to_csv(OUT+'benchmark.csv',index=False)
# rupee value of the FY25 gap
y25=seg[seg.year=='2024-25'].set_index('segment')
extraAC=yr.AC.iloc[-1]-yr.pred_AC.iloc[-1]; shortN=yr.NonAC.iloc[-1]-yr.pred_NonAC.iloc[-1]; shortS=yr.Suburban.iloc[-1]-yr.pred_Suburban.iloc[-1]
valAC=extraAC*y25.loc['AC','yield_p']/1000; valN=shortN*y25.loc['NonAC','yield_p']/1000; valS=shortS*y25.loc['Suburban','yield_p']/1000
R['rupee']=dict(extraAC_mnpkm=extraAC,shortNonAC_mnpkm=shortN,shortSub_mnpkm=shortS,valAC_cr=valAC,valNonAC_cr=valN,valSub_cr=valS,yieldAC=y25.loc['AC','yield_p'],yieldN=y25.loc['NonAC','yield_p'],yieldS=y25.loc['Suburban','yield_p'],totEarnFY25=seg[seg.year=='2024-25'].earn.sum())
print({k:round(v,1) for k,v in R['rupee'].items()})
# share of AC rise explained by income
R['income_explained']=dict(pred_rise=yr.sh_pred.iloc[-1]-yr.sh_pred.iloc[4],act_rise=yr.sh_act.iloc[-1]-yr.sh_act.iloc[4])
print(R['income_explained'])
# ---------- M4 regime model, per capita, segment aggregates FY16-FY25
M4={}
for sname in ['AC','NonAC','Suburban']:
    d=sg[sg.segment==sname].copy(); d['D_SUSP']=(d.year=='2020-21')*1; d['D_RESTR']=(d.year=='2021-22')*1; d['POST']=d.year.isin(YEARS[7:])*1
    r=smf.ols('lnq ~ lnY + D_SUSP + D_RESTR + POST',d).fit()
    M4[sname]=dict(b=r.params.lnY,b_se=r.bse.lnY,g3=r.params.POST,g3_se=r.bse.POST,g3_pct=100*(np.exp(r.params.POST)-1),g1=r.params.D_SUSP,g2=r.params.D_RESTR,r2=r.rsquared)
print(pd.DataFrame(M4).T.round(3))
# class-level panel non-AC with D_FARE
d=df[(df.segment=='NonAC')&df.lnq.notna()&~df.clas.isin(['First Class (M/E)','First Class (Ordinary)'])]
r=smf.ols('lnq ~ lnY + D_SUSP + D_RESTR + POST + D_FARE + C(clas)',d).fit()
M4['NonAC_classpanel']=dict(b=r.params.lnY,g3=r.params.POST,dfare=r.params.D_FARE,dfare_se=r.bse.D_FARE,n=int(r.nobs))
print('class panel nonAC',{k:round(v,3) for k,v in M4['NonAC_classpanel'].items()})
# price: relative real yields
p=df.pivot(index='year',columns='clas',values='ryield')
R['rel_yield']={y:round(p.loc[y,'AC 3 Tier']/p.loc[y,'Sleeper (M/E)'],3) for y in YEARS}
R['ryield_change_FY20_25']=((p.loc['2024-25']/p.loc['2019-20']-1)*100).round(1).to_dict()
segry=seg.pivot(index='year',columns='segment',values='ryield'); R['seg_ryield']=segry.round(1).to_dict()
print(R['rel_yield'])
json.dump(dict(R=R,M2=M2,M4=M4,B={k:{kk:(vv.tolist() if hasattr(vv,'tolist') else vv) for kk,vv in v.items()} for k,v in B.items()}),open(OUT+'results.json','w'),indent=1,default=float)
