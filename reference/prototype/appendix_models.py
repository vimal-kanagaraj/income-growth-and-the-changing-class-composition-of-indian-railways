import pandas as pd, numpy as np, statsmodels.formula.api as smf, json, warnings; warnings.filterwarnings('ignore')
OUT='/home/claude/rail/a3/out/'
df=pd.read_csv(OUT+'panel.csv'); seg=pd.read_csv(OUT+'segment_year.csv')
YEARS=sorted(df.year.unique())
def tab(r,keep):
    return [dict(term=k,coef=r.params[k],se=r.bse[k],t=r.tvalues[k]) for k in keep]+[dict(term='n',coef=int(r.nobs)),dict(term='R2',coef=r.rsquared)]
out={}
pre=df[df.t<=4]
for s in ['AC','NonAC']:
    d=pre[(pre.segment==s)&pre.lnq.notna()]
    if s=='NonAC': d=d[~d.clas.isin(['First Class (M/E)','First Class (Ordinary)'])]
    out[f'M2 pooled {s}']=tab(smf.ols('lnq ~ lnY + C(clas)',d).fit(),['lnY'])
    out[f'M2 pooled {s} with price']=tab(smf.ols('lnq ~ lnY + lnP + C(clas)',d).fit(),['lnY','lnP'])
seg['lnq']=np.log(seg.pkm_pc); Y=df.groupby('year').Y.first(); seg['lnY']=np.log(seg.year.map(Y))
for s in ['AC','NonAC','Suburban']:
    d=seg[(seg.segment==s)&seg.year.isin(YEARS[:5])]
    out[f'M3 segment fit {s}']=tab(smf.ols('lnq ~ lnY',d).fit(),['Intercept','lnY'])
    d=seg[seg.segment==s].copy(); d['D_SUSP']=(d.year=='2020-21')*1; d['D_RESTR']=(d.year=='2021-22')*1; d['POST']=d.year.isin(YEARS[7:])*1
    out[f'M4 regime {s}']=tab(smf.ols('lnq ~ lnY + D_SUSP + D_RESTR + POST',d).fit(),['lnY','D_SUSP','D_RESTR','POST'])
d=df[(df.segment=='NonAC')&df.lnq.notna()&~df.clas.isin(['First Class (M/E)','First Class (Ordinary)'])]
out['M4 class panel NonAC']=tab(smf.ols('lnq ~ lnY + D_SUSP + D_RESTR + POST + D_FARE + C(clas)',d).fit(),['lnY','D_SUSP','D_RESTR','POST','D_FARE'])
json.dump(out,open(OUT+'models.json','w'),indent=1,default=float)
for k,v in out.items(): print(k,[(x['term'],round(x['coef'],3),round(x.get('se',np.nan),3)) for x in v])
