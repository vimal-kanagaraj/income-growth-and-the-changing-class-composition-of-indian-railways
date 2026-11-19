import sys; sys.path.insert(0,'/home/claude/rail/a3')
import pandas as pd, numpy as np, json
from data import *
OUT='/home/claude/rail/a3/out/'
res=json.load(open(OUT+'results.json'))
seg=pd.read_csv(OUT+'segment_year.csv'); pop={y:NNI_TOT[y]*1e7/NNI_CON[y]/1e6 for y in YEARS}
pop['2010-11']=7373384e7/62170/1e6; Y={**NNI_CON,'2010-11':62170}
g=seg.pivot(index='year',columns='segment',values='pkm')
# Year Book 2018-19 Table II, 2010-11 (mn pkm): upper class = AC + First Class
yb={'Upper':62203,'SecME':500631,'SecOrd':278547,'Sub':137127}
fc20=PKM['2019-20'][5]+PKM['2019-20'][6]
up20=g.loc['2019-20','AC']+fc20; nonac20_exFC=g.loc['2019-20','NonAC']-fc20
ld={}
for k,a10,a20 in [('AC (upper class)',yb['Upper'],up20),('NonAC (second class)',yb['SecME']+yb['SecOrd'],nonac20_exFC),('Suburban',yb['Sub'],g.loc['2019-20','Suburban'])]:
    q10=a10/pop['2010-11']; q20=a20/pop['2019-20']
    ld[k]=dict(pc10=q10,pc20=q20,elast=np.log(q20/q10)/np.log(Y['2019-20']/Y['2010-11']))
print('long difference 2010-11 -> 2019-20',{k:round(v['elast'],2) for k,v in ld.items()})
print('AC upper share 2010-11',round(100*yb['Upper']/978508,2),'pop10',round(pop['2010-11'],1))
# Anchored-at-FY20 projections of FY25 shares under different elasticity sets
def proj(bAC,bN,bS):
    f=lambda s,b: g.loc['2019-20',s]*(Y['2024-25']/Y['2019-20'])**b*pop['2024-25']/pop['2019-20']
    a,n,s=f('AC',bAC),f('NonAC',bN),f('Suburban',bS); return 100*a/(a+n+s), a
base=res['B']
sets={'Baseline segment fit (FY16-FY20)':(base['AC']['b'],base['NonAC']['b'],base['Suburban']['b']),
      'Leave-one-out bounds most favourable to income':(base['AC']['loo_max'],base['NonAC']['loo_min'],base['Suburban']['loo_min']),
      'Long difference 2010-11 to 2019-20':(ld['AC (upper class)']['elast'],ld['NonAC (second class)']['elast'],ld['Suburban']['elast']),
      'AC elasticity 2.0 (Dargay-Gately level)':(2.0,base['NonAC']['b'],base['Suburban']['b'])}
sh20=100*g.loc['2019-20','AC']/g.loc['2019-20'].sum(); sh25=100*g.loc['2024-25','AC']/g.loc['2024-25'].sum()
rows=[]
for k,(a,n,s) in sets.items():
    p,acp=proj(a,n,s); rows.append(dict(set=k,bAC=a,bNonAC=n,bSub=s,pred_share25=p,income_rise_pp=p-sh20,actual_rise_pp=sh25-sh20,share_explained=100*(p-sh20)/(sh25-sh20),AC_gap_pct=100*(g.loc['2024-25','AC']/acp-1)))
proj_df=pd.DataFrame(rows); print(proj_df.round(2).to_string(index=False)); proj_df.to_csv(OUT+'projection_sets.csv',index=False)
# income-attributable vs excess AC pkm (fitted benchmark)
bm=pd.read_csv(OUT+'benchmark.csv').set_index('year')
addAC=g.loc['2024-25','AC']-g.loc['2019-20','AC']; inc=bm.loc['2024-25','pred_AC']-bm.loc['2019-20','pred_AC']
print('AC added',round(addAC),'income-path added',round(inc),'share',round(100*inc/addAC,1))
# AC 3 tier share of added AC pkm, sleeper ME
print('AC3 share of added', round(100*(PKM['2024-25'][2]-PKM['2019-20'][2])/addAC,1))
json.dump(dict(long_diff=ld,proj=rows,ac_added=addAC,ac_income_added=inc,sh20=sh20,sh25=sh25),open(OUT+'results2.json','w'),indent=1,default=float)
