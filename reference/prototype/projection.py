import pandas as pd, json
seg=pd.read_csv('out/segment_year.csv'); bm=pd.read_csv('out/benchmark.csv').set_index('year')
g=seg[seg.year=='2024-25'].set_index('segment').pkm/1000
gp={s:bm.loc['2024-25','pred_'+s]/1000 for s in ['AC','NonAC','Suburban']}
sets={'Baseline (0.94, −0.68, −0.38)':(0.94,-0.68,-0.38),'Long difference (1.50, −0.29, −0.29)':(1.50,-0.29,-0.29)}
rows=[]
for start,base in [('Actual 2024-25 (excess persists)',g),('Income path 2024-25 (excess unwinds)',gp)]:
    for k,(a,n,s) in sets.items():
        f=lambda b:((1.05**b)*1.01)**5
        A,N,Su=base['AC']*f(a),base['NonAC']*f(n),base['Suburban']*f(s)
        rows.append(dict(start=start,elast=k,AC_bn=A,NonAC_bn=N,Sub_bn=Su,AC_share=100*A/(A+N+Su),AC_share_start=100*base['AC']/(base['AC']+base['NonAC']+base['Suburban'])))
d=pd.DataFrame(rows); print(d.round(1).to_string(index=False)); d.to_csv('out/projection_2030.csv',index=False)
# revenue scenarios
r=json.load(open('out/results.json'))['R']['rupee']; print('half unwind cr',round(r['valAC_cr']/2),'pct',round(100*r['valAC_cr']/2/r['totEarnFY25'],1))
