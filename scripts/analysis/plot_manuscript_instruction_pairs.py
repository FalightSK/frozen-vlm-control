import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
rows=list(csv.DictReader(open('docs/analysis/stage3/stage3_instruction_sensitivity.csv',newline='')))
fig,ax=plt.subplots(figsize=(3.42,3.7))
colors=['#6d5b9c','#397ba3','#d99138','#dedede']
keys=['both','canonical_only','swapped_only','neither']
for i,(task,seed,model) in enumerate((t,s,m) for t in (6,9) for s in (3,4,5) for m in ('groot','qwen3vl')):
 r=next(r for r in rows if int(r['task'])==task and int(r['training_seed'])==seed and r['backbone']==model)
 y=11-i + (0.5 if task==6 else -0.5); left=0
 for key,c in zip(keys,colors):
  v=int(r[key]); ax.barh(y,v,left=left,height=.72,color=c,edgecolor='white',linewidth=.45)
  if v>=3: ax.text(left+v/2,y,str(v),ha='center',va='center',fontsize=7,color='white' if key in keys[:2] else 'black')
  left+=v
ax.set_yticks([11-i+(0.5 if i<6 else -0.5) for i in range(12)], [f"{'G' if i%2==0 else 'Q'} · {3+(i%6)//2}" for i in range(12)],fontsize=7.5)
ax.text(-8,12.5,'Task 6',fontsize=8,fontweight='bold'); ax.text(-8,5.5,'Task 9',fontsize=8,fontweight='bold')
ax.set_xlim(0,40);ax.set_xlabel('Matched starts (out of 40)',fontsize=8);ax.tick_params(axis='x',labelsize=7);ax.tick_params(axis='y',length=0);ax.spines[['top','right','left']].set_visible(False)
ax.legend([Patch(facecolor=c) for c in colors],['Both','Original only','Swapped only','Neither'],loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,fontsize=7,frameon=False)
fig.subplots_adjust(left=.16,right=.98,top=.94,bottom=.22)
fig.savefig('manuscript/figures/libero_instruction_compact.pdf')
