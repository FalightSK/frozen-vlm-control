import csv, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import median
rows=list(csv.DictReader(open('docs/analysis/stage2/aloha_reward_event_rates.csv',newline='')))
timing=list(csv.DictReader(open('docs/analysis/stage2/aloha_matched_score2_timing.csv',newline='')))
fig,(a,b)=plt.subplots(1,2,figsize=(7.1,2.6),gridspec_kw={'width_ratios':[1.35,1]})
for i,seed in enumerate((3,4,5)):
 for j,(model,col) in enumerate((('GR00T','#2873ae'),('Qwen','#df8a2d'))):
  r=next(x for x in rows if int(x['training_seed'])==seed and x['backbone']==model)
  for k,key in enumerate(('score2_episodes','score2_then_score4')):
   x=i*3.2+k*.95+j*.35
   a.bar(x,int(r[key]),width=.34,color=col,label=model if i==0 and k==0 else None)
   a.text(x,int(r[key])+4,r[key],ha='center',va='bottom',rotation=90,fontsize=7)
a.set_xticks([.17,1.12,3.37,4.32,6.57,7.52],['2','4','2','4','2','4']);a.set_ylim(0,260);a.set_ylabel('Episodes / 200',fontsize=8);a.tick_params(labelsize=7);a.legend(frameon=False,fontsize=7,ncol=2,loc='upper right');a.set_title('A  Task progress',loc='left',fontsize=9,fontweight='bold')
for i,seed in enumerate((3,4,5)):
 a.text(i*3.2+.65,-.24,f'Seed {seed}',transform=a.get_xaxis_transform(),ha='center',fontsize=8)
rng=random.Random(44)
for i,seed in enumerate((3,4,5)):
 vals=[int(r['qwen_minus_groot_steps']) for r in timing if int(r['seed'])==seed and r['outcome']=='groot_only']
 b.scatter(vals,[i+rng.uniform(-.19,.19) for _ in vals],s=10,color='#507da2',alpha=.5,edgecolors='none');m=median(vals);b.scatter([m],[i],marker='D',s=30,color='#d47b16',zorder=3);b.text(m+7,i+.18,f'{m:g}',fontsize=8)
b.axvline(0,color='#444',lw=.7);b.set_yticks([0,1,2],['3 (n=35)','4 (n=39)','5 (n=39)']);b.invert_yaxis();b.set_xlabel('Qwen minus GR00T score-2 step',fontsize=8);b.tick_params(labelsize=7);b.set_title('B  Timing on selected starts',loc='left',fontsize=9,fontweight='bold')
for ax in (a,b): ax.spines[['top','right']].set_visible(False)
fig.subplots_adjust(left=.07,right=.99,bottom=.26,top=.87,wspace=.4)
fig.savefig('manuscript/figures/aloha_progress_timing.pdf')
