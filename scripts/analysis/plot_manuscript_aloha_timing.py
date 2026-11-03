import csv, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import median
rows=list(csv.DictReader(open('docs/analysis/stage2/aloha_matched_score2_timing.csv',newline='')))
rng=random.Random(44)
fig,ax=plt.subplots(figsize=(3.45,1.85))
for i,seed in enumerate((3,4,5)):
 vals=[int(r['qwen_minus_groot_steps']) for r in rows if int(r['seed'])==seed and r['outcome']=='groot_only']
 ax.scatter(vals,[i+rng.uniform(-.19,.19) for _ in vals],s=10,color='#507da2',alpha=.5,edgecolors='none')
 m=median(vals);ax.scatter([m],[i],marker='D',s=37,color='#d47b16',zorder=3);ax.text(m+8,i+.18,f'{m:g}',fontsize=7)
ax.axvline(0,color='#444',lw=.7);ax.set_yticks([0,1,2],['3 (n=35)','4 (n=39)','5 (n=39)']);ax.invert_yaxis();ax.set_xlabel('Qwen minus GR00T score-2 step',fontsize=8);ax.set_ylabel('Training seed',fontsize=8);ax.tick_params(labelsize=7);ax.spines[['top','right']].set_visible(False);fig.tight_layout(pad=.4);fig.savefig('manuscript/figures/aloha_score2_delay.pdf')
