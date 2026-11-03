from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'manuscript/figures'
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue,orange='#1769aa','#d97721'
def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight'); plt.close(fig)

d=pd.read_csv(ROOT/'manuscript/figure_data/libero_goal_scene_crossover.csv')
fig,axes=plt.subplots(2,1,figsize=(3.45,3.5),sharex=True)
for ax,task in zip(axes,[6,9]):
    for y,goal in [(1,'cheese_bowl'),(0,'wine_rack')]:
        cell=d[(d.scene_task==task)&(d.requested_goal==goal)]; assert len(cell)==3
        for seed,marker,shift,color in [(3,'o',.2,blue),(4,'s',0,orange),(5,'^',-.2,'#316b51')]:
            v=float(cell.loc[cell.training_seed.eq(seed),'gr00t_minus_qwen_pp'].iloc[0]); ax.scatter(v,y+shift,marker=marker,s=25,color=color,label=f'Seed {seed}' if y==1 else None); ax.annotate(f'{v:+g}',(v,y+shift),xytext=(4,0),textcoords='offset points',va='center',fontsize=7)
    ax.set_yticks([1,0],['Cheese: original','Wine: swapped'] if task==6 else ['Cheese: swapped','Wine: original']); ax.set_title(f'Task {task} scene'); ax.axvline(0,color='#555',lw=.8); ax.set_xlim(-25,33); ax.set_xticks([-20,-10,0,10,20,30]); ax.grid(axis='x',alpha=.2); ax.set_xlabel('GR00T − Qwen (percentage points)')
axes[0].legend(frameon=False,loc='lower right',fontsize=8,ncol=1); fig.tight_layout(h_pad=1.2); save(fig,'libero_goal_comparison_review.pdf')

d=pd.read_csv(ROOT/'docs/analysis/stage5/checkpoint_anchor_alignment.csv')
fig,axes=plt.subplots(3,2,figsize=(3.45,4.25),sharex='col')
for col,(dataset,title) in enumerate([('aloha','ALOHA'),('libero','LIBERO')]):
    for row,(field,lab) in enumerate([('train_loss','Training loss'),('validation_loss','Validation loss'),('development_success','Success (%)')]):
        ax=axes[row,col]
        for model,color,marker,ls in [('GR00T',blue,'o','-'),('Qwen',orange,'s','--')]:
            sub=d[(d.dataset==dataset)&(d.backbone==model)]; k=100 if row==2 else 1
            for _,s in sub.groupby('training_seed'): ax.plot(s.epoch,s[field]*k,color=color,alpha=.25,lw=.7)
            means=sub.groupby('epoch')[field].mean()*k; ax.plot(means.index,means,color=color,marker=marker,ms=2.5,lw=1.3,ls=ls,label=model)
        ax.set_title(title if row==0 else ''); ax.grid(alpha=.18)
        if row<2: ax.set_yscale('log'); ax.set_ylabel(lab if col==0 else '')
        else: ax.set_ylim(0,100); ax.set_ylabel('Success (%)' if col==0 else ''); ax.set_xlabel('Epoch')
        if row==1: ax.axvline(25,color='#666',ls=':',lw=1)
axes[0,0].legend(frameon=False,fontsize=7); fig.tight_layout(h_pad=.6,w_pad=.5); save(fig,'training_progression_review.pdf')
