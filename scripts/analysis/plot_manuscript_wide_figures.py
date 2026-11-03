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

d=pd.read_csv(ROOT/'docs/analysis/stage2/aloha_reward_event_rates.csv')
fig,axes=plt.subplots(1,3,figsize=(7.16,2.05),sharey=True)
for ax,seed in zip(axes,[3,4,5]):
    for off,model,color,hatch in [(-.19,'GR00T',blue,''),(.19,'Qwen',orange,'//')]:
        r=d[(d.training_seed==seed)&(d.backbone==model)].iloc[0]; vals=[r.score2_episodes,r.score2_then_score4]
        bars=ax.bar(np.arange(2)+off,vals,.36,color=color,hatch=hatch,label=model)
        ax.bar_label(bars,labels=[str(int(x)) for x in vals],padding=2,fontsize=7)
    ax.set_title(f'Seed {seed}'); ax.set_xticks([0,1],['Score 2','Score 4']); ax.set_ylim(0,225); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
axes[0].set_ylabel('Episodes / 200')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,fontsize=7,bbox_to_anchor=(.5,1.03))
fig.tight_layout(rect=(0,0,1,.91),w_pad=.8); save(fig,'aloha_progress_review.pdf')

d=pd.read_csv(ROOT/'manuscript/figure_data/libero_goal_scene_crossover.csv')
fig,axes=plt.subplots(1,2,figsize=(7.16,2.25),sharex=True)
for ax,task in zip(axes,[6,9]):
    for y,goal in [(1,'cheese_bowl'),(0,'wine_rack')]:
        cell=d[(d.scene_task==task)&(d.requested_goal==goal)]; assert len(cell)==3
        for seed,marker,shift,color in [(3,'o',.2,blue),(4,'s',0,orange),(5,'^',-.2,'#316b51')]:
            v=float(cell.loc[cell.training_seed.eq(seed),'gr00t_minus_qwen_pp'].iloc[0]); ax.scatter(v,y+shift,marker=marker,s=25,color=color,label=f'Seed {seed}' if y==1 else None); ax.annotate(f'{v:+g}',(v,y+shift),xytext=(4,0),textcoords='offset points',va='center',fontsize=7)
    ax.set_yticks([1,0],['Cheese: original','Wine: swapped'] if task==6 else ['Cheese: swapped','Wine: original']); ax.set_title(f'Task {task} scene'); ax.axvline(0,color='#555',lw=.8); ax.set_xlim(-25,33); ax.set_xticks([-20,-10,0,10,20,30]); ax.grid(axis='x',alpha=.2); ax.set_xlabel('GR00T − Qwen (percentage points)')
axes[0].legend(frameon=False,loc='lower right',fontsize=6.5,ncol=1); fig.tight_layout(w_pad=1.2); save(fig,'libero_goal_comparison_review.pdf')

d=pd.read_csv(ROOT/'docs/analysis/stage3/stage3_instruction_sensitivity.csv')
cats=[('both','Both episodes','#246b98',''),('canonical_only','Original only','#d28a35','//'),('swapped_only','Swapped only','#65967a','..'),('neither','Neither','#d8dfe4','xx')]
fig,axes=plt.subplots(2,2,figsize=(7.16,3.15),sharex=True,sharey=True)
for ax,(task,model,title) in zip(axes.flat,[(6,'groot','Task 6 · GR00T'),(6,'qwen3vl','Task 6 · Qwen'),(9,'groot','Task 9 · GR00T'),(9,'qwen3vl','Task 9 · Qwen')]):
    sub=d[(d.task==task)&(d.backbone==model)].sort_values('training_seed'); assert len(sub)==3 and sub[['both','canonical_only','swapped_only','neither']].sum(axis=1).eq(40).all()
    for y,(_,r) in zip([2,1,0],sub.iterrows()):
        left=0
        for field,label,color,hatch in cats:
            value=int(r[field]); ax.barh(y,value,left=left,height=.6,color=color,hatch=hatch,edgecolor='white',linewidth=.4)
            if value: ax.text(left+value/2,y,str(value),ha='center',va='center',fontsize=7,color='white' if field=='both' else '#111')
            left+=value
    ax.set_yticks([2,1,0],['Seed 3','Seed 4','Seed 5']); ax.set_xlim(0,40); ax.set_xticks([0,10,20,30,40]); ax.set_title(title); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
for ax in axes[-1]: ax.set_xlabel('Matched starts (out of 40)')
fig.legend(handles=[Patch(facecolor=c,hatch=h,label=l) for _,l,c,h in cats],loc='upper center',bbox_to_anchor=(.5,1),ncol=4,frameon=False,fontsize=7)
fig.tight_layout(rect=(0,0,1,.91),h_pad=.8,w_pad=1.5); save(fig,'libero_paired_instruction_review.pdf')

d=pd.read_csv(ROOT/'docs/analysis/stage5/checkpoint_anchor_alignment.csv')
fig,axes=plt.subplots(3,2,figsize=(7.16,4.0),sharex='col')
for col,(dataset,title) in enumerate([('aloha','ALOHA'),('libero','LIBERO')]):
    for row,(field,lab) in enumerate([('train_loss','Training loss'),('validation_loss','Validation loss'),('development_success','Success (%)')]):
        ax=axes[row,col]
        for model,color,marker,ls in [('GR00T',blue,'o','-'),('Qwen',orange,'s','--')]:
            sub=d[(d.dataset==dataset)&(d.backbone==model)]; k=100 if row==2 else 1
            for _,s in sub.groupby('training_seed'): ax.plot(s.epoch,s[field]*k,color=color,alpha=.25,lw=.7)
            means=sub.groupby('epoch')[field].mean()*k; ax.plot(means.index,means,color=color,marker=marker,ms=2.5,lw=1.3,ls=ls,label=model)
        ax.set_title(f'{title}: {lab}'); ax.grid(alpha=.18)
        if row<2: ax.set_yscale('log'); ax.set_ylabel('Flow loss (log)')
        else: ax.set_ylim(0,100); ax.set_ylabel('Success (%)'); ax.set_xlabel('Checkpoint epoch')
        if row==1: ax.axvline(25,color='#666',ls=':',lw=1)
axes[0,0].legend(frameon=False,fontsize=7); fig.tight_layout(h_pad=.6,w_pad=1.1); save(fig,'training_progression_review.pdf')
print('Built four project-owned manuscript figures')
