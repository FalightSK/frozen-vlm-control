from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd(); d=pd.read_csv(root/'docs/analysis/stage4/aloha_boundary_queries.csv')
s=d[(d.outcome=='groot_only')&(d.event=='before_score2_chunk')]
fig,ax=plt.subplots(figsize=(3.5,2.7))
colors={'groot':'#2c6db2','qwen3vl':'#d77c17'}
for idx,(seed,source) in enumerate(( (3,'groot'),(3,'qwen3vl'),(4,'groot'),(4,'qwen3vl'),(5,'groot'),(5,'qwen3vl') )):
    y=5-idx; t=s[(s.seed==seed)&(s.source==source)].set_index('query_model'); assert len(t)==2
    g=t.loc['groot','mean_right_gripper']; q=t.loc['qwen3vl','mean_right_gripper']
    ax.plot([g,q],[y+.07,y-.07],color='#999999',lw=.7,zorder=1)
    ax.scatter(g,y+.07,s=29,marker='o',color=colors['groot'],zorder=3,label='GR00T' if idx==0 else None)
    ax.scatter(q,y-.07,s=29,marker='s',color=colors['qwen3vl'],zorder=3,label='Qwen' if idx==0 else None)
ax.set_yticks(range(5,-1,-1),['3 · GR00T path','3 · Qwen path','4 · GR00T path','4 · Qwen path','5 · GR00T path','5 · Qwen path'])
ax.set_xlim(0,1.24); ax.set_ylim(-.6,5.8); ax.set_xlabel('Mean right-gripper target (0 closed, 1 open)',fontsize=8)
ax.tick_params(labelsize=8); ax.grid(axis='x',alpha=.2); ax.legend(loc='upper right',ncol=2,fontsize=7,frameon=False)
fig.tight_layout(pad=.4); fig.savefig(root/'manuscript/figures/aloha_boundary_compact.pdf',bbox_inches='tight')
