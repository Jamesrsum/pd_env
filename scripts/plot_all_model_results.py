#!/usr/bin/env python3
"""Export aggregate model comparisons only; never opens predictions/test labels."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--leaderboard',required=True)
    p.add_argument('--summary',default='docs/results/frozen_development_summary.json')
    p.add_argument('--output',default='docs/results/all_models')
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(args.leaderboard)
    allowed=['model','context','arm','representation','head','auroc','log_loss','participants','threshold','specificity','sensitivity','peak_gpu_gib','seconds_per_recording']
    table=source[allowed].copy()
    if table[['auroc','log_loss']].isna().any().any():raise ValueError('Invalid comparison metrics')
    table.to_csv(out/'all_364_development_pipelines.csv',index=False)
    summary=json.loads(Path(args.summary).read_text());age=summary['covariate_baselines']['age_only']['auroc']
    order=table[table.arm.eq('technical_primary')].groupby('model').auroc.max().sort_values().index.tolist()
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,arm in zip(axes,['technical_primary','pitch_sensitivity']):
        data=table[table.arm.eq(arm)].groupby(['model','context']).auroc.max().unstack().reindex(order)
        y=list(range(len(order)))
        for context,color,offset in [('full','#285f8f',-.14),('matched_2s','#c06324',.14)]:
            ax.scatter(data[context],[v+offset for v in y],label=context,s=45,color=color)
        if arm=='technical_primary':
            ax.axvline(age,color='#666666',ls='--',label=f'Age-only {age:.3f}')
        counts=table.loc[table.arm.eq(arm),'participants'].unique()
        ax.set(xlim=(.75,.96),xlabel='Participant validation AUROC',title=arm.replace('_',' ').title()+f' (n={"/".join(map(str,counts))})')
        ax.set_yticks(y,order);ax.grid(axis='x',alpha=.2)
        ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('All nine frozen models: best validation-selected configuration per context\nDifferent arm eligibility; no held-out test results',fontsize=12)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/f'all_models_validation.{ext}',dpi=300,bbox_inches='tight')
    svg=out/'all_models_validation.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    # The completed-report builder owns the cross-experiment model_status.csv.
    # Do not overwrite it with historical assumptions about unrun adaptation.
    status=pd.DataFrame([{'model':m,'regime':'frozen_encoder','status':'completed','auroc':table[(table.model.eq(m)) & table.arm.eq('technical_primary')].auroc.max()} for m in order])
    status.to_csv(out/'frozen_model_status.csv',index=False)
    print(f'Exported {len(table)} aggregate pipelines and all-model PNG/SVG comparison')


if __name__=='__main__':main()
