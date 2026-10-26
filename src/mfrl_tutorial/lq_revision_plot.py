"""Redraw the five revised book figures from numeric data only.

No training, model, SciPy, native library or research-tree import is needed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, StrMethodFormatter, NullLocator
from . import lq_revision_style as base
from .lq_revision_style import moving_average,thin_style as style,COLORS

METHODS=['Exact','MKV','Pop_N2','Pop_N10','Pop_N100']
FIELDS=['MF_relative_error','K_Frobenius_error','L_Frobenius_error']

def configure_style():
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
        'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65})

def read_dataset(root,description):
    if description.get('partial',False):
        raise ValueError('Partial smoke data cannot replace the twenty-run book aggregate')
    path=root/description['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=description['sha256']:
        raise ValueError('Data checksum mismatch: '+path.name)
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    if set(data)!=set(description['arrays']):raise ValueError('Array key mismatch')
    if len(set(description['seeds']))!=20:raise ValueError('Expected twenty distinct seed IDs')
    for method in description['methods']:
        if data[method+'__theta'].shape[:2]!=(20,801):
            raise ValueError('Book plots require twenty complete800-update trajectories per method')
    for key,value in data.items():
        spec=description['arrays'][key]
        if list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype']:
            raise ValueError('Array schema mismatch: '+key)
        if not np.isfinite(value).all():raise ValueError('Nonfinite data: '+key)
    return data

def export_scalar(raw,optimum,out):
    base.OUT=out
    summary={"optimum_gains":optimum}
    records,displayed={},{}
    def panel(ax,j):
        base.decorate(ax,'Relative cost error (%)' if j==0 else ['$K$','$L$'][j-1],log=j==0)
        if j==0:
            ax.set_ylim(1e-5,200)
        ax.set_title(['(a) Mean-field cost error: mean $\\pm$ SEM',
                      '(b) $K$: mean $\\pm$ SD','(c) $L$: mean $\\pm$ SD'][j])
        moments={}
        for m in base.METHODS1:
            value=raw[m+'__mf_relative_gap'] if j==0 else raw[m+'__theta'][...,j-1]
            factor=100 if j==0 else 1
            if m=='Exact':
                mean=factor*value
                spread=None
            else:
                smoothed=moving_average(value,21)
                mean=factor*smoothed.mean(axis=0)
                spread=factor*smoothed.std(axis=0,ddof=1)/(np.sqrt(20) if j==0 else 1)
                if j==0:
                    assert np.all(mean-spread>0)
                ax.fill_between(base.X,mean-spread,mean+spread,color=base.COLORS[m],
                                alpha=.26 if j==0 else .17,lw=0,zorder=1)
            moments[m]=mean
            displayed[f"B_MA21_linear__panel{j}__{m}__mean"]=mean
            if spread is not None:displayed[f"B_MA21_linear__panel{j}__{m}__band"]=spread
        if j:
            ax.axhline(summary['optimum_gains'][j-1],color=base.COLORS['Optimum'],ls='--',lw=.85,zorder=3)
        for m in [v for v in base.METHODS1 if v!='MKV']+['MKV']:
            ax.plot(base.X,moments[m],label=base.LABELS[m],**style(m))
        assert ax.lines[-1].get_label()=='MKV'

    def legend(fig,location='outside lower center'):
        handles=[Line2D([],[],label=base.LABELS[m],**style(m)) for m in base.METHODS1]
        handles.append(Line2D([],[],color=base.COLORS['Optimum'],ls='--',lw=.85,label='Optimum'))
        fig.legend(handles=handles,loc=location,ncol=3,frameon=False,
                   handlelength=1.9,columnspacing=.9,handletextpad=.5)

    caption=('Training with exact mean-field gradients, an MKV simulator, and population simulators (N=2,10,100). '
             'Each stochastic run uses a centred 21-point moving average, shortened symmetrically near endpoints, '
             'before aggregation across 20 runs. Cost bands are SEM; gain bands are one sample SD. '
             'Grey dashed lines show common optimal gains. The exact-gradient reference is deterministic and unsmoothed.')
    for j,name in enumerate(['training_cost','training_K','training_L']):
        fig,ax=plt.subplots(figsize=(4.65,2.45) if j==0 else (2.26,2.45),layout='constrained')
        panel(ax,j)
        base.export(fig,name,caption,records)
    fig=plt.figure(figsize=(4.65,.46))
    legend(fig,'center')
    base.export(fig,'training_legend','Shared legend for the selected scalar training panels.',records)
    np.savez_compressed(out/"displayed_curves_1d.npz",**displayed)
    return records

def export_population(scalar,out):
    base.OUT=out
    ns=[2,5,10,20,50,100]
    records={}
    fig, axes = plt.subplots(1,2,sharey=True,figsize=(4.65,2.85),layout='constrained')
    population_records = {}
    for ax, field, title in zip(axes,['mf_relative_gap','own_relative_gap'],['(a) Mean-field','(b) Finite population']):
        vals = np.array([100*scalar[f'Pop_N{n}__{field}'][:,601:801].mean(axis=1) for n in ns])
        means, sem = vals.mean(axis=1), vals.std(axis=1,ddof=1)/np.sqrt(20)
        collections = []
        for i,n in enumerate(ns):
            points = ax.scatter(np.full(20,n), vals[i], marker='o', s=8,facecolors='none',
                                edgecolors='.45',linewidths=.45,alpha=.5,zorder=2)
            np.testing.assert_array_equal(points.get_offsets()[:,0],np.full(20,n))
            collections.append(points)
        ax.errorbar(ns,means,yerr=sem,color='.25',marker='D',ms=3.2,capsize=2,lw=1.1,zorder=5)
        if field=='mf_relative_gap':
            ref = 100*scalar['MKV__mf_relative_gap'][:,601:801].mean(axis=1)
            mu,se=ref.mean(),ref.std(ddof=1)/np.sqrt(20)
            ax.axhspan(mu-se,mu+se,color=COLORS['MKV'],alpha=.09,zorder=0)
            ax.axhline(mu,**base.style('MKV'))
        ax.set(xscale='log',yscale='log',xticks=ns,xticklabels=[str(n) for n in ns],
               xlabel='Population size $N$',title=title,xlim=(1.6,130),ylim=(.018,.75))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator([.02,.05,.1,.2,.5]))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        ax.grid(alpha=.18,lw=.5)
        population_records[field]=dict(values_percent=vals.tolist(),means_percent=means.tolist(),
                                       SEM_percentage_points=sem.tolist(),x_coordinates=ns)
    axes[0].set_ylabel('Relative cost error (%)')
    pop_handles=[Line2D([],[],color='.25',marker='D',ms=3.2,label='Population mean $\\pm$ SEM'),
                 Line2D([],[],label='MKV mean $\\pm$ SEM',**base.style('MKV')),
                 Line2D([],[],color='.45',marker='o',mfc='none',ls='',ms=3.2,label='Individual runs')]
    lg=fig.legend(handles=pop_handles,loc='outside lower center',ncol=3,frameon=False,
                  handlelength=1.25,columnspacing=.8,handletextpad=.4)
    fig.canvas.draw()
    # Matplotlib's row packing must actually produce a single legend row.
    text_boxes=[t.get_window_extent(fig.canvas.get_renderer()) for t in lg.get_texts()]
    assert max(b.y0 for b in text_boxes)<min(b.y1 for b in text_boxes)
    base.export(fig,'chapter_population_size','Scalar population-size comparison. Circles are the 20 run averages over updates 601-800, placed at their exact N without horizontal offsets. Dark grey diamonds and bars show mean plus or minus SEM. The dotted blue MKV reference and band show mean plus or minus SEM. Panels have the same vertical scale.',records)

    return records,population_records

def export_matrix(data,optimum,out):
    OUT=out
    base.OUT=OUT
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
        'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.65})
    records,displayed={},{}
    def moments(field,method,entry=None):
        value=data[method+'__'+field]
        if entry is not None:value=value[(slice(None),)*(1 if method=='Exact' else 2)+entry]
        if method=='Exact':return value,None
        smoothed=moving_average(value,21)
        mean,spread=smoothed.mean(axis=0),smoothed.std(axis=0,ddof=1)
        if field=='MF_relative_error':spread/=np.sqrt(20)
        key=field+('' if entry is None else '_'+''.join(map(str,entry)))+'__'+method
        displayed[key+'__mean']=mean;displayed[key+'__band']=spread
        return mean,spread

    def draw(ax,field,entry=None,factor=1):
        curves={}
        for method in METHODS:
            mean,spread=moments(field,method,entry)
            curves[method]=factor*mean
            if spread is not None:
                mean,spread=factor*mean,factor*spread
                if ax.get_yscale()=='log':assert np.all(mean-spread>0)
                ax.fill_between(base.X,mean-spread,mean+spread,color=base.COLORS[method],
                    alpha=.26 if field=='MF_relative_error' else .17,lw=0,zorder=1)
        for method in [m for m in METHODS if m!='MKV']+['MKV']:
            ax.plot(base.X,curves[method],label=base.LABELS[method],**style(method))
        assert ax.lines[-1].get_label()=='MKV'

    def legend(fig,optimum=False,location='outside lower center'):
        handles=[Line2D([],[],label=base.LABELS[m],**style(m)) for m in METHODS]
        if optimum:handles.append(Line2D([],[],label='Optimum',color=base.COLORS['Optimum'],ls='--',lw=.85))
        fig.legend(handles=handles,loc=location,ncol=3,frameon=False,handlelength=1.9,
                   columnspacing=.9,handletextpad=.5)

    common=('Coupled two-dimensional example; 20 runs per stochastic method. '
        'Stochastic trajectories use a centred 21-point moving average, shortened symmetrically near endpoints, '
        'before across-run aggregation. The exact MF-gradient reference is deterministic and unsmoothed. '
        'Blue dotted MKV is drawn last, above orange N=100; N=2 purple, N=10 green dash-dot. ')
    def summary_panel(ax,j):
        labels=['Relative cost error (%)',r'$\|K-K^*\|_F$',r'$\|L-L^*\|_F$']
        titles=['(a) Mean-field cost error: mean $\\pm$ SEM',
                '(b) $K$ error: mean $\\pm$ SD','(c) $L$ error: mean $\\pm$ SD']
        base.decorate(ax,labels[j],log=True);ax.set_title(titles[j])
        ax.set_ylim([(1e-5,100),(1e-3,.5),(5e-6,1)][j])
        draw(ax,FIELDS[j],factor=100 if j==0 else 1)

    summary_caption=common+("Cost bands are SEM; gain-error bands one sample SD. Norms precede per-run smoothing and averaging; exact reference unsmoothed.")
    parts=OUT
    for j,name in enumerate(['training_cost_2d','training_K_error_2d','training_L_error_2d']):
        fig,ax=plt.subplots(figsize=(4.65,2.45) if j==0 else (2.26,2.45),layout='constrained')
        summary_panel(ax,j);base.OUT=parts
        base.export(fig,name,summary_caption,records);base.OUT=OUT
    fig=plt.figure(figsize=(4.65,.46));legend(fig,False,'center');base.OUT=parts
    base.export(fig,'training_legend_2d','Shared legend for the independent two-dimensional summary panels.',records);base.OUT=OUT
    for block,label in [(0,'K'),(1,'L')]:
        fig,axes=plt.subplots(2,2,figsize=(4.65,4.85),layout='constrained')
        for (i,j),ax in zip(np.ndindex(2,2),axes.flat):
            base.decorate(ax,f'${label}_{{{i+1}{j+1}}}$')
            ax.set_title(f'({"abcd"[2*i+j]}) ${label}_{{{i+1}{j+1}}}$: mean $\\pm$ SD')
            ax.axhline(optimum[block,i,j],color=base.COLORS['Optimum'],ls='--',lw=.85,zorder=3)
            draw(ax,'theta',entry=(block,i,j))
        legend(fig,True)
        base.export(fig,label+'_entries_2d',common+
            f'All four signed entries of {label}; mean +/- one sample SD. Grey dashed lines show '
            'the corresponding common MF/finite-population optimal entries. Small entries can remain noisy despite low cost.',records)

    np.savez_compressed(out/'displayed_curves_2d.npz',**displayed)
    return records


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,type=Path,
                    help='Local directory produced by lq_revision_analyze; datasets are not distributed')
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--dimension',choices=['both','1d','2d'],default='both')
    args=ap.parse_args()
    metadata_path=args.data/'lq_revision_metadata.json'
    if not metadata_path.is_file():
        ap.error('No lq_revision_metadata.json in --data; run lq_revision_analyze first')
    metadata=json.loads(metadata_path.read_text())
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    configure_style()
    dimensions=['1d','2d'] if args.dimension=='both' else [args.dimension]
    records,popstats={},{}
    if '1d' in dimensions:
        scalar=read_dataset(args.data,metadata['datasets']['1d'])
        records.update(export_scalar(scalar,metadata['datasets']['1d']['optimum'],out))
        population,popstats=export_population(scalar,out)
        records.update(population)
    if '2d' in dimensions:
        matrix=read_dataset(args.data,metadata['datasets']['2d'])
        records.update(export_matrix(matrix,np.asarray(metadata['datasets']['2d']['optimum']),out))
    figure_map={k:v for k,v in metadata['figure_map'].items() if v['dataset'] in dimensions}
    expected={f for fig in figure_map.values() for f in fig['files']}
    assert {k+'.pdf' for k in records}==expected
    manifest=dict(status='passed',figures=records,figure_map=figure_map,
        data_sha256={k:v['sha256'] for k,v in metadata['datasets'].items() if k in dimensions},
        population_statistics=popstats,reporting=metadata['reporting'],
        imported_package_files={k:str(Path(v.__file__).resolve().relative_to(Path(__file__).resolve().parents[2]))
            for k,v in sys.modules.items() if k.startswith('mfrl_tutorial') and getattr(v,'__file__',None)},
        simulator_calls=0,training_calls=0)
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',PDF_count=len(records),output=str(out),training_calls=0)))

if __name__=='__main__':main()
