"""Plot the sampled strong-well spectral areas, without interpolation or rescaling.

Run from the repository root:
    python assets/img/pimd-series/trpmd-anharmonic/plot_spectral_area.py
The JSON input and PNG output default to this script's directory.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=here/'metrics.json')
    parser.add_argument('--output',type=Path,default=here)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.data.read_text(encoding='utf-8'))
    rows=sorted((x for x in data['metrics'] if x['alpha']==.5),key=lambda x:x['lambda_'])
    x=np.array([z['lambda_'] for z in rows]);assert np.all(np.diff(x)>0)
    colors=['#c64c38','#6f5ca8','#176f9b','#29946d']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'savefig.dpi':190,'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.1))
    for ax,key,se,exact,title in [
        (axs[0],'C0','C0_SE','C0_exact','All-positive-frequency area: C(0)'),
        (axs[1],'band_area','band_area_SE','band_area_exact','Displayed main-band area: [1.25, 2.15]')]:
        for i,z in enumerate(rows):
            assert np.isfinite([z[key],z[se],z[exact]]).all() and z[se]>=0
            ax.errorbar(x[i],z[key],yerr=z[se],fmt='o',color=colors[i],capsize=4,ms=7)
        ax.axhline(rows[0][exact],color='#202020',ls='--',lw=1.4,label='Exact')
        ax.set(xticks=x,xticklabels=['RPMD','TR .5','TR 1','TR 2'],ylabel='Absolute area',title=title,xlim=(-.2,2.2))
        ax.legend(fontsize=9);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle('Strong quartic, alpha=0.5; markers and one batch standard error',fontsize=13)
    fig.tight_layout();fig.savefig(args.output/'spectral_area.png',bbox_inches='tight')
    svg=args.output/'spectral_area.svg';fig.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(z.rstrip() for z in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
    quality={'finite':True,'strictly_increasing_friction':True,'duplicate_points':0,'representation':'discrete markers, no interpolation or continuous fit','standard_error':'independent batch SE; RPMD uses 32 batches, TRPMD 16','source_curve_gates':'metrics from validated production; no new continuous sampled curve generated'}
    (args.output/'spectral_area_quality.json').write_text(json.dumps(quality,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(quality))

if __name__=='__main__':main()
