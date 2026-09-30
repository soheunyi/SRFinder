"""Render new-store power summaries without reading or overwriting legacy caches."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from run_files.generate_continuous_affine_power_figures import plot_power,plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tex',action='store_true')
    args=ap.parse_args();audit=json.loads((args.summary/'audit.json').read_text())
    path=args.summary/'power_figure_summary.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['outputs'][path.name]:raise ValueError('Summary checksum differs')
    data=pd.read_csv(path);plt.rcParams['text.usetex']=args.tex
    null=data[(data.signal=='HH4b')&(data.epsilon==0)].copy()
    args.output.mkdir(parents=True,exist_ok=False);files=[]
    for family in sorted(data.signal.unique()):
        selected=data[data.signal==family].copy()
        if family!='HH4b' and not (selected.epsilon==0).any():
            shared=null.copy();shared['signal']=family;selected=pd.concat([shared,selected],ignore_index=True)
        if selected.duplicated(['epsilon','sr_fraction']).any():raise ValueError('Duplicate plot cell')
        selected=selected.rename(columns={'epsilon':'signal_ratio','sr_fraction':'SR_size'})
        prefix='' if audit['scope']=='full' else audit['scope']+'_'
        name=prefix+f'power_plot_{family}_noise_scale=2.0.pdf'
        plot_power(selected,args.output/name);files.append(name)
    (args.output/'provenance.json').write_text(json.dumps({'summary_audit_sha256':hashlib.sha256((args.summary/'audit.json').read_bytes()).hexdigest(),
        'scope':audit['scope'],'primary_rule':audit['recipe']['decision']['primary_rule'],'files':files,'tex':args.tex},indent=2)+'\n')
    print(json.dumps({'status':'POWER_FIGURES_WRITTEN','scope':audit['scope'],'files':files}),flush=True)

if __name__=='__main__':main()
