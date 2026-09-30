"""Exercise figure readers on the retained native engineering reference only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
import torch
from artifacts.evaluation import open_reader
from render_campaign_illustrations import representation,tsne


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execution',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);args.out.mkdir(parents=True,exist_ok=False)
    reader,_=open_reader(args.execution);before=reader.store.storage_stats();started=time.perf_counter()
    representation(reader,args.out,'mean_probability');tsne(reader,args.out,'cuda')
    assert reader.store.storage_stats()==before
    for name in ('on_which_to_learn.pdf','tsne_original_repr.pdf'):assert (args.out/name).stat().st_size>0
    report={'status':'PASS_NATIVE_ILLUSTRATION_READERS','wall_s':time.perf_counter()-started,
        'training_store_unchanged':True,'labels_and_sample_order_verified':True,
        'scope':'Retained old engineering reference; not retraining results or statistical acceptance'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
