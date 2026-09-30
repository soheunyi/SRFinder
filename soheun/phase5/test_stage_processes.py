"""CPU spawn-process composition test, not a performance benchmark or campaign."""
import argparse,hashlib,json,pathlib,sys
import numpy as np
import pandas as pd
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from constants import FEATURES
from dataset import SCDatasetInfo
from artifacts.training_store import TrainingStore,canonical
from artifacts.source_context import verify_source_context
from artifacts.step1_context import ArtifactStep1Context
from artifacts.stage_processes import run_tasks
from test_three_stage_artifacts import hparams


def context_factory(store,spec):
    if not pathlib.Path(spec['pool']).is_file():raise FileNotFoundError(spec['pool'])
    raw=SCDatasetInfo([pathlib.Path(spec['pool'])],[np.ones(spec['rows'],dtype=bool)])
    hp=spec['source_hparams']
    source=verify_source_context(store,spec['dataset_id'],hp,raw,source_root=pathlib.Path(spec['pool']).parent)
    contexts=[ArtifactStep1Context(source,hparams(1,seed)) for seed in (0,1)]
    return contexts,source


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--tasks',type=int,default=2);ap.add_argument('--nproc',type=int,default=2)
    args=ap.parse_args()
    if args.tasks<2:raise ValueError('At least two tasks are needed')
    options={'device':args.device,'safety_bytes':1024**3,'compute_headroom_bytes':1024**3}
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(31)
    n=512;x=torch.rand(n,4,4)
    x[:,0]=40+100*x[:,0];x[:,1]=2*x[:,1]-1;x[:,2]=6*x[:,2]-3;x[:,3]=5+15*x[:,3]
    frame=pd.DataFrame(x.reshape(n,16).numpy(),columns=FEATURES)
    frame['fourTag']=np.arange(n)%2;frame['weight']=1.
    path=(args.out/'pool.h5').resolve();frame.to_hdf(path,key='df')
    store=TrainingStore(args.out/'store');mask=np.ones(n,dtype=bool);tasks=[]
    for seed in range(7,7+args.tasks):
        params={'seed':seed,'n_3b':n//2,'ratio_4b':.5,'signal_ratio':0.,
                'signal_filename':path.name,'base_fvt_train_ratio':.5}
        hp={**hparams(1,0),'dataset':params}
        descriptor={'pools':[{'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],
                    'mother_parameters':params,'mother_selection_sha256':[hashlib.sha256(mask.tobytes()).hexdigest()]}
        dataset=store.put_dataset(hashlib.sha256(canonical(descriptor)).hexdigest(),n,descriptor)
        tasks.append({'pool':str(path),'rows':n,'source_hparams':hp,'dataset_id':dataset})
    serial=run_tasks(store,tasks,context_factory,args.out/'serial',nproc=1,export_batch_size=128,**options)
    parallel=run_tasks(store,tasks,context_factory,args.out/'parallel',nproc=args.nproc,export_batch_size=128,**options)
    assert serial==parallel
    checkpoints=list((args.out/'parallel').glob('*/training/last.ckpt'))
    assert len(checkpoints)==len(tasks)
    before={p:p.stat().st_mtime_ns for p in checkpoints}
    resumed=run_tasks(store,tasks,context_factory,args.out/'parallel',nproc=1,resident=False,resume=True,export_batch_size=128,**options)
    assert resumed==parallel and all(p.stat().st_mtime_ns==stamp for p,stamp in before.items())
    assert json.loads((args.out/'parallel'/'latest-execution.json').read_text())['active_worker_limit']==1
    try:run_tasks(store,tasks+tasks[:1],context_factory,args.out/'duplicates')
    except ValueError:pass
    else:raise AssertionError('Duplicate tasks accepted')
    # A missing source fails one child; the coordinator must not mark success.
    delayed=(args.out/'delayed'/'pool.h5').resolve()
    recovery_tasks=[{**tasks[0],'pool':str(delayed)},tasks[1]]
    try:run_tasks(store,recovery_tasks,context_factory,args.out/'recovery',nproc=2,export_batch_size=128,**options)
    except FileNotFoundError:pass
    else:raise AssertionError('Missing-source worker failure was ignored')
    assert not (args.out/'recovery'/'completion.json').exists()
    delayed.parent.mkdir();delayed.write_bytes(path.read_bytes())
    recovered=run_tasks(store,recovery_tasks,context_factory,args.out/'recovery',nproc=1,resume=True,export_batch_size=128,**options)
    assert recovered['stage_completion_ids']==serial['stage_completion_ids'][:2]
    (args.out/'report.json').write_text(json.dumps({'status':'PASS','tasks':len(tasks),'members_per_task':2,'device':args.device,'parallel_workers':args.nproc,
        'serial_parallel_artifacts_exact':True,'resume_worker_override_without_refit':True,'worker_failure_recovery':True},indent=2)+'\n')
    print('PASS: exact serial/spawn artifact identities and lower-worker resume without refitting',flush=True)


if __name__=='__main__':main()
