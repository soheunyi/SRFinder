"""Pruning completed working files must preserve verifiable, reusable artifacts."""
import argparse,fcntl,json,pathlib,sys
from unittest.mock import patch
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'phase5')]
from test_campaign_runtime import fixture
from artifacts.training_store import TrainingStore
from artifacts.campaign_runtime import run_campaign,import_sources
from artifacts.case_registry import CaseRegistry
from artifacts.bound_tasks import build_stage_contexts
from artifacts.run_stage import run_stage
from artifacts.stage_processes import _initialize
from artifacts.cleanup_completed import prune_completed_case

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);_initialize()
    plan=fixture(args.out);store=TrainingStore(args.out/'store');import_sources(store,plan)
    case=next(n['id'] for n in plan['nodes'] if n['stage']==1)
    root=args.out/'execution'
    run_campaign(store,plan,root,case_ids=[case],nproc=1,device='cpu',export_batch_size=128)
    before=store.storage_stats()
    dry=prune_completed_case(store,root,case)
    assert len(dry['files'])==5 and dry['reclaimable_bytes']>0
    assert all((root/r['path']).exists() for r in dry['files'])
    with (root/'.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:prune_completed_case(store,root,case,apply=True)
        except RuntimeError:pass
        else:raise AssertionError('Active coordinator was pruned')
    registry=CaseRegistry(store,plan,root/'registry',resume=True)
    value=registry.get(case)
    payload=store.payload_path(value['completion']['model_ids'][0])
    backup=payload.with_suffix('.held');payload.rename(backup)
    try:
        try:prune_completed_case(store,root,case,apply=True)
        except (ValueError,FileNotFoundError):pass
        else:raise AssertionError('Cleanup accepted missing permanent weights')
        assert all((root/r['path']).exists() for r in dry['files'])
    finally:backup.rename(payload)
    # Never prune changed best weights or a symlink, even with a valid receipt.
    best=next(root/r['path'] for r in dry['files'] if r['path'].endswith('_best.pt'))
    original=best.read_bytes()
    best.write_bytes(original+b'changed')
    try:
        try:prune_completed_case(store,root,case,apply=True)
        except ValueError:pass
        else:raise AssertionError('Cleanup accepted changed working best weights')
    finally:best.write_bytes(original)
    best.unlink();best.symlink_to(payload)
    try:
        try:prune_completed_case(store,root,case,apply=True)
        except ValueError:pass
        else:raise AssertionError('Cleanup accepted a symlink')
    finally:best.unlink();best.write_bytes(original)
    models=best.parent
    moved=models.with_name('models-held');models.rename(moved);models.symlink_to(moved)
    try:
        try:prune_completed_case(store,root,case,apply=True)
        except ValueError:pass
        else:raise AssertionError('Cleanup accepted a symlinked working directory')
    finally:models.unlink();moved.rename(models)
    unlink=pathlib.Path.unlink
    def interrupted_unlink(path,*args,**kwargs):
        if path==best:raise RuntimeError('Synthetic interruption after first deletion')
        return unlink(path,*args,**kwargs)
    with patch.object(pathlib.Path,'unlink',new=interrupted_unlink):
        try:prune_completed_case(store,root,case,apply=True)
        except RuntimeError:pass
        else:raise AssertionError('Expected cleanup interruption')
    journal=json.loads(next((root/'cleanup').glob('*.json')).read_text())
    assert len(journal['files'])==5 and len(journal['deleted_paths'])==1 and not journal['applied']
    done=prune_completed_case(store,root,case,apply=True)
    assert len(done['files'])==5 and len(done['deleted_paths'])==5
    assert done['applied'] and all(not (root/r['path']).exists() for r in done['files'])
    assert prune_completed_case(store,root,case,apply=True)==done
    assert store.storage_stats()==before
    contexts,source=build_stage_contexts(store,value['task'])
    with patch('pytorch_lightning.Trainer.fit',side_effect=AssertionError('Unexpected refit')):
        reused=run_stage(store,contexts,source,root/'tasks'/case,device='cpu',resume=True,export_batch_size=128)
    assert reused['completion_id']==value['completion_id']
    result={'status':'PASS','removed_files':len(done['files']),'reclaimed_bytes':done['reclaimable_bytes'],
            'permanent_artifacts_preserved':True,'completed_reuse_without_checkpoint':True,
            'active_output_and_missing_payload_rejected':True}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
