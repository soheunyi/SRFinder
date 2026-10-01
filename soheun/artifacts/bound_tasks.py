"""Bind stage recipes to verified source selections and upstream artifacts.

The source pointer references a trusted existing local MotherSamples pickle;
its exact bytes are fingerprinted before loading. This is not an untrusted
pickle ingestion service. No legacy cache or split-index array is written.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from dataset import MotherSamples
from .training_store import canonical,sha
from .source_context import verify_source_context,_file_sha
from .source_fingerprints import SourceFingerprints

# Only digests/file stamps persist; no MotherSamples masks or feature arrays.
_SOURCE_FINGERPRINTS = SourceFingerprints()
from .step1_context import ArtifactStep1Context
from .step2_context import ArtifactStep2Context
from .step3_context import ArtifactStep3Context
from .step3_representation import ArtifactStep3RepresentationContext


def _load_mother(path,expected=None):
    content=Path(path).read_bytes();fingerprint=hashlib.sha256(content).hexdigest()
    if expected is not None and fingerprint!=expected:raise ValueError('Mother selection record changed')
    mother=pickle.loads(content)
    if not isinstance(mother,MotherSamples):raise ValueError('Expected a trusted MotherSamples record')
    return mother,fingerprint


def _check_source_recipe(mother,hparams):
    if hparams.get('step')!=1 or hparams.get('model')!='FvTClassifier':
        raise ValueError('Source recipe must define the Step-1 outer split')
    params=hparams['dataset']
    if any(params.get(key)!=value for key,value in mother.hparams.items()):
        raise ValueError('Mother record parameters differ from source recipe')


def register_source_pointer(store,mother_record,source_hparams,*,source_root,fingerprints=None):
    """Register existing raw pools/selection once; return a JSON worker pointer."""
    path=Path(mother_record).resolve();source_root=Path(source_root).resolve()
    mother,fingerprint=_load_mother(path);_check_source_recipe(mother,source_hparams)
    raw=mother.scdinfo
    paths=[(Path(p) if Path(p).is_absolute() else source_root/p).resolve() for p in raw.files]
    if any(np.asarray(mask).dtype!=np.bool_ for mask in raw.inner_idxs):raise ValueError('Mother masks must be boolean')
    descriptor={'pools':[{'name':p.name,'sha256':fingerprints.sha256(p) if fingerprints is not None else _file_sha(p)} for p in paths],
        'mother_parameters':deepcopy(source_hparams['dataset']),
        'mother_selection_sha256':[hashlib.sha256(np.asarray(mask).tobytes()).hexdigest() for mask in raw.inner_idxs]}
    dataset_id=store.put_dataset(sha(canonical(descriptor)),len(raw),descriptor)
    verify_source_context(store,dataset_id,source_hparams,raw,source_root=source_root,fingerprints=fingerprints)
    return {'dataset_id':dataset_id,'mother_record':str(path),'mother_record_sha256':fingerprint,
            'source_root':str(source_root),'hparams':deepcopy(source_hparams)}


def make_stage_task(source_pointer,member_hparams,*,upstream=None):
    if not member_hparams:raise ValueError('Empty member list')
    stages={hp['step'] for hp in member_hparams}
    if len(stages)!=1 or next(iter(stages)) not in (1,2,3):raise ValueError('Mixed or unsupported stages')
    # Freeze a JSON-only recipe rather than keeping mutable caller references.
    return json.loads(canonical({'schema':1,'stage':next(iter(stages)),
        'source':source_pointer,'members':member_hparams,'upstream':upstream or {}}))


def resolve_source_pointer(store,pointer,*,fingerprints=None):
    if fingerprints is None: fingerprints = _SOURCE_FINGERPRINTS
    mother,_=_load_mother(pointer['mother_record'],pointer['mother_record_sha256'])
    _check_source_recipe(mother,pointer['hparams'])
    return verify_source_context(store,pointer['dataset_id'],pointer['hparams'],mother.scdinfo,
                                 source_root=pointer['source_root'],fingerprints=fingerprints)


def build_stage_contexts(store,task):
    """Top-level factory suitable for stage_processes.run_tasks."""
    if task.get('schema')!=1:raise ValueError('Unsupported task schema')
    stage=task['stage'];members=task['members'];pointer=task['source'];upstream=task['upstream']
    if not members or any(hp.get('step')!=stage for hp in members):raise ValueError('Member stages differ')
    source=resolve_source_pointer(store,pointer)
    if stage==1:
        if upstream:raise ValueError('Step 1 has no upstream models')
        contexts=[ArtifactStep1Context(source,hp) for hp in members]
    elif stage==2:
        if set(upstream)!={'encoder_ids'} or len(upstream['encoder_ids'])!=len(members):
            raise ValueError('Step 2 needs one explicit encoder per member')
        contexts=[]
        for hp,key in zip(members,upstream['encoder_ids']):
            base=store.read(key,'model')['identity']['training_recipe']['hparams']
            if base.get('model_seed')!=hp['model_seed']:raise ValueError('Encoder/member seed pairing differs')
            contexts.append(ArtifactStep2Context(store,key,source,source.dataset_id,hp))
    elif stage==3:
        required={'region_id','base_X2_scores'}
        if not required<=set(upstream) or set(upstream)-required-{'smeared_X2_scores','representation_encoder_id'}:
            raise ValueError('Step 3 needs a frozen region and explicit X2 scores')
        contexts=[]
        for hp in members:
            args=(store,upstream['region_id'],source,upstream['base_X2_scores'],upstream.get('smeared_X2_scores'),hp)
            if hp['model']=='AttentionClassifier':
                if 'representation_encoder_id' not in upstream:raise ValueError('Representation encoder must be explicit')
                contexts.append(ArtifactStep3RepresentationContext(*args,upstream['representation_encoder_id']))
            else:
                if 'representation_encoder_id' in upstream:raise ValueError('Raw-feature task has an unexpected representation encoder')
                contexts.append(ArtifactStep3Context(*args))
    else:raise ValueError('Unsupported stage')
    if len({context.hash for context in contexts})!=len(contexts):raise ValueError('Duplicate member identities')
    return contexts,source
