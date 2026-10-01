"""Raw source verification and seed-reconstructed outer split without cache writes."""
import hashlib
import pathlib
import sys
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore,canonical
from artifacts.source_context import verify_source_context
from dataset import SCDatasetInfo,MotherSamples
from step_1_preprocessing import get_step_1_tinfo


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Changed source was accepted')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp)
        path=root/'pool.h5';pd.DataFrame({'x':np.arange(8),'weight':1.}).to_hdf(path,key='df')
        mask=np.array([True,False,True,True,False,True,True,False])
        source=SCDatasetInfo([path],[mask])
        params={'n_3b':3,'ratio_4b':.5,'signal_ratio':0.,'signal_filename':'pool.h5','seed':7,'base_fvt_train_ratio':.6}
        hp={'model':'FvTClassifier','dataset':params,'step':1,'val_ratio':.33,'data_seed':0,'train_seed':0}
        desc={'pools':[{'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],
              'mother_parameters':params,'mother_selection_sha256':[hashlib.sha256(mask.tobytes()).hexdigest()]}
        store=TrainingStore(root/'store')
        dataset=store.put_dataset(hashlib.sha256(canonical(desc)).hexdigest(),len(source),desc)
        state=np.random.get_state()
        verified=verify_source_context(store,dataset,hp,source,source_root=root)
        after=np.random.get_state();assert state[0]==after[0] and state[2:]==after[2:]
        np.testing.assert_array_equal(state[1],after[1])
        mother=MotherSamples(source,'fixture',params)
        with patch.object(MotherSamples,'find',return_value=['fixture']),patch.object(MotherSamples,'load',return_value=mother):
            legacy=get_step_1_tinfo(hp)
            np.testing.assert_array_equal(verified.ms_idx,legacy.ms_idx)
            pd.testing.assert_frame_equal(verified.scdinfo.fetch_data(),legacy.scdinfo.fetch_data())
        assert sorted(np.r_[verified.indices('X1'),verified.indices('X2')].tolist())==list(range(len(source)))
        changed=mask.copy();changed[1]=True;changed[2]=False
        rejects(lambda:verify_source_context(store,dataset,hp,SCDatasetInfo([path],[changed]),source_root=root))
        changed_hp={**hp,'dataset':{**params,'seed':8}}
        rejects(lambda:verify_source_context(store,dataset,changed_hp,source,source_root=root))
        with path.open('ab') as f:f.write(b'changed')
        rejects(lambda:verify_source_context(store,dataset,hp,source,source_root=root))
    print('PASS: raw pool and mother-mask verification, native X1/X2 reconstruction, caller RNG preservation and changed-source rejection')


if __name__=='__main__':main()
