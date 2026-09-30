"""Paired score provenance and exact compatibility with current region selection."""
import hashlib
import pathlib
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.training_store import TrainingStore,canonical
from artifacts.regions import derive_region_scores,define_regions,classify_X2
from signal_region import get_SR_CR_cut


class Events(SimpleNamespace):
    def __len__(self): return len(self.weights)


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid region input accepted')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp);store=TrainingStore(root/'store')
        dataset=store.put_dataset(hashlib.sha256(b'fixture').hexdigest(),32,{'version':1})
        splits=[store.put_split(dataset,name,np.arange(offset,offset+16,dtype=np.int64),
                  {'algorithm':'fixture','version':1,'seed':0}) for name,offset in [('X1',0),('X2',16)]]
        path=root/'weights';path.write_bytes(b'opaque fixture weights')
        bases=[];smooth=[]
        for m in range(2):
            base=store.put_model(path,{'member':m},{'hparams':{'step':1,'model':'FvTClassifier'}},[splits[0]])
            smear=store.put_model(path,{'member':m},{'hparams':{'step':2,'model':'AttentionClassifier','encoder_hash':'artifact:'+base}},[splits[0]])
            bases.append(base);smooth.append(smear)
        bp=[];sp=[];expected=[]
        for s,split in enumerate(splits):
            bs=[np.linspace(-1,2,16,dtype=np.float32),np.linspace(-.5,1.5,16,dtype=np.float32)]
            ss=[np.linspace(0,.2,16,dtype=np.float32),np.full(16,-.2,dtype=np.float32)]
            if s:bs=[x[::-1].copy() for x in bs];ss=[x[::-1].copy() for x in ss]
            bp.append([store.put_scores(owner,split,v,'log_density_ratio') for owner,v in zip(bases,bs)])
            sp.append([store.put_scores(owner,split,v,'log_density_ratio') for owner,v in zip(smooth,ss)])
            expected.append(np.max([a-b for a,b in zip(bs,ss)],axis=0))
        before=store.storage_stats()
        values,_=derive_region_scores(store,bp[0],sp[0]);np.testing.assert_array_equal(values,expected[0])
        assert store.storage_stats()==before
        metadata=np.empty(16,dtype=[('is_4b','?'),('weight','<f8')]);metadata['is_4b']=np.arange(16)%2;metadata['weight']=1.;metadata['weight'][3]=-.1
        mid=store.put_array('event_metadata',{'dataset_id':dataset,'split_id':splits[0]},metadata)
        region=define_regions(store,bp[0],mid,sp[0],sr_fraction=.25,cr_fraction=.75)
        cuts=get_SR_CR_cut(expected[0],Events(weights=metadata['weight'],is_4b=metadata['is_4b']),{'4b_in_SR':.25,'4b_in_CR':.75})
        assert store.read(region)['payload']['log_tau_s']==float(cuts[0])
        before=store.storage_stats();groups=classify_X2(store,region,bp[1],sp[1]);assert store.storage_stats()==before
        np.testing.assert_array_equal(groups['SR'],expected[1]>=cuts[0])
        np.testing.assert_array_equal(groups['CR'],expected[1]<cuts[0])
        assert not groups['neither'].any()
        assert store.read(region)['payload']['log_tau_c'] is None
        assert store.read(region)['identity']['cr_lower_bound']=='none'
        assert b'"log_tau_c":null' in canonical(store.read(region))
        old=define_regions(store,bp[0],mid,sp[0],sr_fraction=.25,cr_fraction=.75,
                           quantile_recipe='existing_get_SR_CR_cut_v1')
        assert old!=region and store.read(old)['payload']['log_tau_c']==float(cuts[1])
        legacy=classify_X2(store,old,bp[1],sp[1])
        np.testing.assert_array_equal(legacy['CR'],(expected[1]>=cuts[1])&(expected[1]<cuts[0]))
        assert legacy['neither'].any()
        partial=define_regions(store,bp[0],mid,sp[0],sr_fraction=.25,cr_fraction=.5)
        partial_cuts=get_SR_CR_cut(expected[0],Events(weights=metadata['weight'],is_4b=metadata['is_4b']),{'4b_in_SR':.25,'4b_in_CR':.5})
        assert store.read(partial)['payload']['log_tau_c']==float(partial_cuts[1])
        assert store.read(partial)['identity']['cr_lower_bound']=='finite'
        positive=metadata.copy();positive['weight']=1.
        positive_id=store.put_array('event_metadata',{'dataset_id':dataset,'split_id':splits[0],'fixture':'positive'},positive)
        balanced=define_regions(store,bp[0],positive_id,sp[0],sr_fraction=.25,cr_fraction=.75)
        threshold=store.read(balanced)['payload']['log_tau_s']
        x1cr=expected[0]<threshold
        assert positive['weight'][positive['is_4b'] & x1cr].sum()/positive['weight'][positive['is_4b']].sum()==.75
        assert np.all(x1cr | (expected[0]>=threshold))
        assert np.all(groups['SR'].astype(int)+groups['CR']+groups['neither']==1)
        rejects(lambda:derive_region_scores(store,bp[0],sp[0][::-1]))
        rejects(lambda:define_regions(store,bp[1],mid,sp[1],sr_fraction=.25,cr_fraction=.75))
        rejects(lambda:classify_X2(store,region,bp[0],sp[0]))
    print('PASS: complete complement CR, unchanged SR cuts, finite legacy/partial reads, canonical null and positive X1 weight fraction')


if __name__=='__main__':main()
