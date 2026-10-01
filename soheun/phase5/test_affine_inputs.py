"""Compare the new-score adapter against the existing signed bootstrap reference."""
import pathlib
import sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'run_files')]
from artifacts.affine_inputs import prepare_affine_inputs
from affine_weighted_ks_signed_reference import affine_ks_test


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid input accepted')


def main():
    keys=np.array([[0,1],[0,2],[0,3],[1,1],[1,2],[2,1],[0,4]],dtype=np.int64)
    labels=np.array([0,0,0,1,1,1,0],dtype=bool)
    weights=np.array([1.,2.,3.,2.,1.,-.2,1.])
    psi=np.array([.2,.7,3.,.4,1.2,.8,-1.],dtype=np.float32)
    gamma=np.array([.1,.2,.3,0.,0.,0.,0.],dtype=np.float32)
    def prepare(k=keys,y=labels,w=weights,s=psi,g=gamma,domain='X2'):
        return prepare_affine_inputs(k,y,w,s,g,dataset_id='synthetic-version-1',domain=domain,lower=0.,upper=2.)
    data,audit=prepare()
    baseline=(np.minimum(psi[:3].astype(float),2.),np.exp(gamma[:3].astype(float))*weights[:3],
              psi[3:6].astype(float),weights[3:6])
    a=affine_ks_test(*data,L=0.,U=2.,B=31,seed=1729)
    b=affine_ks_test(*baseline,L=0.,U=2.,B=31,seed=1729)
    assert a.p_value==b.p_value and a.max_exceedances==b.max_exceedances
    np.testing.assert_allclose(a.ks_statistic,b.ks_statistic,rtol=0,atol=1e-12)
    perm=np.array([6,3,2,5,1,4,0])
    changed,other=prepare(keys[perm],labels[perm],weights[perm],psi[perm],gamma[perm])
    assert other==audit
    for x,y in zip(data,changed):np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(data[3],weights[3:6])
    assert audit['negative_4b_count']==1 and audit['clipped_count_3b']==1
    huge=gamma.copy();huge[:3]+=1000.
    stable,extreme=prepare(g=huge)
    assert all(np.isfinite(x).all() for x in stable)
    assert extreme['log_q_common_scale']>1000
    rejects(lambda:prepare(domain='X1'))
    duplicate=keys.copy();duplicate[1]=duplicate[0]
    rejects(lambda:prepare(k=duplicate))
    negative=weights.copy();negative[0]=-1.
    rejects(lambda:prepare(w=negative))
    negative4=weights.copy();negative4[3:6]=[-2.,1.,-.2]
    rejects(lambda:prepare(w=negative4))
    boundary=psi.copy();boundary[:6]=2.
    rejects(lambda:prepare(s=boundary))
    print('PASS: signed bootstrap agreement, canonical order, upper-only clipping, stable log weights and endpoint checks')


if __name__=='__main__':main()
