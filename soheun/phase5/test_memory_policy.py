"""Budget boundaries and fallback must not imply model/batch changes."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from artifacts.memory_policy import choose_placement,worker_budgets


def main():
    budgets=worker_budgets(10000,safety_bytes=1000)
    assert budgets==[1800]*5 and sum(budgets)+1000<=10000
    base=dict(device='cuda',data_bytes=1500,budget_bytes=1800,headroom_bytes=300,available_bytes=9000)
    assert choose_placement('auto',**base)['placement']=='resident'
    assert choose_placement('auto',**{**base,'data_bytes':1501})['placement']=='cpu'
    assert choose_placement('auto',**{**base,'available_bytes':1799})['placement']=='cpu'
    assert choose_placement('cpu',**base)['placement']=='cpu'
    assert choose_placement('auto',device='cpu',data_bytes=1500)['placement']=='cpu'
    cases=[(MemoryError,lambda:choose_placement('resident',**{**base,'data_bytes':1501})),
           (MemoryError,lambda:choose_placement('auto',**{**base,'headroom_bytes':1801})),
           (ValueError,lambda:choose_placement('auto',device='cuda',data_bytes=10)),
           (ValueError,lambda:worker_budgets(10000,workers=0,safety_bytes=1000)),
           (MemoryError,lambda:worker_budgets(10000,safety_bytes=10000))]
    for exception,call in cases:
        try:call()
        except exception:pass
        else:raise AssertionError('Invalid memory admission accepted')
    assert set(choose_placement('auto',**base))=={'requested','data_bytes','budget_bytes','headroom_bytes','available_bytes','placement','reason'}
    print('PASS: disjoint five-worker budgets, exact-fit admission, staged fallback and impossible-budget rejection')


if __name__=='__main__':main()
