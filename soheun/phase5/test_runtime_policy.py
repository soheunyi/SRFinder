"""Check process-global policy scoping without allocating a CUDA tensor."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from artifacts.runtime_policy import numerical_state,runtime_policy,validated_gpu_runtime


@validated_gpu_runtime
def probe(*,device='cpu',fail=False):
    state=numerical_state()
    if fail:raise RuntimeError('fixture')
    return state


def main():
    original=numerical_state()
    try:
        for precision in ('highest','high','medium'):
            torch.set_float32_matmul_precision(precision)
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=True
            caller=numerical_state()
            assert probe(device='cpu')==caller
            result=probe(device='cuda')
            assert result['matmul_precision']=='medium' and result['cuda_matmul_tf32']
            assert result['cudnn_tf32'] and not result['cudnn_benchmark']
            assert numerical_state()==caller
            with runtime_policy('cuda'):
                outer=numerical_state();assert probe(device='cuda')==outer
                assert numerical_state()==outer
            assert numerical_state()==caller
            try:probe(device='cuda',fail=True)
            except RuntimeError:pass
            else:raise AssertionError('Exception swallowed')
            assert numerical_state()==caller
    finally:
        torch.backends.cuda.matmul.allow_tf32=original['cuda_matmul_tf32']
        torch.set_float32_matmul_precision(original['matmul_precision'])
        torch.backends.cudnn.allow_tf32=original['cudnn_tf32']
        torch.backends.cudnn.benchmark=original['cudnn_benchmark']
    assert numerical_state()==original
    print('PASS: validated GPU policy, CPU preservation, nesting and exception-safe exact restoration')


if __name__=='__main__':main()
