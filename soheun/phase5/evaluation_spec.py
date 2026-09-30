"""Verify frozen bootstrap implementation pins without loading data or running tests."""
from copy import deepcopy
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def verified_spec(spec):
    value=deepcopy(spec)
    for name,expected in value['implementation_sha256'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Bootstrap implementation differs from audited source: {name}')
    if (value['bootstrap_replicates']!=1000 or value['alpha']!=.05
            or value['clipping']!={'selection':'raw log_psi >= log_tau_s before clipping',
                                  'lower':'log_tau_s; no lower clipping','upper':10.0,
                                  'score':'minimum(raw log_psi, 10.0)','applies_to':'every eta, including infinity'}):
        raise ValueError('Evaluation recipe differs from reviewed manuscript convention')
    value['input_adapter_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('artifacts/affine_inputs.py','artifacts/campaign_reader.py')}
    return value
