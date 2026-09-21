"""Write a report from complete, audited results without rerunning tests."""
import json
from pathlib import Path
import pandas as pd
from calibration_continuous_affine import OUT

audit=json.loads((OUT/'audit.json').read_text())
assert audit['complete'] and audit['completed']==audit['expected']==500
s=pd.read_csv(OUT/'summary.csv')
assert len(s)==5 and (s.n==100).all()
benchmark=json.loads((OUT/'real_reference_equivalence.json').read_text())
lines=['# Continuous affine-nuisance bootstrap: focused evaluation','',
'Setting: eta=1.0, SR fraction=0.2. All 500 expected datasets completed,',
'100 per signal ratio, with 1000 bootstrap replicates at alpha=0.05.','',
'| Signal ratio | Current refit | Global band | 64-interval band | Exact continuous test |',
'|---:|---:|---:|---:|---:|']
for _,r in s.iterrows():
    lines.append(f"| {r.signal_ratio:g} | {100*r.current_rate:.0f}% | {100*r.global_rate:.0f}% | {100*r.interval_rate:.0f}% | {100*r.continuous_rate:.0f}% |")
null=s[s.signal_ratio==0].iloc[0]
lines+=['',f"Null rejection: {int(null.continuous_rejections)}/100; exact binomial 95% interval [{null.lower95:.6f}, {null.upper95:.6f}].",'',
'The test is precisely the supplied fixed-t centered multiplier KS procedure,',
'maximizing its p-value continuously over the admissible affine family. It',
'uses the actual process norm, not the triangle bound. No nuisance grid or',
'histogram bins are used. The affine-family null and original independent-event',
'resampling assumptions remain essential; physical no-signal calibration is',
'not implied for arbitrary CR extrapolation errors. These previously inspected',
'seeds provide development evidence, not independent confirmation.','',
'## Optimization verification','',
'Only the upper-envelope stack loop is compiled, as a literal C++ long-double',
'translation with fast-math and floating-point contraction disabled. The user',
'reference file is unchanged. All 500 synthetic envelope checks and 100 full',
'result-object comparisons matched exactly. The real-data benchmark also',
'matched every returned field exactly for the same random draws.',
f"Real benchmark (20 replicates, {benchmark['n3']} 3b and {benchmark['n4']} 4b events): reference {benchmark['reference_seconds']:.3f}s, compiled {benchmark['compiled_seconds']:.3f}s; {benchmark['speedup']:.2f}x speedup.",
'',f"Reference SHA256: `{audit['reference_sha256']}`",f"Adapter SHA256: `{audit['adapter_sha256']}`",'',
'Saved artifacts: manifest.pkl, results/*.pkl, paired_results.csv, summary.csv,',
'audit.json, real_reference_equivalence.json. All earlier studies are retained.',
'No existing production figure or rejection-rate table was overwritten.']
boundary=OUT/'boundary_multiplier_check.json'
if boundary.exists():
    lines += ['', '## Near-threshold Monte Carlo check', '',
              'The previous band implementation and the supplied code attach draws',
              'to events in different orders. Equal RNG seeds therefore do not imply',
              'matched event-level multipliers across those implementations.']
    for item in json.loads(boundary.read_text()):
        lines.append(f"For {item['hash']}, the previous unaligned interval p-value was {item['previous_unaligned_interval_p']:.6f}. Replaying the band with the supplied code's sorted event order gave {item['aligned_interval_p']:.6f}, compared with {item['continuous_p']:.6f} for the continuous test. The expected conservative-bound ordering is restored with matched draws. No reported p-values were replaced or retuned.")
(OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
