"""Produce the focused comparison only after both complete checkpoint audits."""
import subprocess
import sys
from pathlib import Path
import pandas as pd

repo=Path(__file__).resolve().parents[1]
scripts=repo/'run_files'
subprocess.run([sys.executable,str(scripts/'audit_linear_calibration.py')],check=True)
for extra in ([],['--interval']):
    subprocess.run([sys.executable,str(scripts/'summarize_linear_band.py'),*extra],check=True)
root=repo/'data/refit_bootstrap'
g=pd.read_csv(root/'linear_band_eta1_sr020_v1/paired_results.csv')
i=pd.read_csv(root/'linear_interval_eta1_sr020_v1/paired_results.csv')
assert len(g)==len(i)==500
assert set(g['hash'])==set(i['hash'])
assert (g.groupby('signal_ratio').size()==100).all()
assert (i.groupby('signal_ratio').size()==100).all()
counts=[]
for ratio in sorted(i.signal_ratio.unique()):
    a=g[g.signal_ratio==ratio]; b=i[i.signal_ratio==ratio]
    counts.append({'signal_ratio':ratio,'n':len(a),'current_rejections':int(a.new_reject.sum()),'global_band_rejections':int(a.band_reject.sum()),'interval_band_rejections':int(b.band_reject.sum())})
table=pd.DataFrame(counts)
destination=root/'linear_interval_eta1_sr020_v1'
table.to_csv(destination/'final_comparison.csv',index=False)
lines=['# Focused linear-bootstrap calibration comparison','',
'Setting: eta=1.0, SR fraction=0.20. Each row uses the same 100 saved seeds.',
'Each new test uses 1000 Poisson multiplier replicates and alpha=0.05.',
'No neural network was retrained. The affine family has no quadratic term.','',
'| Signal ratio | Trials | Current refit rejects | Global-band rejects | Interval-band rejects |',
'|---:|---:|---:|---:|---:|']
for r in counts:
    lines.append(f"| {r['signal_ratio']:g} | {r['n']} | {r['current_rejections']} | {r['global_band_rejections']} | {r['interval_band_rejections']} |")
lines += ['',
'## Conclusion for this focused development study',
'',
'Prefer the interval-band candidate to the global envelope: both reject 2/100',
'null datasets, but the interval version retains 74/100 and 99/100 rejections',
'at signal ratios .01 and .02. The current refit procedure gives 11/100 null',
'rejections and 86/100 and 100/100 at those alternatives. The interval method',
'does lose weak-signal sensitivity: 5/100 at both .005 and .0075, versus 9/100',
'and 26/100 currently. This is conservative calibration evidence, not a claim',
'of unchanged power. The exact 95% binomial interval for 2/100 is approximately',
'[0.00243, 0.07038]; that uncertainty prevents an empirical proof of a 5% bound.',
'',
'Both result sets cover all 500 expected hashes, 100 per signal ratio. All 500',
'paired inputs/statistics/shared bootstrap draws agree between the two methods.',
'Fifty unfinished global-reference outputs were recovered from the identical',
'global calculations embedded in completed interval files, with derived_from',
'provenance; no statistical result was fabricated or imputed. The redundant',
'global job was stopped only after every interval result was saved. No null',
'events were clipped in this particular setting. Existing production test',
'results and manuscript figures have not been replaced by this diagnostic.',
'',
'The new methods test compatibility with some nonnegative affine correction on',
'the training-defined, clipped SR score support. They minimize a tie-aware KS',
'distance rather than the previous mean-CDF fitting objective. Independent',
'Poisson(1)-1 multipliers retain each class\'s original score-weight pairs.',
'The global method uses a simultaneous endpoint envelope; the interval method',
'inverts 64 fixed local envelopes and takes the maximum interval p-value.',
'The interval refinement has the same affine family as the global method.','',
'Validity is conditional and asymptotic: it requires an adequate affine',
'background correction, appropriate weighted empirical-process regularity,',
'and independent events with learned maps treated as fixed. The plus-one',
'p-value convention does not make this finite-sample exact. Physical absence',
'of signal alone does not guarantee absence of CR extrapolation bias.','',
'This is a development comparison using previously inspected seeds. It is',
'evidence about this specific eta/SR setting, not an independent confirmation',
'or a guarantee at other settings. No alpha or interval-width tuning was used',
'to force a desired observed rejection count.','',
'See run_files/linear_band_protocol.md for the coverage argument and validation.',
'See audit.json, summary.csv, paired_results.csv and results/*.pkl in both',
'method directories for coverage checks, binomial intervals and raw outputs.']
(destination/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
print(table.to_string(index=False)); print(destination/'FINAL_REPORT.md')
