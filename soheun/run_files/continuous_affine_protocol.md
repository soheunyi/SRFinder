# Exact continuous affine-nuisance inversion

This evaluation uses the user-supplied `affine_weighted_ks_reference.py`
UNCHANGED. `affine_weighted_ks_compiled.py` loads a private copy and replaces
only its upper-envelope stack loop with a literal C++ long-double translation.
Sorting, duplicate-slope reduction, interval intersections, closed-endpoint
overlap counting, Poisson draws, tolerances and p-value calculation remain in
the supplied Python implementation. No nuisance grid is introduced.

Compilation uses -O2 -fno-fast-math -ffp-contract=off. ABI size and long-double
precision are checked at import. Reference and adapter hashes are saved in
every checkpoint. `test_affine_ks_compiled.py` checks exact equality of 500
envelopes and 100 complete result objects. The real-data equivalence benchmark
checks every returned field on one saved null dataset with the same draws.

Fixed scientific scope: eta=1, SR fraction=.2, epsilon=0,.005,.0075,.01,.02,
100 existing seeds per ratio (500 tests). Original SR membership is selected
before clipping scores to [-10,10]. The previously audited training-defined
lower support bound is reused by hash; full saved event weights are passed
once. B=1000, alpha=.05, RNG seed=1729, numerical_tol=1e-12.

The bootstrap statistic at each t is the ACTUAL centered process norm
||t Gplus+(1-t)Gminus-G4||, not a triangle bound. The reported p-value maximizes
fixed-t bootstrap tail probabilities continuously over t in [0,1]. The
bootstrap assignments use the reference's sorted per-class event order.
Thus old band checkpoints share datasets but not necessarily event-level
multiplier assignments, even if the integer RNG seed is the same.

Theoretical guarantee remains conditional and asymptotic under independent
event records, appropriate weight moments, positive endpoint denominators,
and an adequate nonnegative affine background-correction family. Physical
absence of signal does not itself guarantee the affine-family null. A smaller
p-value than the band methods is not proof of better calibration; null and
power results must both be examined. This uses previously inspected datasets
and is a development comparison, not independent confirmation.

No production tests, figures, or previous calibration results are overwritten.
Every experiment writes an atomic per-hash checkpoint and is skipped on resume
only when its version and bootstrap count match. Final aggregation requires
all 500 expected hashes, 100 per signal ratio, and checks numerical metadata.
