# Focused linear-correction calibration diagnostic

Scope fixed before inspection of outcomes: eta=1.0, SR fraction=0.20,
100 existing seeds for each of epsilon=0,.005,.0075,.01,.02 (500 tests).
No neural training. B=1000; alpha=.05; original SR membership; score cap [-10,10].

For the training-defined score support [L,U] (L is the clipped SR cutoff,
U=10), any affine nonnegative weight tilt is a nonnegative combination
of s-L and U-s. Its normalized weighted CDF is therefore a convex mixture
of the two endpoint-tilted CDFs A and B. Compute
T=inf_{t in [0,1]} ||t*Ahat+(1-t)*Bhat-F4hat||_infinity.
Evaluate CDFs only after combining all jumps at tied scores.

Independently draw Poisson(1)-1 multipliers per ORIGINAL event in each class.
Use the same 3b multiplier for its contributions to both endpoint CDFs.
For each endpoint, use the centered ratio-estimator influence process
sum u_i normalized_weight_i (1{s_i<=y}-Fhat(y)).
For each replicate record max(||ZA-Z4||,||ZB-Z4||). This equals the supremum
of the joint centered fluctuation over every convex mixture. Compare T to
this simultaneous critical value; report (1+#bootstrap>=T)/(B+1).
This plus-one convention does not make a multiplier bootstrap finite-sample exact.

Reason for conservative validity: if a true affine correction exists,
there is a population mixture t0. The minimized observed discrepancy is at
most the error at t0, which is bounded by the simultaneous endpoint band.
Joint multiplier bootstrap approximation requires independent events,
appropriate moment/no-dominating-weight conditions and fixed learned maps.
Population endpoint mixture coefficients and sample coefficients need not
agree; the existence-of-mixture confidence-set argument does not plug in a
random estimated slope as if it were known.

This changes the affine fitting objective from mean CDF difference to minimax
CDF difference and imposes positivity on the training-defined clipped support.
It retains a linear correction, with no quadratic term. The domain restriction
must be reported in comparing methods.

LIMIT: physical absence of signal alone does not establish existence of the
affine correction for a noisy CR density-ratio estimate. Reusing these 100 seeds
is a development diagnostic, not independent confirmation. Never choose alpha
or band width after examining rejection counts. Assess power alongside null
rejection; do not claim calibration merely because all p-values are large.

Reference for the general empirical-process multiplier-bootstrap framework:
Chernozhukov, Chetverikov, Kato (2016), https://arxiv.org/abs/1502.00352.
Application-specific assumptions above still need justification.

## Coverage argument

Let A,B be the population CDFs obtained from the two endpoint tilts of the
reweighted 3b distribution, and C the population 4b CDF. Let E_A=Ahat-A,
E_B=Bhat-B and E_C=Chat-C. Under the affine composite null, C=t0*A+(1-t0)*B
for some fixed population t0 in [0,1]. Then, deterministically,

    inf_t ||t*Ahat+(1-t)*Bhat-Chat||
      <= ||t0*E_A+(1-t0)*E_B-E_C||
      <= max(||E_A-E_C||, ||E_B-E_C||).

The multiplier bootstrap approximates the distribution of the last expression.
If that approximation provides 1-alpha simultaneous coverage, the rejection
probability is at most alpha (asymptotically, under the stated assumptions).
This argument explains why no fitted correction is reapplied or refitted in
the bootstrap; the bootstrap estimates an envelope of empirical errors rather
than the sampling distribution of a plug-in optimized KS statistic.

The true mixture coefficient is not estimated in this coverage bound. The
observed minimization is continuous (bounded scalar convex optimization),
including both endpoints, with a 1e-10 numerical allowance in the comparison.

## Runtime check

One representative real null dataset, 100 identical seeded bootstrap draws:
batch sizes 1/4/8/16 took 1.62/2.63/2.98/4.40 seconds respectively on Wright.
Default is therefore batch size 1. It retains NumPy vectorization over events
and shares sorted indices, endpoint weights and CDFs across replicates. Larger
batches remain available and have been checked for numerical equivalence.
Independent datasets run concurrently; no inference on neural networks is done.

## Evaluation interpretation

Report 100-seed binomial intervals and paired rejection changes. The 400 signal
cases guard against a vacuous always-accept test. Do not aggregate these four
signal ratios into a single claim of detection probability. These same seeds
have already been inspected in prior diagnostics, so the result is development
evidence. A successful focused experiment is not a guarantee for other eta/SR
settings or for arbitrary background-model misspecification.

## Separately versioned interval refinement

The global band may be unnecessarily wide when its largest fluctuations occur
at an extreme correction far from any plausible observed fit. In the separate
linear_interval_eta1_sr020_v1 diagnostic, partition t in [0,1] into 64 fixed
equal intervals, chosen without inspecting interval-test rejection rates.
For interval I=[l,r], let T_I=inf_{t in I} D(t), with D the same weighted KS
discrepancy. For each multiplier draw compute endpoint norms M_A and M_B as
in the global method and define

    Q_I = max(l*M_A+(1-l)*M_B, r*M_A+(1-r)*M_B).

Triangle inequality bounds every error norm at t in I by this quantity.
Compute each interval's tail probability using Q_I and T_I; the composite
p-value is the MAXIMUM of all interval p-values. Reject only if every interval
is rejected. If a true correction t0 exists, its fixed interval I0 cannot be
rejected more often than its bootstrap confidence bound fails. Taking the
intersection of rejections therefore needs no multiple-testing penalty.

This is a conservative confidence-set inversion, not an attempt to pick the
interval with the best significance. It has the same conditional asymptotic
validity assumptions as the global method. It is never more conservative than
the global band for the same draws, but may still be conservative relative to
a pointwise ideal bootstrap. Both methods retain precisely the same affine
family; only the uncertainty bound is refined. Power and null counts are
reported separately, and no outcomes are used to tune alpha or the partition.
