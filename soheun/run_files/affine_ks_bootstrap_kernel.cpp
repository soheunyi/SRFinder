// Bootstrap replicates for the affine-nuisance weighted KS test
// (run_files/affine_weighted_ks_reference.py), moved out of the Python loop.
//
// The reference does, for each of B replicates:
//   1. draw Poisson(1) multipliers xi for every 3b and 4b event (centred: xi - 1),
//   2. form the multiplier processes g+, g-, g4 at every distinct score y_k
//      (weighted cumulative sums minus the observed CDF times the total),
//   3. build the upper envelope on t in [0, 1] of the 2m lines  +-(a_k + d_k t),
//      a_k = g-_k - g4_k,  d_k = g+_k - g-_k,
//   4. intersect it with the observed envelope to get the t-intervals where the
//      replicate exceeds the observed statistic.
// Step 3 used argsort over all 2m slopes (64% of the time); here the envelope is
// traced directly from t = 0, which costs one pass over the lines per piece and
// the envelopes have only a few pieces.
//
// Multipliers: either supplied by the caller (testing: NumPy's stream, so the
// arithmetic can be compared exactly with the reference) or drawn here from an
// independent seeded stream per replicate (production: thread-count independent).
// Build: g++ -O2 -std=c++17 -fPIC -shared -fopenmp -fno-fast-math -ffp-contract=off
//        affine_ks_bootstrap_kernel.cpp -o affine_ks_bootstrap_kernel.so
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double LD;

struct Piece { LD left, right, slope, intercept; };

// ---------------------------------------------------------------- random numbers
// splitmix64 seeds xoshiro256**; each replicate gets its own stream derived from
// (seed, replicate index), so results do not depend on scheduling or thread count.
static inline std::uint64_t splitmix64(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Xoshiro256 {
    std::uint64_t s[4];
    explicit Xoshiro256(std::uint64_t seed, std::uint64_t stream) {
        std::uint64_t x = seed ^ (0xD1B54A32D192ED03ULL * (stream + 1));
        for (auto& v : s) v = splitmix64(x);
    }
    static inline std::uint64_t rotl(std::uint64_t v, int k) { return (v << k) | (v >> (64 - k)); }
    inline std::uint64_t next() {
        const std::uint64_t result = rotl(s[1] * 5, 7) * 9;
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return result;
    }
    // Uniform double in [0, 1) with 53 random bits.
    inline double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

// Poisson(1) by inversion: P(K <= k) = e^-1 * sum_{i<=k} 1/i!. Mean 1, variance 1.
struct PoissonOne {
    double cdf[20];
    PoissonOne() {
        double term = std::exp(-1.0), total = 0.0;
        for (int k = 0; k < 20; ++k) { total += term; cdf[k] = total; term /= (k + 1); }
        cdf[19] = 1.0;  // the remaining tail mass is below 1e-17
    }
    inline int draw(Xoshiro256& rng) const {
        const double u = rng.uniform();
        int k = 0;
        while (u >= cdf[k]) ++k;
        return k;
    }
};

// ---------------------------------------------------------------- envelope
// Upper envelope on [0, 1] of the lines t -> s_j t + a_j for j < 2m, where line j
// is (+d_j, +a_j) for j < m and (-d, -a) of j - m otherwise.  Traced from t = 0:
// start with the highest line at 0 (ties: the larger slope, which wins just right
// of 0), then repeatedly jump to the line that overtakes the current one first.
// Breakpoints use the reference's expression (a_cur - a_j) / (s_j - s_cur), with
// the lower-slope line first, in long double, as in affine_envelope_kernel.cpp.
static void trace_envelope(std::size_t m, const double* a, const double* d, std::vector<Piece>& out) {
    out.clear();
    const std::size_t n = 2 * m;
    auto A = [&](std::size_t j) -> LD { return j < m ? static_cast<LD>(a[j]) : static_cast<LD>(-a[j - m]); };
    auto S = [&](std::size_t j) -> LD { return j < m ? static_cast<LD>(d[j]) : static_cast<LD>(-d[j - m]); };
    std::size_t cur = 0;
    for (std::size_t j = 1; j < n; ++j) {
        const LD aj = A(j), ac = A(cur);
        if (aj > ac || (aj == ac && S(j) > S(cur))) cur = j;
    }
    LD t = 0;
    while (true) {
        const LD ac = A(cur), sc = S(cur);
        LD best_t = std::numeric_limits<LD>::infinity();
        std::size_t best = n;
        for (std::size_t j = 0; j < n; ++j) {
            const LD sj = S(j);
            if (!(sj > sc)) continue;                 // only steeper lines can overtake
            LD x = (ac - A(j)) / (sj - sc);
            if (x < t) x = t;                         // rounding: already on top at t
            if (x < best_t || (x == best_t && sj > S(best))) { best_t = x; best = j; }
        }
        if (best == n || best_t >= 1) { out.push_back({t, 1, sc, ac}); return; }
        if (best_t > t) out.push_back({t, best_t, sc, ac});
        cur = best; t = best_t;
    }
}

// ---------------------------------------------------------------- exceedances
// Literal port of _exceedance_intervals: closed t-intervals where
// Q_boot(t) + tol >= D_observed(t), merged within the replicate.
static void exceedances(const Piece* o, std::size_t no, const std::vector<Piece>& b, LD tol,
                        std::vector<std::pair<LD, LD>>& out) {
    out.clear();
    std::size_t i = 0, j = 0;
    while (i < no && j < b.size()) {
        const LD lo = o[i].left > b[j].left ? o[i].left : b[j].left;
        const LD hi = o[i].right < b[j].right ? o[i].right : b[j].right;
        const LD d = b[j].slope - o[i].slope;
        const LD a = b[j].intercept - o[i].intercept + tol;
        if (lo <= hi) {
            const LD flo = a + d * lo, fhi = a + d * hi;
            bool have = false; LD left = 0, right = 0;
            if (flo >= 0 && fhi >= 0) { have = true; left = lo; right = hi; }
            else if (flo >= 0) {
                LD c = -a / d; if (c < lo) c = lo;
                have = true; left = lo; right = hi < c ? hi : c;
            } else if (fhi >= 0) {
                LD c = -a / d; if (c < lo) c = lo;
                have = true; left = hi < c ? hi : c; right = hi;
            }
            if (have) {
                if (!out.empty() && left <= out.back().second) {
                    if (right > out.back().second) out.back().second = right;
                } else out.push_back({left, right});
            }
        }
        const LD old_o = o[i].right, old_b = b[j].right;
        if (old_o <= old_b) ++i;
        if (old_b <= old_o) ++j;
    }
}

// ---------------------------------------------------------------- one replicate
struct Inputs {
    std::size_t n3, n4, m;
    const double *up, *um, *r, *fp, *fm, *f4;
    const std::int64_t *i3, *i4;
    const Piece* obs; std::size_t nobs; LD tol;
};

struct Work {
    std::vector<double> cp, cm, c4, a, d;
    std::vector<int> xi3, xi4;
    std::vector<Piece> env;
    std::vector<std::pair<LD, LD>> iv;
    explicit Work(const Inputs& in)
        : cp(in.n3 + 1), cm(in.n3 + 1), c4(in.n4 + 1), a(in.m), d(in.m), xi3(in.n3), xi4(in.n4) {}
};

// Centred multipliers xi - 1 must already be in w.xi3 / w.xi4.  Mirrors process():
// cumulative = r_[0, cumsum(xi * q)];  g = cumulative[idx] - f * cumulative[-1].
static void replicate(const Inputs& in, Work& w) {
    w.cp[0] = w.cm[0] = w.c4[0] = 0.0;
    for (std::size_t e = 0; e < in.n3; ++e) {
        const double x = static_cast<double>(w.xi3[e]);
        w.cp[e + 1] = w.cp[e] + x * in.up[e];
        w.cm[e + 1] = w.cm[e] + x * in.um[e];
    }
    for (std::size_t e = 0; e < in.n4; ++e)
        w.c4[e + 1] = w.c4[e] + static_cast<double>(w.xi4[e]) * in.r[e];
    const double tp = w.cp[in.n3], tm = w.cm[in.n3], t4 = w.c4[in.n4];
    for (std::size_t k = 0; k < in.m; ++k) {
        const double gp = w.cp[in.i3[k]] - in.fp[k] * tp;
        const double gm = w.cm[in.i3[k]] - in.fm[k] * tm;
        const double g4 = w.c4[in.i4[k]] - in.f4[k] * t4;
        w.a[k] = gm - g4;
        w.d[k] = gp - gm;
    }
    trace_envelope(in.m, w.a.data(), w.d.data(), w.env);
    exceedances(in.obs, in.nobs, w.env, in.tol, w.iv);
}

// ---------------------------------------------------------------- C interface
extern "C" int affine_ks_ld_mant_dig() { return LDBL_MANT_DIG; }

// Runs replicates [first, first + count).  Multipliers come from xi3_given /
// xi4_given (count x n3, count x n4, already centred) when non-null, otherwise
// from the seeded per-replicate streams.  Intervals for replicate q are written
// to out_lo/out_hi[q * cap ...] with out_n[q] entries.  Returns 0, or 1 if some
// replicate produced more than cap intervals (caller retries with a larger cap).
extern "C" int affine_ks_bootstrap(
    std::size_t n3, std::size_t n4, std::size_t m,
    const double* up, const double* um, const double* r,
    const std::int64_t* i3, const std::int64_t* i4,
    const double* fp, const double* fm, const double* f4,
    std::size_t nobs, const LD* ol, const LD* orr, const LD* os, const LD* oa,
    double tol, std::uint64_t seed, std::size_t first, std::size_t count,
    const std::int64_t* xi3_given, const std::int64_t* xi4_given, int threads,
    std::size_t cap, LD* out_lo, LD* out_hi, std::int64_t* out_n) {
    std::vector<Piece> obs(nobs);
    for (std::size_t i = 0; i < nobs; ++i) obs[i] = {ol[i], orr[i], os[i], oa[i]};
    const Inputs in{n3, n4, m, up, um, r, fp, fm, f4, i3, i4, obs.data(), nobs, static_cast<LD>(tol)};
    const PoissonOne pois;
    int overflow = 0;
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel reduction(|:overflow)
    {
        Work w(in);
#pragma omp for schedule(dynamic, 4)
        for (std::int64_t q = 0; q < static_cast<std::int64_t>(count); ++q) {
            if (xi3_given) {
                for (std::size_t e = 0; e < n3; ++e) w.xi3[e] = static_cast<int>(xi3_given[q * n3 + e]);
                for (std::size_t e = 0; e < n4; ++e) w.xi4[e] = static_cast<int>(xi4_given[q * n4 + e]);
            } else {
                Xoshiro256 rng(seed, first + static_cast<std::uint64_t>(q));
                for (std::size_t e = 0; e < n3; ++e) w.xi3[e] = pois.draw(rng) - 1;
                for (std::size_t e = 0; e < n4; ++e) w.xi4[e] = pois.draw(rng) - 1;
            }
            replicate(in, w);
            if (w.iv.size() > cap) { overflow = 1; out_n[q] = -1; continue; }
            out_n[q] = static_cast<std::int64_t>(w.iv.size());
            for (std::size_t v = 0; v < w.iv.size(); ++v) {
                out_lo[q * cap + v] = w.iv[v].first;
                out_hi[q * cap + v] = w.iv[v].second;
            }
        }
    }
    return overflow;
}

// Draws n Poisson(1) values from replicate stream `stream` (for testing the sampler).
extern "C" void affine_ks_poisson_sample(std::uint64_t seed, std::uint64_t stream, std::size_t n, std::int64_t* out) {
    const PoissonOne pois;
    Xoshiro256 rng(seed, stream);
    for (std::size_t i = 0; i < n; ++i) out[i] = pois.draw(rng);
}
