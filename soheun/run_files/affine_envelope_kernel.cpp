#include <cstddef>
#include <cfloat>
#include <limits>

extern "C" int affine_ld_mant_dig() { return LDBL_MANT_DIG; }

// Literal translation of the reference upper-envelope stack loop.
// Inputs have already been sorted and duplicate slopes reduced by NumPy.
extern "C" std::size_t affine_upper_stack(
    std::size_t n, const double* s, const double* a,
    long double* hs, long double* ha, long double* hx) {
    std::size_t k=0;
    for (std::size_t i=0; i<n; ++i) {
        const long double si=static_cast<long double>(s[i]);
        const long double ai=static_cast<long double>(a[i]);
        long double x=-std::numeric_limits<long double>::infinity();
        while (k) {
            x=(ha[k-1]-ai)/(si-hs[k-1]);
            if (x>hx[k-1]) break;
            --k;
        }
        if (!k) x=-std::numeric_limits<long double>::infinity();
        hs[k]=si; ha[k]=ai; hx[k]=x; ++k;
    }
    return k;
}
