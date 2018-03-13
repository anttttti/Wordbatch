#include "avx_ext.h"

#define USE_AVX2
#define USE_OMP

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_AVX2

#include <immintrin.h>

#endif

//#include <malloc.h> //Deprecated
#include <stdlib.h>
#include <math.h>

double predict_fm_ftrl_avx(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
    double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm, 
    int bias_term, int n_threads) {
    double e = 0.0;
    double e2 = 0.0;
    if (bias_term) e += *w = -*z / ((beta + sqrt(*n)) * ialpha);
    int k, ii;

    /*
    #ifdef USE_OMP
    #pragma omp parallel for
    #endif
    */
    for (ii = 0; ii < lenn; ii++) {
        const int i = inds[ii];
        const double zi = z[i];
        const double sign = (zi < 0) ? -1.0 : 1.0;
        if (sign * zi > L1) {
            const double wi = (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2);
            w[ii + 1] = wi;
            e += wi * vals[ii];
        } else w[ii + 1] = 0.0;
    }

    int num_thread = 1;
    #ifdef USE_OMP
    if (n_threads <= 0) num_thread = omp_get_max_threads();
    else num_thread = n_threads;
    #endif

    double* acwfmk = (double*)malloc(sizeof(double) * D_fm * num_thread);
    #ifdef USE_OMP
    #pragma omp parallel for num_threads(n_threads) private(k)
    #endif
    for (k = 0; k < D_fm * num_thread; k++) acwfmk[k] = 0.0;

    double* wi2_acc = (double*)malloc(sizeof(double) * num_thread * 4);

    double wi2 = 0.0;
    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread) private(k)
    #endif
    for (k = 0; k < num_thread * 4; k++) wi2_acc[k] = 0.0;

    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread) private(ii)
    #endif
    for (ii = 0; ii < lenn; ii++) {

        #ifdef USE_OMP
        const int i_thread = omp_get_thread_num();
        #else
        const int i_thread = 0;
        #endif

        double* pAcwfmk = acwfmk + i_thread * D_fm;
        double* wi2_acck = wi2_acc + i_thread * 4;
        const int i = inds[ii];
        double v = vals[ii];
        const int iD_fm = i * D_fm;
        int k = 0;
        double* z_fmik = z_fm + iD_fm;
        double* w_fmk = pAcwfmk;

        #ifdef USE_AVX2
        __m256d v256 = _mm256_set1_pd(v);
        __m256d w2_256 = _mm256_loadu_pd(wi2_acck);
        while (k + 3 < D_fm) {
            __m256d d = _mm256_mul_pd(_mm256_loadu_pd(z_fmik), v256);
            _mm256_storeu_pd(w_fmk, _mm256_add_pd(_mm256_loadu_pd(w_fmk), d));
            w2_256 = _mm256_add_pd(w2_256, _mm256_mul_pd(d, d));
            k += 4;
            z_fmik += 4;
            w_fmk += 4;
        }
        _mm256_storeu_pd(wi2_acck, w2_256);
        #endif

        // Tail end
        double d;
        while(k < D_fm) {
            pAcwfmk[k++] += d = *z_fmik++ * v;
            wi2 += d*d;
        }
    }

    for (k = 0; k < D_fm; k++) {
        double wfmk = 0.0;
        for (int i_thread = 0; i_thread < num_thread;) wfmk += acwfmk[i_thread++ * D_fm + k];
        *w_fm++ = wfmk;
        e2 += wfmk* wfmk;
    }

    for (k = 0; k < num_thread * 4;) wi2 += wi2_acc[k++];

    free(acwfmk);
    free(wi2_acc);
    e2 = (e2 - wi2) * 0.5 * weight_fm;
    return e + e2;
}

void update_fm_ftrl_avx(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z,
    double* n, double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, int bias_term,
    int n_threads) {

    #ifdef USE_OMP
    int num_thread;
    if (n_threads <= 0) num_thread = omp_get_max_threads();
    else num_thread = n_threads;
    #endif

    const double e_sq = e * e;

    if (bias_term) {
        *z += e - ((sqrt(*n + e_sq) - sqrt(*n)) * ialpha) * *w;
        *n += e_sq;
    }
    const double L2_fme = L2_fm / e;

    int ii;
    #ifdef USE_OMP
    #pragma omp parallel for num_threads(num_thread) private(ii)
    #endif
    for (ii = 0; ii < lenn; ii++) {
        const int i = inds[ii];
        const double v = vals[ii];
        const double g = e * v;
        const double g2 = g * g;
        const double ni = n[i];

        z[i] += g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[ii + 1];
        n[i] += g2;

        double* z_fmik = z_fm + i * D_fm;
        double* w_fmk = w_fm;
        const double lr = g* alpha_fm / (sqrt(n_fm[i]) + 1.0);
        const double reg = v - L2_fme;

        int k = 0;
        #ifdef USE_AVX2
        __m256d reg2 = _mm256_set1_pd(reg);
        __m256d lr2 = _mm256_set1_pd(lr);
        while (k + 3 < D_fm) {
            __m256d z0 = _mm256_loadu_pd(z_fmik);
            _mm256_storeu_pd(z_fmik,
                _mm256_sub_pd(z0, _mm256_mul_pd(lr2,
                             _mm256_sub_pd(_mm256_loadu_pd(w_fmk),
                             _mm256_mul_pd(z0, reg2)))));
            w_fmk+= 4;
            z_fmik+= 4;
            k+= 4;
        }
        #endif
        while (k++ < D_fm) *z_fmik++ -= lr * (*w_fmk++ - *z_fmik * reg); // Tail end

        n_fm[i] += e_sq;
    }
}
