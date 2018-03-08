#ifndef _AVX_EXT_H
#define _AVX_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

double predict_fm_ftrl_avx(const int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
    double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm, int D_fm,
    int bias_term, int nThreads);

int halfer_EXT_INT(int d);

void update_fm_ftrl_avx(const int* inds, double* vals, int lenn, const double e, double ialpha, double* w, double* z,
    double* n, double alpha_fm, const double L2_fm, double* w_fm, double* z_fm, double* n_fm, int D_fm, int bias_term,
    int nThreads);

int doubler_EXT(int d);

#ifdef __cplusplus
}
#endif

#endif   // _AVX_EXT_H
