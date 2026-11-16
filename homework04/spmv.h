#ifndef SPMV_H
#define SPMV_H
extern "C" {
#include "common.h"
}

void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b);
void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b);
void spmv_gpu_2(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
                int m, int n, int nnz, double* x, double* b);
void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b);
void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b);
void get_result_gpu(double* dev_b, double* b, int m);
template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in);
#endif

