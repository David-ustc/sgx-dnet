/*
 * Created on Fri Feb 14 2020
 *
 * Copyright (c) 2020 xxx xxx, xxxx
 */

#ifndef DNET_OCALLS_H
#define DNET_OCALLS_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "darknet.h"

//for open
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//edgerator header file
#include "Enclave_u.h"

#if defined(__cplusplus)

extern "C"
{
#endif

    void ocall_free_sec(section *sec);
    void ocall_free_list(list *list);
    void ocall_print_string(const char *str);
    void ocall_open_file(const char *filename, flag oflag);
    void ocall_close_file();
    void ocall_fread(void *ptr, size_t size, size_t nmemb);
    void ocall_conv_fread(void *ptr, int fread_index, size_t size, size_t nmemb);
    void ocall_conv_weights(float *ptr, int fread_index, size_t size, size_t nmemb);
    
    void ocall_fwrite(void *ptr, size_t size, size_t nmemb);

    void ocall_conv_outsourcing(float* input, float* output, int fread_index, size_t size, size_t nmemb, 
int l_n, int l_groups, int l_c, int l_size, int l_h, int l_w, int l_stride, int l_pad, int output_size);

    void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

    void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
            float *A, int lda, 
            float *B, int ldb,
            float BETA,
            float *C, int ldc);
#if defined(__cplusplus)
}
#endif

#endif /* DNET_OCALLS_H */
