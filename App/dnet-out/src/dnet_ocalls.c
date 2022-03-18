/*
 * Created on Fri Feb 14 2020
 *
 * Copyright (c) 2020 xxx xxx, xxxx
 */

#include "dnet_ocalls.h"

//#define DISABLE_CACHE //open files with O_DIRECT thus bypassing kernel page cache for reads and writes

//File pointers used for reading and writing files from within the enclave runtime
FILE *write_fp = NULL;
FILE *read_fp = NULL;
//file descriptor used by open
int fd;
float* conv_weights[20];
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate
     * the input string to prevent buffer overflow.
     */
    printf("%s", str);
}

/* Free section in untrusted memory*/
void ocall_free_sec(section *sec)
{
    //printf("Freeing section in ocall..\n");
    free_section(sec);
}

void ocall_free_list(list *list)
{
    free_list(list);
}

// 0 for read: 1 for write
void ocall_open_file(const char *filename, flag oflag)
{

    if (!write_fp && !read_fp) //fp == NULL
    {
        switch (oflag)
        {
        case O_RDONLY:
#ifdef DISABLE_CACHE
            fd = open(filename, O_RDONLY | O_CREAT | O_DIRECT);
            read_fp = fdopen(fd, "rb");

#else
            read_fp = fopen(filename, "rb");
#endif

            printf("Opened file in read only mode\n");
            break;
        case O_WRONLY:
#ifdef DISABLE_CACHE
            fd = open(filename, O_WRONLY | O_CREAT | O_DIRECT);
            write_fp = fdopen(fd, "wb");

#else
            write_fp = fopen(filename, "wb");
#endif

            printf("Opened file in write only mode\n");
            break;
        case O_RDPLUS:
#ifdef DISABLE_CACHE
//TODO
#endif
            read_fp = fopen(filename, "r+");
            break;
        case O_WRPLUS:
#ifdef DISABLE_CACHE
//TODO
#endif
            write_fp = fopen(filename, "w+");
            break;
        default:; //nothing to do
        }
    }
    else
    {
        printf("Problem with file pointer..\n");
    }
}

/**
 * Close all file descriptors
 */
void ocall_close_file()
{
#ifdef DISABLE_CACHE

#endif
    if (read_fp) //fp != NULL
    {
        fclose(read_fp);
        read_fp = NULL;
    }
    if (write_fp)
    {
        fclose(write_fp);
        write_fp = NULL;
    }
}

void ocall_fread(void *ptr, size_t size, size_t nmemb)
{
    if (read_fp)
    {
        fread(ptr, size, nmemb, read_fp);
    }
    else
    {
        printf("Corrupt file pointer..\n");
        abort();
    }
}
void ocall_conv_fread(void *ptr, int fread_index, size_t size, size_t nmemb){
    conv_weights[fread_index] = calloc(nmemb, size);
    fread(conv_weights[fread_index], size, nmemb, read_fp);
}
void ocall_conv_weights(float *ptr, int fread_index, size_t size, size_t nmemb){
    for(size_t i=0; i<nmemb; i++) ptr[i] = conv_weights[fread_index][i]; 
}
void ocall_fwrite(void *ptr, size_t size, size_t nmemb)
{
    int ret;
    if (write_fp)
    {
        fwrite(ptr, size, nmemb, write_fp);
        //make sure it is flushed to disk first
        ret = fflush(write_fp);
        if (ret != 0)
            printf("fflush did not work..\n");
        /*  ret = fsync(fileno(fp));
        if (ret < 0)
            printf("fsync did not work..\n"); */
        return;
    }
    else
    {
        printf("Corrupt file pointer..\n");
        abort();
    }
}

void ocall_conv_outsourcing(float* input, float* output, int fread_index, size_t size, size_t nmemb, 
int l_n, int l_groups, int l_c, int l_size, int l_h, int l_w, int l_stride, int l_pad, int output_size){
    int m = l_n / l_groups;
    int k = l_size * l_size * l_c / l_groups;
    int n = output_size;
    int nweights = l_c / l_groups * l_size*l_size*l_n;
    int workspace_size = output_size*l_size*l_size*l_c/l_groups;
    int input_size = l_w*l_h;
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < l_groups; ++j)
        {
            float *a = conv_weights[fread_index] + j * nweights / l_groups;
            float *b = calloc(workspace_size, sizeof(float));
            float *c = output + (i * l_groups + j) * n * m;
            float *im = input + (i * l_groups + j) * l_c / l_groups * input_size;

            if (l_size == 1)
            {
                b = im;
            }
            else
            {
                im2col_cpu(im, l_c / l_groups, l_h, l_w, l_size, l_stride, l_pad, b);
            }
            gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            
            //printf("group %d / %d done\n", j, l_groups);
        }
    }
}
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;

    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}