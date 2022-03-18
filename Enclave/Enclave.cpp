/*
 * Created on Fri Feb 14 2020
 *
 * Copyright (c) 2020 xxx xxx, xxxx
 */

#include "Enclave.h"
#include "sgx_trts.h"
#include "sgx_thread.h" //for thread manipulation
#include "Enclave_t.h"  /* print_string */
#include <stdarg.h>
#include <stdio.h>
//#include <thread>

void printf(const char *fmt, ...)
{
    PRINT_BLOCK();
}

void sgx_printf(const char *fmt, ...)
{
    PRINT_BLOCK();
}

void empty_ecall()
{
    sgx_printf("Inside empty ecall\n");
}

void fread(void *ptr, size_t size, size_t nmemb, int fp)
{

    ocall_fread(ptr, size, nmemb);
}
void conv_fread(void *ptr, int fread_index, size_t size, size_t nmemb, int fp)
{
    ocall_conv_fread(ptr, fread_index, size, nmemb);
}
void conv_weights(float* ptr, int fread_index, size_t size, size_t nmemb){
    ocall_conv_weights(ptr, fread_index, size, nmemb);
}
void fwrite(void *ptr, size_t size, size_t nmemb, int fp)
{

    ocall_fwrite(ptr, size, nmemb);
}

void conv_outsourcing(float* input, float* output, int fread_index, size_t size, size_t nmemb, 
int l_n, int l_groups, int l_c, int l_size, int l_h, int l_w, int l_stride, int l_pad, int output_size){
    ocall_conv_outsourcing(input, output, fread_index, size, nmemb, l_n, l_groups, l_c, l_size, l_h, l_w, l_stride, l_pad, output_size);
}