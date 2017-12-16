#include <iostream>
#include <cblas.h>
#include <omp.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <algorithm>

int lengths []= {4,8,10,100,200,500,1000,2000};
int lengths_for_blocked[] = {4,8,16,128,256,512,1024,2048};
int block_lengths[] = {2,4,8,32,64,128,256,512};

void populate(double* matr,int n,int m){
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            matr[i*n+j] = (double)rand() / RAND_MAX;
        }
    }
}

void populateMatrix(double** matr,int n,int m){
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            matr[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void matmul(double **a,double **b,double **c,int size){

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
           c[i][j]=0;
            for(int k=0;k<size;k++){
               c[i][j]=c[i][j]+(a[i][k] * b[k][j]);
            }
        }
    }
}

void matmulAsVec(double *a,double *b,double *c,int size){

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            for(int k=0;k<size;k++){
                c[i*size+j]=c[i*size+j]+(a[i*size+k] * b[k*size+j]);
            }
        }
    }
}

void matmulTranspose(double **a,double **b,double **c,int size){
     double  **bT = (double**)calloc(size,sizeof(double*));
    for (int i=0;i<size;i++){
        bT[i] = (double*)calloc(size,sizeof(double));
    }

    for (int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            bT[i][j] = b[j][i];
        }
    }

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            c[i][j]=0;
            for(int k=0;k<size;k++){
                c[i][j]=c[i][j]+(a[i][k] * bT[j][k]);
            }
        }
    }
}

void matmulBlock(double **a, double **b,double **c,int size,int block_size){
    for (int k = 0; k < size; k += block_size)
        for (int j = 0; j < size; j += block_size)
            for (int i = 0; i < size; ++i)
                for (int jj = j; jj < std::min(j + block_size, size); ++jj)
                    for (int kk = k; kk < std::min(k + block_size, size); ++kk)
                        c[i][jj] += a[i][kk] * b[kk][jj];
}

int main() {
    double *a,*b,*c;
    double **am,**bm,**cm;

    srand(time(NULL));
    for (int i=0;i<8;i++){

        a = (double*)calloc(lengths[i]*lengths[i],sizeof(double));
        b = (double*)calloc(lengths[i]*lengths[i],sizeof(double));
        c = (double*)calloc(lengths[i]*lengths[i],sizeof(double));

        int lda = lengths[i];
        int ldb = lengths[i];
        int ldc = lengths[i];
        populate(a,lengths[i],lengths[i]);
        populate(b,lengths[i],lengths[i]);
        double start = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lengths[i], lengths[i], lengths[i],
                    1.0, a, lda, b, ldb, 0, c, ldc);
        double end = omp_get_wtime();
        printf("BLAS multiply (%dx%d) time is %f\n",lengths[i],lengths[i],end-start);
        free(a);
        free(b);
        free(c);

    }
    for (int i=0;i<8;i++){

        am = (double**)calloc(lengths[i],sizeof(double*));
        bm = (double**)calloc(lengths[i],sizeof(double*));
        cm = (double**)calloc(lengths[i],sizeof(double*));

        for (int j=0;j<lengths[i];j++){
            am[j] = (double*)calloc(lengths[i], sizeof(double));
            bm[j] = (double*)calloc(lengths[i], sizeof(double));
            cm[j] = (double*)calloc(lengths[i], sizeof(double));
        }

        populateMatrix(am,lengths[i],lengths[i]);
        populateMatrix(bm,lengths[i],lengths[i]);
        double start = omp_get_wtime();
        matmul(am,bm,cm,lengths[i]);
        double end = omp_get_wtime();
        printf("matrix multiply (%dx%d) time is %f\n",lengths[i],lengths[i],end-start);
        free(am);
        free(bm);
        free(cm);

    }
    for (int i=0;i<8;i++){

        am = (double**)calloc(lengths[i],sizeof(double*));
        bm = (double**)calloc(lengths[i],sizeof(double*));
        cm = (double**)calloc(lengths[i],sizeof(double*));

        for (int j=0;j<lengths[i];j++){
            am[j] = (double*)calloc(lengths[i], sizeof(double));
            bm[j] = (double*)calloc(lengths[i], sizeof(double));
            cm[j] = (double*)calloc(lengths[i], sizeof(double));
        }

        populateMatrix(am,lengths[i],lengths[i]);
        populateMatrix(bm,lengths[i],lengths[i]);
        double start = omp_get_wtime();
        matmulTranspose(am,bm,cm,lengths[i]);
        double end = omp_get_wtime();
        printf("matrix transpose multiply (%dx%d) time is %f\n",lengths[i],lengths[i],end-start);
        free(am);
        free(bm);
        free(cm);

    }

    for (int i=0;i<8;i++){

        a = (double*)calloc(lengths[i]*lengths[i],sizeof(double));
        b = (double*)calloc(lengths[i]*lengths[i],sizeof(double));
        c = (double*)calloc(lengths[i]*lengths[i],sizeof(double));

        populate(a,lengths[i],lengths[i]);
        populate(b,lengths[i],lengths[i]);
        double start = omp_get_wtime();
        matmulAsVec(a,b,c,lengths[i]);
        double end = omp_get_wtime();
        printf("vector multiply (%d) time is %f\n",lengths[i],end-start);
        free(a);
        free(b);
        free(c);

    }

    for (int i=0;i<8;i++){

        am = (double**)calloc(lengths_for_blocked[i],sizeof(double*));
        bm = (double**)calloc(lengths_for_blocked[i],sizeof(double*));
        cm = (double**)calloc(lengths_for_blocked[i],sizeof(double*));

        for (int j=0;j<lengths_for_blocked[i];j++){
            am[j] = (double*)calloc(lengths_for_blocked[i], sizeof(double));
            bm[j] = (double*)calloc(lengths_for_blocked[i], sizeof(double));
            cm[j] = (double*)calloc(lengths_for_blocked[i], sizeof(double));
        }

        populateMatrix(am,lengths_for_blocked[i],lengths_for_blocked[i]);
        populateMatrix(bm,lengths_for_blocked[i],lengths_for_blocked[i]);
        double start = omp_get_wtime();
        matmulBlock(am,bm,cm,lengths_for_blocked[i],block_lengths[i]);
        double end = omp_get_wtime();
        printf("block matrix multiply (%dx%d) time is %f\n",lengths_for_blocked[i],lengths_for_blocked[i],end-start);
        free(am);
        free(bm);
        free(cm);

    }
    return 0;
}