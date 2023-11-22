#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dgesv.h"


int my_dgesv(int N,int nrhs, double *A, double *B) {

    double pivot;
    int *temp_piv,i,j;

    temp_piv = (int *)malloc(N * sizeof(int));

    // Initialize pivot indices
    for (i = 0; i < N; i++) 
        temp_piv[i] = i;


    // Gaussian elimination with partial pivoting
    for (i = 0; i < N; i++) {
        // Find pivot row
        pivot = fabs(A[temp_piv[i] * N + i]);
        int current_row = i;

        for (j = i + 1; j < N; j++) {
            double abs_val = fabs(A[temp_piv[j] * N + i]);
            if (abs_val > pivot) {
                pivot = abs_val;
                current_row = j;
            }
        }

        // Swap pivot indices
        int temp = temp_piv[i];
        temp_piv[i] = temp_piv[current_row];
        temp_piv[current_row] = temp;

        // Swap rows in A and B using the same pivot order
        for (j = 0; j < N; j++) {
            double temp_a = A[temp_piv[i] * N + j];
            A[temp_piv[i] * N + j] = A[temp_piv[current_row] * N + j];
            A[temp_piv[current_row] * N + j] = temp_a;

            double temp_b = B[temp_piv[i] * N + j];
            B[temp_piv[i] * N + j] = B[temp_piv[current_row] * N + j];
            B[temp_piv[current_row] * N + j] = temp_b;
        }

        // Normalize the pivot row
        pivot = A[temp_piv[i] * N + i];
        for (j = i; j < N; j++) 
            A[temp_piv[i] * N + j] /= pivot;
        
        for (j = 0; j < N; j++) 
            B[temp_piv[i] * N + j] /= pivot;
        
        int k;
        // Eliminate non-zero entries in the pivot column
        for (k = 0; k < N; k++) {
            if (k != i) {
                pivot = A[temp_piv[k] * N + i];
                for (j = i; j < N; j++) 
                    A[temp_piv[k] * N + j] -= pivot * A[temp_piv[i] * N + j];
                
                for (j = 0; j < N; j++) 
                    B[temp_piv[k] * N + j] -= pivot * B[temp_piv[i] * N + j];
                
            }
        }
    }

    free(temp_piv);

    return 0;
}