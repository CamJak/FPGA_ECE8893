//Save of oldest best top.cpp
// 51499 cycles * 6.269 CP = 0.3228ms

#include "dcl.h"

// Baseline implementation for HLS.
// Students will optimize this (loops, memory access, etc.).
void top_kernel(data_t A_DRAM[N_ROWS][N_COLS],
                data_t C_DRAM[N_ROWS][N_COLS]) {
#pragma HLS interface m_axi port=A_DRAM offset=slave bundle=A
#pragma HLS interface m_axi port=C_DRAM offset=slave bundle=C
#pragma HLS interface s_axilite port=return

    // On-chip buffers for A_DRAM and C_DRAM
    data_t A[N_ROWS][N_COLS];
    data_t C[N_ROWS][N_COLS];

#pragma HLS array_partition variable=A dim=2 factor=8 cyclic
#pragma HLS array_partition variable=C dim=2 factor=8 cyclic

    // Storage for normalized A
    data_t B[N_ROWS][N_COLS];

#pragma HLS array_partition variable=B dim=2 factor=8 cyclic

    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS pipeline II=1
        for (int j = 0; j < N_COLS; j++) {
            A[i][j] = A_DRAM[i][j];
        }
    }

    // Storage for sums
    data_t col_sum[N_COLS];
    data_t row_sum[N_ROWS];
#pragma HLS array_partition variable=col_sum dim=1 complete
#pragma HLS array_partition variable=row_sum dim=1 complete

    // Init sums to 0
    for (int j = 0; j < N_COLS; j++) {
#pragma HLS unroll
        col_sum[j] = 0;
    }
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS unroll
        row_sum[i] = 0;
    }

    // Phase 1: Row-wise normalization
    for (int i = 0; i < N_ROWS; i++) {
        // Compute row sum
        for (int j = 0; j < N_COLS; j+=8) {
#pragma HLS pipeline
            data_t local_sum = 0;
            for (int jj = 0; jj < 8; jj++) {
#pragma HLS unroll
                local_sum += A[i][j+jj];
            }
            row_sum[i] += local_sum;
        }

        // Avoid division by zero, add small bias
        data_t denom = row_sum[i] + (data_t)1.0;

        // Normalize each element in the row and add to col_sum
        for (int j = 0; j < N_COLS; j+=8) {
#pragma HLS pipeline
            for (int jj = 0; jj < 8; jj++) {
#pragma HLS unroll
                data_t tmp = A[i][j+jj] / denom;
                B[i][j+jj] = tmp;
                col_sum[j+jj] += tmp;
            }
        }
    }

    // Phase 2: Column-wise scaling
    for (int j = 0; j < N_COLS; j++) {
        // Compute average as scale
        data_t scale = col_sum[j] / (data_t)N_ROWS;

        // Apply scale to each element in the column
        for (int i = 0; i < N_ROWS; i+=8) {
#pragma HLS pipeline
            for (int ii = 0; ii < 8; ii++) {
#pragma HLS unroll
                C[i+ii][j] = B[i+ii][j] * scale;
            }
        }
    }
    
    for (int i = 0; i < N_ROWS; i++) {
#pragma HLS pipeline II=1
        for (int j = 0; j < N_COLS; j++) {
            C_DRAM[i][j] = C[i][j];
        }
    }
}