#include "dcl.h"

// Helper functions
static inline data_t max_fp(data_t a, data_t b) {
    return (a > b) ? a : b;
}

void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1
    #pragma HLS interface s_axilite port=return

    static data_t stage1[N];
    static data_t stage2[N];
    static stat_t block_peak[N / BLOCK];
    static data_t stage3[N];

    #pragma HLS array_partition variable=stage1 cyclic factor=4 dim=1
    #pragma HLS array_partition variable=stage2 cyclic factor=4 dim=1
    #pragma HLS array_partition variable=block_peak cyclic factor=4 dim=1
    #pragma HLS array_partition variable=stage3 cyclic factor=4 dim=1

    // ---------------------------------------------------------------------
    // STAGE 1: Pre-scale and Bias
    // ---------------------------------------------------------------------
    for (int i = 0; i < N; i++) {
        stage1[i] = (data_t)((acc_t)in[i] * (acc_t)1.125 + (acc_t)0.5);
    }

    // ---------------------------------------------------------------------
    // STAGE 2: 4-Tap Asymmetric Filter (Sliding Window)
    // ---------------------------------------------------------------------
    for (int i = 0; i < N; i++) {
        #pragma HLS unroll factor=4
        data_t x0 = stage1[i];
        data_t x1 = (i >= 1) ? stage1[i - 1] : (data_t)0;
        data_t x2 = (i >= 2) ? stage1[i - 2] : (data_t)0;
        data_t x3 = (i >= 3) ? stage1[i - 3] : (data_t)0;

        acc_t acc = (acc_t)x0 * (acc_t)0.4 + 
                    (acc_t)x1 * (acc_t)0.3 + 
                    (acc_t)x2 * (acc_t)0.2 + 
                    (acc_t)x3 * (acc_t)0.1;
        stage2[i] = (data_t)acc;
    }

    // ---------------------------------------------------------------------
    // STAGE 3: Block Peak Detection (Slow path reduction)
    // ---------------------------------------------------------------------
    for (int b = 0; b < (N / BLOCK); b++) {
        data_t peak = 0;
        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = stage2[base + i];
            data_t abs_val = (val < 0) ? (data_t)(-val) : val;
            peak = max_fp(peak, abs_val);
        }
        // Add epsilon to prevent division by zero in the next stage
        block_peak[b] = (stat_t)(peak + (data_t)0.1); 
    }

    // ---------------------------------------------------------------------
    // STAGE 4: Join & Normalize
    // ---------------------------------------------------------------------
    for (int b = 0; b < (N / BLOCK); b++) {
        stat_t peak = block_peak[b];
        
        // Unoptimized division bottleneck
        stat_t inv_peak = (stat_t)((acc_t)1.0 / (acc_t)peak);

        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            stage3[base + i] = (data_t)((acc_t)stage2[base + i] * (acc_t)inv_peak);
        }
    }

    // ---------------------------------------------------------------------
    // STAGE 5: Post-process (Leaky ReLU) & Store
    // ---------------------------------------------------------------------
    for (int i = 0; i < N; i++) {
        data_t val = stage3[i];
        data_t out_val = (val > 0) ? val : (data_t)((acc_t)val * (acc_t)0.1);
        
        // Final Clamp
        if (out_val > (data_t)10.0) out_val = (data_t)10.0;
        if (out_val < (data_t)(-10.0)) out_val = (data_t)(-10.0);
        
        out[i] = out_val;
    }
}