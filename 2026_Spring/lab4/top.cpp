#include "dcl.h"

// =========================================================================
// STAGE 1: Sensor Calibration (Polynomial Expansion)
// =========================================================================
void stage1_prescale(const data_t in[N], data_t out[N]) {
    // A 3rd-order polynomial transformation simulating sensor calibration
    // Equation: y = c3*x^3 + c2*x^2 + c1*x + c0
    const acc_t c3 =  0.002;
    const acc_t c2 = -0.015;
    const acc_t c1 =  1.150;
    const acc_t c0 =  0.500;

    for (int i = 0; i < N; i++) {
        acc_t x = (acc_t)in[i];
        
        // Calculate powers sequentially
        acc_t x2 = x * x;
        acc_t x3 = x2 * x;
        
        // Compute individual polynomial terms
        acc_t term3 = c3 * x3;
        acc_t term2 = c2 * x2;
        acc_t term1 = c1 * x;
        
        // Accumulate terms
        acc_t res = term3 + term2 + term1 + c0;
        
        // Apply a simulated high-frequency noise floor correction
        // based on odd/even indexing (simulates an alternating current artifact)
        acc_t artifact_correction = (i % 2 == 0) ? (acc_t)0.05 : (acc_t)(-0.05);
        res += artifact_correction;

        out[i] = (data_t)res;
    }
}

// =========================================================================
// STAGE 2: 16-Tap FIR Filter
// =========================================================================
void stage2_fir_filter(const data_t in[N], data_t out[N]) {
    // 16-tap asymmetric low-pass filter coefficients
    const acc_t taps[16] = {
        0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25,
        0.20, 0.15, 0.10, 0.05, 0.02, -0.02, -0.05, -0.08
    };
    
    // Explicit shift register (Standard sequential C implementation)
    data_t shift_reg[16];
    for (int k = 0; k < 16; k++) {
        shift_reg[k] = 0;
    }

    for (int i = 0; i < N; i++) {
        // Shift data down the register
        for (int k = 15; k > 0; k--) {
            shift_reg[k] = shift_reg[k - 1];
        }
        // Load new input sample
        shift_reg[0] = in[i];

        // Multiply-Accumulate (MAC) loop
        acc_t acc = 0;
        for (int k = 0; k < 16; k++) {
            acc += (acc_t)shift_reg[k] * taps[k];
        }
        
        // Baseline drift compensation based on the oldest sample
        acc_t drift_corr = (acc_t)shift_reg[15] * (acc_t)0.01;
        acc -= drift_corr;

        out[i] = (data_t)acc;
    }
}

// =========================================================================
// STAGE 3: Block Statistics (Mean and Mean Absolute Deviation)
// =========================================================================
void stage3_compute_stats(const data_t in[N], stat_t block_mean[N/BLOCK], stat_t block_peak[N/BLOCK], stat_t block_mad[N/BLOCK]) {
    for (int b = 0; b < (N / BLOCK); b++) {
        int base = b * BLOCK;
        
        // Pass 1: Find Mean and Peak Absolute Value
        acc_t sum = 0;
        data_t peak = 0;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = in[base + i];
            sum += (acc_t)val;
            
            data_t abs_val = (val < 0) ? (data_t)(-val) : val;
            if (abs_val > peak) {
                peak = abs_val;
            }
        }
        stat_t mean = (stat_t)(sum / (acc_t)BLOCK);
        block_mean[b] = mean;
        block_peak[b] = (stat_t)peak;

        // Pass 2: Calculate Mean Absolute Deviation (MAD) for variance estimation
        acc_t mad_sum = 0;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = in[base + i];
            acc_t diff = (acc_t)val - (acc_t)mean;
            acc_t abs_diff = (diff < (acc_t)0) ? (acc_t)(-diff) : (acc_t)diff;
            mad_sum += abs_diff;
        }
        block_mad[b] = (stat_t)(mad_sum / (acc_t)BLOCK);
    }
}

// =========================================================================
// STAGE 4: Dynamic Range Normalization
// =========================================================================
void stage4_normalize(const data_t in[N], const stat_t block_mean[N/BLOCK], const stat_t block_peak[N/BLOCK], const stat_t block_mad[N/BLOCK], data_t out[N]) {
    for (int b = 0; b < (N / BLOCK); b++) {
        stat_t mean = block_mean[b];
        stat_t peak = block_peak[b];
        stat_t mad  = block_mad[b];
        
        // Calculate dynamic gain based on MAD (Target variance / Actual variance)
        acc_t target_mad = 2.0;
        acc_t raw_gain = target_mad / ((acc_t)mad + (acc_t)0.1); // Add epsilon to prevent div-by-zero
        
        // Cap the maximum gain so we don't amplify silent blocks (noise)
        acc_t max_gain = 4.0;
        acc_t gain = (raw_gain > max_gain) ? max_gain : raw_gain;
        
        // Peak suppression: If the block contains an extreme spike, reduce the gain
        acc_t peak_suppression = 1.0;
        if (peak > (data_t)12.0) {
            peak_suppression = (acc_t)12.0 / (acc_t)peak;
        }
        
        acc_t final_gain = gain * peak_suppression;

        // Apply normalization to the block
        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = in[base + i];
            
            // Mean-center the signal, then apply dynamic gain
            acc_t centered = (acc_t)val - (acc_t)mean;
            acc_t scaled = centered * final_gain;
            
            out[base + i] = (data_t)scaled;
        }
    }
}

// =========================================================================
// STAGE 5: Multi-Zone Activation and Output Formatting
// =========================================================================
void stage5_post_process(const data_t in[N], data_t out[N]) {
    // Squelch zone for noise, linear active region, and soft saturation for peaks
    const data_t DEAD_ZONE = 0.5;
    const acc_t  LINEAR_GAIN = 1.25;
    const data_t SAT_LIMIT = 8.0;

    for (int i = 0; i < N; i++) {
        data_t val = in[i];
        data_t abs_val = (val < 0) ? (data_t)(-val) : val;
        data_t sign = (val < 0) ? (data_t)(-1.0) : (data_t)(1.0);
        
        data_t processed_val;

        // Zone 1: Dead-zone suppression (Squelch)
        if (abs_val < DEAD_ZONE) {
            processed_val = 0;
        } 
        // Zone 2: Linear Region
        else if (abs_val < SAT_LIMIT) {
            acc_t active_val = (acc_t)(abs_val - DEAD_ZONE);
            processed_val = (data_t)(active_val * LINEAR_GAIN) * sign;
        } 
        // Zone 3: Soft Saturation Region (Compressing the overage)
        else {
            acc_t max_linear_val = (acc_t)(SAT_LIMIT - DEAD_ZONE);
            acc_t overage = (acc_t)(abs_val - SAT_LIMIT);
            
            // Compress the signal that exceeds the saturation limit
            acc_t compressed = overage * (acc_t)0.25; 
            processed_val = (data_t)((max_linear_val * LINEAR_GAIN) + compressed) * sign;
        }
        
        // Final Hard Clamp for hardware safety limits
        if (processed_val > (data_t)10.0) {
            processed_val = (data_t)10.0;
        } else if (processed_val < (data_t)(-10.0)) {
            processed_val = (data_t)(-10.0);
        }
        
        out[i] = processed_val;
    }
}

// =========================================================================
// TOP LEVEL MODULE
// =========================================================================
void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1
    #pragma HLS interface s_axilite port=return

    // Static buffers to hold intermediate data between sequential stages
    static data_t buf1[N];
    static data_t buf2[N];
    static data_t buf3[N];
    
    // Statistics buffers
    static stat_t block_mean[N / BLOCK];
    static stat_t block_peak[N / BLOCK];
    static stat_t block_mad[N / BLOCK];

    // Execute kernels sequentially (Baseline behavior)
    stage1_prescale(in, buf1);
    stage2_fir_filter(buf1, buf2);
    stage3_compute_stats(buf2, block_mean, block_peak, block_mad);
    stage4_normalize(buf2, block_mean, block_peak, block_mad, buf3);
    stage5_post_process(buf3, out);
}