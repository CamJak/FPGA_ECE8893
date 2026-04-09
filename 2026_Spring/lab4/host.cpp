#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib> // Added for rand()
#include "dcl.h"

// ---------------------------------------------------------------------
// Golden Kernel Stages (Bit-Accurate to Hardware)
// ---------------------------------------------------------------------

void golden_stage1(const data_t in[N], data_t out[N]) {
    const acc_t c3 =  0.002;
    const acc_t c2 = -0.015;
    const acc_t c1 =  1.150;
    const acc_t c0 =  0.500;

    for (int i = 0; i < N; i++) {
        acc_t x = (acc_t)in[i];
        acc_t x2 = x * x;
        acc_t x3 = x2 * x;
        
        acc_t term3 = c3 * x3;
        acc_t term2 = c2 * x2;
        acc_t term1 = c1 * x;
        
        acc_t res = term3 + term2 + term1 + c0;
        acc_t artifact_correction = (i % 2 == 0) ? (acc_t)0.05 : (acc_t)(-0.05);
        
        res += artifact_correction;
        out[i] = (data_t)res;
    }
}

void golden_stage2(const data_t in[N], data_t out[N]) {
    const acc_t taps[16] = {
        0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25,
        0.20, 0.15, 0.10, 0.05, 0.02, -0.02, -0.05, -0.08
    };
    data_t shift_reg[16] = {0};

    for (int i = 0; i < N; i++) {
        for (int k = 15; k > 0; k--) {
            shift_reg[k] = shift_reg[k - 1];
        }
        shift_reg[0] = in[i];

        acc_t acc = 0;
        for (int k = 0; k < 16; k++) {
            acc += (acc_t)shift_reg[k] * taps[k];
        }
        
        acc_t drift_corr = (acc_t)shift_reg[15] * (acc_t)0.01;
        acc -= drift_corr;
        out[i] = (data_t)acc;
    }
}

void golden_stage3(const data_t in[N], stat_t block_mean[N/BLOCK], stat_t block_peak[N/BLOCK], stat_t block_mad[N/BLOCK]) {
    for (int b = 0; b < (N / BLOCK); b++) {
        int base = b * BLOCK;
        acc_t sum = 0;
        data_t peak = 0;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = in[base + i];
            sum += (acc_t)val;
            data_t abs_val = (val < 0) ? (data_t)(-val) : val;
            if (abs_val > peak) peak = abs_val;
        }
        stat_t mean = (stat_t)(sum / (acc_t)BLOCK);
        block_mean[b] = mean;
        block_peak[b] = (stat_t)peak;

        acc_t mad_sum = 0;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = in[base + i];
            acc_t diff = (acc_t)val - (acc_t)mean;
            acc_t abs_diff = (diff < 0) ? (acc_t)(-diff) : diff;
            mad_sum += abs_diff;
        }
        block_mad[b] = (stat_t)(mad_sum / (acc_t)BLOCK);
    }
}

void golden_stage4(const data_t in[N], const stat_t block_mean[N/BLOCK], const stat_t block_peak[N/BLOCK], const stat_t block_mad[N/BLOCK], data_t out[N]) {
    for (int b = 0; b < (N / BLOCK); b++) {
        stat_t mean = block_mean[b];
        stat_t peak = block_peak[b];
        stat_t mad  = block_mad[b];
        
        acc_t target_mad = 2.0;
        acc_t raw_gain = target_mad / ((acc_t)mad + (acc_t)0.1); 
        acc_t max_gain = 4.0;
        acc_t gain = (raw_gain > max_gain) ? max_gain : raw_gain;
        
        acc_t peak_suppression = 1.0;
        if (peak > (data_t)12.0) {
            peak_suppression = (acc_t)12.0 / (acc_t)peak;
        }
        acc_t final_gain = gain * peak_suppression;

        int base = b * BLOCK;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = in[base + i];
            acc_t centered = (acc_t)val - (acc_t)mean;
            acc_t scaled = centered * final_gain;
            out[base + i] = (data_t)scaled;
        }
    }
}

void golden_stage5(const data_t in[N], data_t out[N]) {
    const data_t DEAD_ZONE = 0.5;
    const acc_t  LINEAR_GAIN = 1.25;
    const data_t SAT_LIMIT = 8.0;

    for (int i = 0; i < N; i++) {
        data_t val = in[i];
        data_t abs_val = (val < 0) ? (data_t)(-val) : val;
        data_t sign = (val < 0) ? (data_t)(-1.0) : (data_t)(1.0);
        
        data_t processed_val;

        if (abs_val < DEAD_ZONE) {
            processed_val = 0;
        } else if (abs_val < SAT_LIMIT) {
            acc_t active_val = (acc_t)(abs_val - DEAD_ZONE);
            processed_val = (data_t)(active_val * LINEAR_GAIN) * sign;
        } else {
            acc_t max_linear_val = (acc_t)(SAT_LIMIT - DEAD_ZONE);
            acc_t overage = (acc_t)(abs_val - SAT_LIMIT);
            acc_t compressed = overage * (acc_t)0.25; 
            processed_val = (data_t)((max_linear_val * LINEAR_GAIN) + compressed) * sign;
        }
        
        if (processed_val > (data_t)10.0) {
            processed_val = (data_t)10.0;
        } else if (processed_val < (data_t)(-10.0)) {
            processed_val = (data_t)(-10.0);
        }
        out[i] = processed_val;
    }
}

// Wrapper for the full software reference
void golden_kernel(const data_t in[N], data_t out[N]) {
    // Replaced large static arrays with std::vector to prevent 
    // persistent data bugs between runs and stack overflows.
    std::vector<data_t> b1(N), b2(N), b3(N);
    std::vector<stat_t> m(N/BLOCK), p(N/BLOCK), d(N/BLOCK);

    golden_stage1(in, b1.data());
    golden_stage2(b1.data(), b2.data());
    golden_stage3(b2.data(), m.data(), p.data(), d.data());
    golden_stage4(b2.data(), m.data(), p.data(), d.data(), b3.data());
    golden_stage5(b3.data(), out);
}

// ---------------------------------------------------------------------
// Main Testbench
// ---------------------------------------------------------------------
int main() {
    std::vector<data_t> test_input(N);
    std::vector<data_t> hw_output(N);
    std::vector<data_t> sw_output(N);

    // Initialize with a noisy signal
    for (int i = 0; i < N; i++) {
        test_input[i] = (data_t)(sin(i * 0.05) * 6.0 + (rand() % 20) * 0.05);
    }

    std::cout << ">> Running Golden Reference..." << std::endl;
    golden_kernel(test_input.data(), sw_output.data());

    std::cout << ">> Running Hardware Kernel..." << std::endl;
    top_kernel(test_input.data(), hw_output.data());

    int error_count = 0;
    double max_diff = 0;
    
    // Define 1% Relative Tolerance
    const double TOLERANCE = 0.01; 
    
    // Small constant to prevent division by zero and handle values near the noise floor
    const double MIN_THRESHOLD = 1e-5; 

    for (int i = 0; i < N; i++) {
        double hw_val = (double)hw_output[i];
        double sw_val = (double)sw_output[i];
        double diff = std::abs(hw_val - sw_val);
        
        if (diff > max_diff) max_diff = diff;
        
        double error_metric = 0;
        
        if (std::abs(sw_val) > MIN_THRESHOLD) {
            // Use Relative Error for significant values
            error_metric = diff / std::abs(sw_val);
        } else {
            // Use Absolute Error for values near zero (to avoid division by zero)
            error_metric = diff;
        }

        if (error_metric > TOLERANCE) {
            if (error_count < 10) {
                std::cout << "Tolerance Violation at [" << i << "]: " 
                          << "Expected " << sw_val << ", Got " << hw_val 
                          << " (Error: " << (error_metric * 100.0) << "%)" << std::endl;
            }
            error_count++;
        }
    }

    std::cout << "---------------------------------------" << std::endl;
    if (error_count == 0) {
        std::cout << "SUCCESS: All outputs match perfectly! (Max Diff: " << max_diff << ")" << std::endl;
    } else {
        std::cout << "FAILURE: " << error_count << " errors found. Max diff: " << max_diff << std::endl;
    }
    std::cout << "---------------------------------------" << std::endl;

    return (error_count == 0) ? 0 : 1;
}