#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "dcl.h"

// ---------------------------------------------------------------------
// Golden Kernel: The Reference Implementation
// ---------------------------------------------------------------------
void golden_kernel(const data_t in[N], data_t out[N]) {
    static data_t stage1[N];
    static data_t stage2[N];
    static stat_t block_peak[N / BLOCK];
    static data_t stage3[N];

    // Stage 1: Pre-scale and Bias
    for (int i = 0; i < N; i++) {
        stage1[i] = (data_t)((double)in[i] * 1.125 + 0.5);
    }

    // Stage 2: 4-Tap Asymmetric Filter
    for (int i = 0; i < N; i++) {
        data_t x0 = stage1[i];
        data_t x1 = (i >= 1) ? stage1[i - 1] : (data_t)0;
        data_t x2 = (i >= 2) ? stage1[i - 2] : (data_t)0;
        data_t x3 = (i >= 3) ? stage1[i - 3] : (data_t)0;

        double acc = (double)x0 * 0.4 + (double)x1 * 0.3 + (double)x2 * 0.2 + (double)x3 * 0.1;
        stage2[i] = (data_t)acc;
    }

    // Stage 3: Block Peak Detection
    for (int b = 0; b < (N / BLOCK); b++) {
        data_t peak = 0;
        for (int i = 0; i < BLOCK; i++) {
            data_t val = stage2[b * BLOCK + i];
            data_t abs_val = (val < 0) ? (data_t)(-val) : val;
            if (abs_val > peak) peak = abs_val;
        }
        block_peak[b] = (stat_t)(peak + (data_t)0.1);
    }

    // Stage 4: Join & Normalize
    for (int b = 0; b < (N / BLOCK); b++) {
        double inv_peak = 1.0 / (double)block_peak[b];
        for (int i = 0; i < BLOCK; i++) {
            stage3[b * BLOCK + i] = (data_t)((double)stage2[b * BLOCK + i] * inv_peak);
        }
    }

    // Stage 5: Post-process (Leaky ReLU) & Store
    for (int i = 0; i < N; i++) {
        data_t val = stage3[i];
        double out_val = (val > 0) ? (double)val : ((double)val * 0.1);
        
        if (out_val > 10.0) out_val = 10.0;
        if (out_val < -10.0) out_val = -10.0;
        
        out[i] = (data_t)out_val;
    }
}

// ---------------------------------------------------------------------
// Main Testbench Logic
// ---------------------------------------------------------------------
int main() {
    // Allocate memory for inputs and outputs
    std::vector<data_t> test_input(N);
    std::vector<data_t> hw_output(N);
    std::vector<data_t> sw_output(N);

    // Initialize input with dummy data (e.g., a sine wave with noise)
    for (int i = 0; i < N; i++) {
        test_input[i] = (data_t)(sin(i * 0.1) * 5.0 + (rand() % 10) * 0.1);
    }

    std::cout << ">> Running Golden Kernel (Software Reference)..." << std::endl;
    golden_kernel(test_input.data(), sw_output.data());

    std::cout << ">> Running Top Kernel (Hardware/Baseline)..." << std::endl;
    // In a real Vitis/HLS flow, this would call the actual hardware function
    top_kernel(test_input.data(), hw_output.data());

    // Error Checking
    int error_count = 0;
    double max_diff = 0.0;
    const double EPSILON = 0.01; // Allow for slight floating point variations

    for (int i = 0; i < N; i++) {
        double diff = std::abs((double)hw_output[i] - (double)sw_output[i]);
        if (diff > max_diff) max_diff = diff;

        if (diff > EPSILON) {
            if (error_count < 10) { // Log first 10 errors
                std::cout << "Error at index " << i 
                          << ": Expected " << sw_output[i] 
                          << ", Got " << hw_output[i] << std::endl;
            }
            error_count++;
        }
    }

    std::cout << "---------------------------------------" << std::endl;
    if (error_count == 0) {
        std::cout << "TEST PASSED! (Max difference: " << max_diff << ")" << std::endl;
    } else {
        std::cout << "TEST FAILED! Total Errors: " << error_count << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
    }
    std::cout << "---------------------------------------" << std::endl;

    return (error_count == 0) ? 0 : 1;
}