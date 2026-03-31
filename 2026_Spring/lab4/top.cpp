#include "dcl.h"
#include <hls_stream.h>

// =========================================================================
// Memory Interfaces (Isolating AXI from compute logic)
// =========================================================================
void read_input(const data_t in[N], hls::stream<data_t>& out_stream) {
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        out_stream.write(in[i]);
    }
}

void write_output(hls::stream<data_t>& in_stream, data_t out[N]) {
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        out[i] = in_stream.read();
    }
}

// =========================================================================
// STAGE 1: Sensor Calibration (Polynomial Expansion)
// =========================================================================
void stage1_prescale(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    const acc_t c3 =  0.002;
    const acc_t c2 = -0.015;
    const acc_t c1 =  1.150;
    const acc_t c0 =  0.500;

    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        acc_t x = (acc_t)in_stream.read();
        
        // Pipelined DSP multipliers will handle this automatically
        acc_t x2 = x * x;
        acc_t x3 = x2 * x;
        
        acc_t term3 = c3 * x3;
        acc_t term2 = c2 * x2;
        acc_t term1 = c1 * x;
        
        acc_t res = term3 + term2 + term1 + c0;
        acc_t artifact_correction = (i % 2 == 0) ? (acc_t)0.05 : (acc_t)(-0.05);
        
        out_stream.write((data_t)(res + artifact_correction));
    }
}

// =========================================================================
// STAGE 2: 16-Tap FIR Filter
// =========================================================================
void stage2_fir_filter(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    const acc_t taps[16] = {
        0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25,
        0.20, 0.15, 0.10, 0.05, 0.02, -0.02, -0.05, -0.08
    };
    
    data_t shift_reg[16];
    // Fully partition the shift register so all 16 shifts happen in 1 clock cycle
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=1
    
    for (int k = 0; k < 16; k++) {
        #pragma HLS UNROLL
        shift_reg[k] = 0;
    }

    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        
        for (int k = 15; k > 0; k--) {
            #pragma HLS UNROLL
            shift_reg[k] = shift_reg[k - 1];
        }
        shift_reg[0] = in_stream.read();

        acc_t acc = 0;
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL
            acc += (acc_t)shift_reg[k] * taps[k];
        }
        
        acc_t drift_corr = (acc_t)shift_reg[15] * (acc_t)0.01;
        out_stream.write((data_t)(acc - drift_corr));
    }
}

// =========================================================================
// STAGE 3: Block Statistics (2-Pass Local BRAM Buffer)
// =========================================================================
void stage3_compute_stats(hls::stream<data_t>& in_stream, 
                          hls::stream<data_t>& out_stream,
                          hls::stream<stat_t>& out_mean, 
                          hls::stream<stat_t>& out_peak, 
                          hls::stream<stat_t>& out_mad) {
                          
    // Local BRAM to buffer the block since we need two passes (Mean, then MAD)
    data_t local_buf[BLOCK];
    
    for (int b = 0; b < (N / BLOCK); b++) {
        acc_t sum = 0;
        data_t peak = 0;
        
        // Pass 1: Stream in, find Mean and Peak, buffer data
        for (int i = 0; i < BLOCK; i++) {
            #pragma HLS PIPELINE II=1
            data_t val = in_stream.read();
            local_buf[i] = val;
            
            sum += (acc_t)val;
            data_t abs_val = (val < 0) ? (data_t)(-val) : val;
            if (abs_val > peak) peak = abs_val;
        }
        
        stat_t mean = (stat_t)(sum / (acc_t)BLOCK);
        
        // Pass 2: Read from local BRAM, find MAD, forward data
        acc_t mad_sum = 0;
        for (int i = 0; i < BLOCK; i++) {
            #pragma HLS PIPELINE II=1
            data_t val = local_buf[i];
            acc_t diff = (acc_t)val - (acc_t)mean;
            acc_t abs_diff = (diff < (acc_t)0) ? (acc_t)(-diff) : (acc_t)diff;
            mad_sum += abs_diff;
            
            out_stream.write(val); // Forward raw data to Stage 4
        }
        
        stat_t mad = (stat_t)(mad_sum / (acc_t)BLOCK);
        
        // Send statistics once per block
        out_mean.write(mean);
        out_peak.write((stat_t)peak);
        out_mad.write(mad);
    }
}

// =========================================================================
// STAGE 4: Dynamic Range Normalization
// =========================================================================
void stage4_normalize(hls::stream<data_t>& in_stream, 
                      hls::stream<stat_t>& in_mean, 
                      hls::stream<stat_t>& in_peak, 
                      hls::stream<stat_t>& in_mad, 
                      hls::stream<data_t>& out_stream) {
                      
    for (int b = 0; b < (N / BLOCK); b++) {
        // Read block statistics once
        stat_t mean = in_mean.read();
        stat_t peak = in_peak.read();
        stat_t mad  = in_mad.read();
        
        acc_t target_mad = 2.0;
        acc_t raw_gain = target_mad / ((acc_t)mad + (acc_t)0.1); 
        acc_t max_gain = 4.0;
        acc_t gain = (raw_gain > max_gain) ? max_gain : raw_gain;
        
        acc_t peak_suppression = (peak > (data_t)12.0) ? (acc_t)((acc_t)12.0 / (acc_t)peak) : (acc_t)1.0;
        acc_t final_gain = gain * peak_suppression;

        // Process the block
        for (int i = 0; i < BLOCK; i++) {
            #pragma HLS PIPELINE II=1
            data_t val = in_stream.read();
            acc_t centered = (acc_t)val - (acc_t)mean;
            out_stream.write((data_t)(centered * final_gain));
        }
    }
}

// =========================================================================
// STAGE 5: Multi-Zone Activation
// =========================================================================
void stage5_post_process(hls::stream<data_t>& in_stream, hls::stream<data_t>& out_stream) {
    const data_t DEAD_ZONE = 0.5;
    const acc_t  LINEAR_GAIN = 1.25;
    const data_t SAT_LIMIT = 8.0;

    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        data_t val = in_stream.read();
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
        
        if (processed_val > (data_t)10.0) processed_val = (data_t)10.0;
        if (processed_val < (data_t)(-10.0)) processed_val = (data_t)(-10.0);
        
        out_stream.write(processed_val);
    }
}

// =========================================================================
// TOP LEVEL MODULE
// =========================================================================
void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1
    #pragma HLS interface s_axilite port=return

    // Internal streaming FIFOs
    hls::stream<data_t> s_in("s_in");
    hls::stream<data_t> s1("s1");
    hls::stream<data_t> s2("s2");
    hls::stream<data_t> s3("s3");
    hls::stream<data_t> s4("s4");
    hls::stream<data_t> s_out("s_out");
    
    // Statistics streams
    hls::stream<stat_t> s_mean("s_mean");
    hls::stream<stat_t> s_peak("s_peak");
    hls::stream<stat_t> s_mad("s_mad");

    #pragma HLS stream depth=1024 variable=s3 
    #pragma HLS DATAFLOW

    // Connect the pipeline
    read_input(in, s_in);
    stage1_prescale(s_in, s1);
    stage2_fir_filter(s1, s2);
    stage3_compute_stats(s2, s3, s_mean, s_peak, s_mad);
    stage4_normalize(s3, s_mean, s_peak, s_mad, s4);
    stage5_post_process(s4, s_out);
    write_output(s_out, out);
}