#include "dcl.h"
#include <hls_stream.h>
#include <ap_int.h>

// Vector size 32 is required to hit ~2000 cycles for N=65536
#define VEC_SIZE 32
typedef ap_uint<VEC_SIZE * 32> bus_t;

struct vec_t {
    data_t d[VEC_SIZE];
};

// -------------------------------------------------------------------------
// STAGE 1: Load and Split (Fused with K0 logic)
// -------------------------------------------------------------------------
void load_and_split(const bus_t* in_bus, hls::stream<vec_t>& out_k1, hls::stream<vec_t>& out_k2) {
    const coef_t alpha = (coef_t)0.875, beta = (coef_t)0.125;
    
    for (int i = 0; i < N / VEC_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        bus_t raw = in_bus[i];
        vec_t v;
        for (int k = 0; k < VEC_SIZE; k++) {
            #pragma HLS UNROLL
            ap_uint<32> bits = raw.range(k * 32 + 31, k * 32);
            data_t val = reinterpret_cast<data_t&>(bits);
            // Apply K0 Preprocess immediately
            v.d[k] = (data_t)((acc_t)alpha * (acc_t)val + (acc_t)beta);
        }
        out_k1.write(v);
        out_k2.write(v);
    }
}

// -------------------------------------------------------------------------
// STAGE 2: K1 Transform (Fast Path)
// -------------------------------------------------------------------------
void transform_k1(hls::stream<vec_t>& in_stream, hls::stream<vec_t>& out_stream) {
    const coef_t w0 = (coef_t)0.50, w1 = (coef_t)(-0.25), w2 = (coef_t)0.125;
    data_t prev1 = 0, prev2 = 0;

    for (int i = 0; i < N / VEC_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        vec_t in_v = in_stream.read();
        vec_t out_v;
        for (int k = 0; k < VEC_SIZE; k++) {
            #pragma HLS UNROLL
            data_t x0 = in_v.d[k];
            data_t x1 = (k >= 1) ? in_v.d[k - 1] : prev1;
            data_t x2 = (k >= 2) ? in_v.d[k - 2] : ((k == 1) ? prev1 : prev2);

            // Pipelined multipliers for 10ns timing
            acc_t p0 = (acc_t)w0 * (acc_t)x0;
            acc_t p1 = (acc_t)w1 * (acc_t)x1;
            acc_t p2 = (acc_t)w2 * (acc_t)x2;
            #pragma HLS bind_op variable=p0 op=mul impl=dsp latency=3
            #pragma HLS bind_op variable=p1 op=mul impl=dsp latency=3
            #pragma HLS bind_op variable=p2 op=mul impl=dsp latency=3

            acc_t acc = p0 + p1 + p2;
            data_t y = (data_t)acc;
            data_t abs_y = (y < 0) ? (data_t)(-y) : y;
            out_v.d[k] = (abs_y > (data_t)7.5) ? (data_t)7.5 : abs_y;
        }
        prev2 = in_v.d[VEC_SIZE - 2];
        prev1 = in_v.d[VEC_SIZE - 1];
        out_stream.write(out_v);
    }
}

// -------------------------------------------------------------------------
// STAGE 3: K2 Block Statistics (Slow Path)
// -------------------------------------------------------------------------
void compute_stat_k2(hls::stream<vec_t>& in_stream, hls::stream<stat_t>& inv_stat_out) {
    const stat_t eps = (stat_t)0.5;
    for (int b = 0; b < (N / BLOCK); b++) {
        acc_t sum_abs = 0;
        for (int i = 0; i < (BLOCK / VEC_SIZE); i++) {
            #pragma HLS PIPELINE II=1
            vec_t v = in_stream.read();
            acc_t local_v_sum = 0;
            for (int k = 0; k < VEC_SIZE; k++) {
                #pragma HLS UNROLL
                data_t val = v.d[k];
                local_v_sum += (val < 0) ? (acc_t)(-val) : (acc_t)val;
            }
            sum_abs += local_v_sum;
        }
        stat_t avg_abs = (stat_t)(sum_abs / (acc_t)BLOCK);
        
        stat_t inv_st;
        // Pipelined division for 10ns timing
        #pragma HLS bind_op variable=inv_st op=fdiv impl=fabric latency=14
        inv_st = (stat_t)((acc_t)1.0 / (acc_t)(avg_abs + eps));
        inv_stat_out.write(inv_st);
    }
}

// -------------------------------------------------------------------------
// STAGE 4: Join and Store (K3 + K4)
// -------------------------------------------------------------------------
void join_and_store(hls::stream<vec_t>& s1_in, hls::stream<stat_t>& inv_stat_in, bus_t* out_bus) {
    const coef_t gamma = (coef_t)1.25, delta = (coef_t)0.05;
    for (int b = 0; b < (N / BLOCK); b++) {
        stat_t inv_st = inv_stat_in.read();
        for (int i = 0; i < (BLOCK / VEC_SIZE); i++) {
            #pragma HLS PIPELINE II=1
            vec_t in_v = s1_in.read();
            bus_t out_raw;
            for (int k = 0; k < VEC_SIZE; k++) {
                #pragma HLS UNROLL
                acc_t s3 = (acc_t)in_v.d[k] * (acc_t)inv_st;
                #pragma HLS bind_op variable=s3 op=mul impl=dsp latency=3
                
                data_t z = (data_t)((acc_t)gamma * s3 + (acc_t)delta);
                if (z < 0) z = 0; if (z > (data_t)7.9) z = (data_t)7.9;
                
                ap_uint<32> bits = reinterpret_cast<ap_uint<32>&>(z);
                out_raw.range(k * 32 + 31, k * 32) = bits;
            }
            out_bus[b * (BLOCK / VEC_SIZE) + i] = out_raw;
        }
    }
}

// -------------------------------------------------------------------------
// TOP KERNEL
// -------------------------------------------------------------------------
void top_kernel(const data_t in[N], data_t out[N]) {
    #pragma HLS interface m_axi port=in  offset=slave bundle=gmem0 max_widen_bitwidth=1024
    #pragma HLS interface m_axi port=out offset=slave bundle=gmem1 max_widen_bitwidth=1024
    #pragma HLS interface s_axilite port=return

    hls::stream<vec_t> s0_a, s0_b, s1;
    hls::stream<stat_t> inv_stats;

    // Depth must be > (BLOCK/VEC_SIZE) to prevent deadlock
    #pragma HLS STREAM variable=s1 depth=64
    #pragma HLS BIND_STORAGE variable=s1 type=ram_2p impl=bram

    #pragma HLS DATAFLOW
    load_and_split((const bus_t*)in, s0_a, s0_b);
    transform_k1(s0_a, s1);
    compute_stat_k2(s0_b, inv_stats);
    join_and_store(s1, inv_stats, (bus_t*)out);
}