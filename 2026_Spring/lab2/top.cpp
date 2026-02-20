#include "dcl.h"

// Set Unroll Factor (Experiment with 2, 4, 8, or 16 depending on available LUTs/DSPs)
#define UF 16

// -------------------------------------------------------------------------
// Internal Compute Kernel
// -------------------------------------------------------------------------
static void compute_stencil(data_t in_grid[NX][NY], data_t out_grid[NX][NY]) {
    
    data_t line_buf[2][NY];
    // Partition line_buf cyclically on the column dimension to allow multiple reads/writes
    #pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=UF dim=2
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS DEPENDENCE variable=line_buf inter false

    data_t window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    Row_Loop: for (int i = 0; i < NX + 1; i++) {
        Col_Loop: for (int j = 0; j < NY + 1; j++) {
            #pragma HLS PIPELINE II=1
            // Unroll the inner loop to process 'UF' pixels per clock cycle
            #pragma HLS UNROLL factor=UF
            #pragma HLS LOOP_FLATTEN off

            // --- DATA READ STAGE ---
            data_t val_in = 0;
            if (i < NX && j < NY) {
                val_in = in_grid[i][j];
            }

            // --- WINDOW SHIFT STAGE ---
            for (int r = 0; r < 3; r++) {
                window[r][0] = window[r][1];
                window[r][1] = window[r][2];
            }

            window[0][2] = line_buf[0][j];
            window[1][2] = line_buf[1][j];
            window[2][2] = val_in;

            if (j < NY) {
                line_buf[0][j] = line_buf[1][j];
                line_buf[1][j] = val_in;
            }

            // --- COMPUTE STAGE ---
            int r = i - 1; 
            int c = j - 1; 

            if (r >= 0 && r < NX && c >= 0 && c < NY) {
                data_t result;

                if (r == 0 || r == NX - 1 || c == 0 || c == NY - 1) {
                    result = window[1][1];
                } 
                else {
                    acc_t sum_axis = (acc_t)window[0][1] + (acc_t)window[2][1] + 
                                     (acc_t)window[1][0] + (acc_t)window[1][2];

                    acc_t sum_diag = (acc_t)window[0][0] + (acc_t)window[0][2] + 
                                     (acc_t)window[2][0] + (acc_t)window[2][2];

                    acc_t center   = (acc_t)window[1][1];

                    result = (data_t)(wc * center + wa * sum_axis + wd * sum_diag);
                }

                out_grid[r][c] = result;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Top Level Function
// -------------------------------------------------------------------------
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    #pragma HLS interface m_axi port=A_in offset=slave bundle=gmem0 depth=NX*NY max_read_burst_length=256
    #pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=NX*NY max_write_burst_length=256
    #pragma HLS interface s_axilite port=return

    data_t buf0[NX][NY];
    data_t buf1[NX][NY];
    
    // Cyclically partition the ping-pong buffers so we can read/write UF elements per cycle
    #pragma HLS ARRAY_PARTITION variable=buf0 cyclic factor=UF dim=2
    #pragma HLS ARRAY_PARTITION variable=buf1 cyclic factor=UF dim=2
    #pragma HLS BIND_STORAGE variable=buf0 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=buf1 type=ram_2p impl=bram

    // Cast the 2D arrays to 1D 512-bit wide pointers
    // Note: This assumes NX*NY is cleanly divisible by 16, and data_t is 32 bits.
    const ap_uint<512>* A_in_wide = (const ap_uint<512>*)A_in;
    ap_uint<512>* A_out_wide = (ap_uint<512>*)A_out;

    int total_chunks = (NX * NY) / 16;

    // ---------------------------------------------------------
    // 1. Explicitly Vectorized Load Loop
    // ---------------------------------------------------------
    Load_Loop: for(int chunk = 0; chunk < total_chunks; chunk++) {
        #pragma HLS PIPELINE II=1
        
        // Read 512 bits (16 elements) in ONE clock cycle
        ap_uint<512> wide_access = A_in_wide[chunk];

        // Unpack the 16 elements
        Unpack_Loop: for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL
            
            // Slice out 32 bits at a time
            unsigned int raw_bits = wide_access.range(32 * (k + 1) - 1, 32 * k);
            
            // Bit-cast the raw integer bits back to your data_t (e.g., float)
            data_t val = *(data_t*)(&raw_bits); 

            // Calculate 2D indices from the 1D chunk index
            int flat_idx = chunk * 16 + k;
            int i = flat_idx / NY;
            int j = flat_idx % NY;
            
            buf0[i][j] = val;
        }
    }

    // 2. Time Stepping
    Time_Loop: for (int t = 0; t < TSTEPS; t++) {
        if (t % 2 == 0) {
            compute_stencil(buf0, buf1);
        } else {
            compute_stencil(buf1, buf0);
        }
    }

    // ---------------------------------------------------------
    // 3. Explicitly Vectorized Store Loop
    // ---------------------------------------------------------
    data_t (*final_src)[NY] = (TSTEPS % 2 == 0) ? buf0 : buf1;

    Store_Loop: for(int chunk = 0; chunk < total_chunks; chunk++) {
        #pragma HLS PIPELINE II=1
        
        ap_uint<512> wide_write = 0;

        // Pack the 16 elements
        Pack_Loop: for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL
            
            int flat_idx = chunk * 16 + k;
            int i = flat_idx / NY;
            int j = flat_idx % NY;
            
            data_t val = final_src[i][j];
            
            // Bit-cast the data_t to raw integer bits
            unsigned int raw_bits = *(unsigned int*)(&val);
            
            // Insert the 32 bits into the 512-bit word
            wide_write.range(32 * (k + 1) - 1, 32 * k) = raw_bits;
        }

        // Write 512 bits in ONE clock cycle
        A_out_wide[chunk] = wide_write;
    }
}