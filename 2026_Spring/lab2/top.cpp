#include "dcl.h"

// Set Unroll Factor (Experiment with 2, 4, 8, or 16 depending on available LUTs/DSPs)
#define UF 8

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

    // 1. Load Data (Burst Read Wide)
    Load_Loop_R: for(int i=0; i<NX; i++) {
        Load_Loop_C: for(int j=0; j<NY; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=UF
            buf0[i][j] = A_in[i][j];
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

    // 3. Write Data (Burst Write Wide)
    data_t (*final_src)[NY] = (TSTEPS % 2 == 0) ? buf0 : buf1;

    Store_Loop_R: for(int i=0; i<NX; i++) {
        Store_Loop_C: for(int j=0; j<NY; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS UNROLL factor=UF
            A_out[i][j] = final_src[i][j];
        }
    }
}