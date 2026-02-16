#include "dcl.h"

// -------------------------------------------------------------------------
// Internal Compute Kernel
// Uses Line Buffering to read data ONCE per pixel instead of 9 times.
// -------------------------------------------------------------------------
static void compute_stencil(data_t in_grid[NX][NY], data_t out_grid[NX][NY]) {
    
    // 1. Line Buffer: Stores previous 2 rows to allow vertical sliding
    // Partitioning dim 1 allows simultaneous read/write of all rows
    data_t line_buf[2][NY];
    #pragma HLS ARRAY_PARTITION variable=line_buf dim=1 complete
    #pragma HLS DEPENDENCE variable=line_buf inter false

    // 2. Sliding Window: Stores 3x3 neighborhood in registers
    // Complete partitioning allows access to all 9 pixels in one cycle
    data_t window[3][3];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // Constant weights
    const data_t wc = (data_t)0.50;
    const data_t wa = (data_t)0.10;
    const data_t wd = (data_t)0.025;

    // 3. Main Loop
    // We iterate slightly past the bounds to account for the window latency.
    // When we read (i, j), we are actually computing the result for (i-1, j-1).
    Row_Loop: for (int i = 0; i < NX + 1; i++) {
        Col_Loop: for (int j = 0; j < NY + 1; j++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN off

            // --- DATA READ STAGE ---
            data_t val_in = 0;
            
            // Only read from memory if we are within the input grid bounds
            if (i < NX && j < NY) {
                val_in = in_grid[i][j];
            }

            // --- WINDOW SHIFT STAGE ---
            // Shift entire window left
            for (int r = 0; r < 3; r++) {
                window[r][0] = window[r][1];
                window[r][1] = window[r][2];
            }

            // Read from line buffer into right column of window
            window[0][2] = line_buf[0][j];
            window[1][2] = line_buf[1][j];
            window[2][2] = val_in;

            // Shift line buffer up and insert new value
            if (j < NY) {
                line_buf[0][j] = line_buf[1][j];
                line_buf[1][j] = val_in;
            }

            // --- COMPUTE STAGE ---
            // The window is fully valid for the pixel at (i-1, j-1).
            // We verify we are inside the valid computation region.
            // Since the window center is at [1][1], window[1][1] corresponds to in_grid[i-1][j-1].
            
            int r = i - 1; // Current Row being computed
            int c = j - 1; // Current Col being computed

            if (r >= 0 && r < NX && c >= 0 && c < NY) {
                
                data_t result;

                // Boundary Condition: Copy unchanged
                if (r == 0 || r == NX - 1 || c == 0 || c == NY - 1) {
                    result = window[1][1];
                } 
                // Interior: Apply Stencil
                else {
                    acc_t sum_axis = (acc_t)window[0][1] + (acc_t)window[2][1] + 
                                     (acc_t)window[1][0] + (acc_t)window[1][2];

                    acc_t sum_diag = (acc_t)window[0][0] + (acc_t)window[0][2] + 
                                     (acc_t)window[2][0] + (acc_t)window[2][2];

                    acc_t center   = (acc_t)window[1][1];

                    result = (data_t)(wc * center + wa * sum_axis + wd * sum_diag);
                }

                // Write output
                out_grid[r][c] = result;
            }
        }
    }
}

// -------------------------------------------------------------------------
// Top Level Function
// -------------------------------------------------------------------------
void top_kernel(const data_t A_in[NX][NY], data_t A_out[NX][NY]) {
    // AXI Interfaces
    #pragma HLS interface m_axi port=A_in offset=slave bundle=gmem0 depth=NX*NY
    #pragma HLS interface m_axi port=A_out offset=slave bundle=gmem1 depth=NX*NY
    #pragma HLS interface s_axilite port=return

    // Ping-Pong Buffers (On-Chip Memory)
    // If NX*NY is too large for BRAM, these must be partitioned or URAM used.
    data_t buf0[NX][NY];
    data_t buf1[NX][NY];
    
    // Partitioning dim 2 improves memory bandwidth for the stencil read logic
    // (Optional, dependent on specific FPGA resources, but usually helps)
    #pragma HLS BIND_STORAGE variable=buf0 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=buf1 type=ram_2p impl=bram

    // 1. Load Data (Burst Read)
    // We manually copy A_in to buf0 to start.
    Load_Loop: for(int i=0; i<NX; i++) {
        for(int j=0; j<NY; j++) {
            #pragma HLS PIPELINE II=1
            buf0[i][j] = A_in[i][j];
        }
    }

    // 2. Time Stepping (Ping-Pong)
    // No copying back! We just swap which buffer is Input and which is Output.
    Time_Loop: for (int t = 0; t < TSTEPS; t++) {
        // If t is even: Read buf0, Write buf1
        // If t is odd:  Read buf1, Write buf0
        if (t % 2 == 0) {
            compute_stencil(buf0, buf1);
        } else {
            compute_stencil(buf1, buf0);
        }
    }

    // 3. Write Data (Burst Write)
    // Copy the final result from the *last written buffer* to A_out.
    // If TSTEPS is even, the last write was to buf1 (at t=TSTEPS-1).
    // If TSTEPS is odd, the last write was to buf0.
    data_t (*final_src)[NY] = (TSTEPS % 2 == 0) ? buf0 : buf1;

    Store_Loop: for(int i=0; i<NX; i++) {
        for(int j=0; j<NY; j++) {
            #pragma HLS PIPELINE II=1
            A_out[i][j] = final_src[i][j];
        }
    }
}