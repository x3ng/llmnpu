// Systolic Array: 16×16 PE mesh with broadcast B and broadcast A.
//
// Architecture (different from canonical systolic shift):
//   - B: b_in[c] is broadcast to ALL rows simultaneously (LOAD_B = 1 cycle).
//        Each PE in column c receives the same b_in[c] during the LOAD_B state.
//   - A: a_in[r] is broadcast to ALL PEs in row r simultaneously.
//        All PEs in a row see the same activation.
//   - Both A and B are stationary inside each PE after loading.
//
// Pipeline: IDLE → LOAD_B(1) → PREFETCH(1) → COMPUTE(K) → REDUCE(1) → WRITEBACK(1)

module systolic_array #(
    parameter int ROWS  = 16,
    parameter int COLS  = 16,
    parameter int K_MAX = 256
) (
    input  logic                        clk,
    input  logic                        rst_n,

    // A input (ROWS × 8-bit): broadcast to all PEs in each row
    input  logic [ROWS-1:0][7:0]        a_in,
    input  logic                        a_valid,

    // B input (COLS × 8-bit): broadcast to all rows simultaneously
    input  logic [COLS-1:0][7:0]        b_in,

    // Control
    input  logic                        start,
    input  logic [7:0]                  k_count,

    // Status
    output logic                        busy,
    output logic                        done,

    // Output (ROWS × COLS × 16-bit partial sums)
    output logic [ROWS-1:0][COLS-1:0][15:0] psum_out,
    output logic                             psum_valid
);

    // ── Controller ────────────────────────────────────────────────────
    logic        load_b, clear_acc;
    logic [2:0]  ctrl_state;

    gemm_controller ctrl (
        .clk, .rst_n, .start, .k_count,
        .load_b    (load_b),
        .clear_acc (clear_acc),
        .done      (done),
        .busy      (busy),
        .state_out (ctrl_state)
    );

    // ── PE internal wires ─────────────────────────────────────────────
    logic [ROWS-1:0][COLS-1:0][31:0] pe_sum;
    logic [ROWS-1:0][COLS-1:0][7:0]  a_term;   // terminate unused PE a_out
    logic [ROWS-1:0][COLS-1:0][7:0]  b_term;   // terminate unused PE b_out

    // PE valid: only during COMPUTE state AND when A data is valid
    logic pe_valid;
    assign pe_valid = a_valid && (ctrl_state == 3'd3);   // 3'd3 = COMPUTE

    // ── 16×16 PE mesh ────────────────────────────────────────────────
    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : pe_row
            for (c = 0; c < COLS; c++) begin : pe_col
                pe #(.K_MAX(K_MAX)) pe_inst (
                    .clk,
                    .rst_n,
                    .a_in      (a_in[r]),       // broadcast: same A to every PE in row r
                    .a_out     (a_term[r][c]),  // terminated (unused in broadcast arch)
                    .b_in      (b_in[c]),       // broadcast: b_in[c] to ALL rows
                    .b_out     (b_term[r][c]),  // terminated (unused in broadcast arch)
                    .load_b    (load_b),
                    .clear_acc (clear_acc),
                    .valid_in  (pe_valid),
                    .psum_out  (pe_sum[r][c])
                );
            end
        end
    endgenerate

    // ── Reduction: INT32 → INT16 saturation ──────────────────────────
    //   Use generate-for with genvar for iverilog compatibility
    //   ($signed() is broken in iverilog — use logic signed instead)
    genvar ri, ci;
    generate
        for (ri = 0; ri < ROWS; ri++) begin : red_row
            for (ci = 0; ci < COLS; ci++) begin : red_col
                logic signed [31:0] s;
                assign s = pe_sum[ri][ci];
                always_ff @(posedge clk) begin
                    if (ctrl_state == 3'd5) begin          // WRITEBACK (PE psum_out lags acc by 1 cycle)
                        if (s > 32767)
                            psum_out[ri][ci] <= 16'h7FFF;
                        else if (s < -32768)
                            psum_out[ri][ci] <= 16'h8000;
                        else
                            psum_out[ri][ci] <= s[15:0];
                    end
                end
            end
        end
    endgenerate

    // ── Output valid strobe ──────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            psum_valid <= 1'b0;
        else
            psum_valid <= (ctrl_state == 3'd5);   // WRITEBACK
    end

endmodule
