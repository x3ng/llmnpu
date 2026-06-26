// ============================================================
// lut_engine — 32-entry parallel LUT with linear interpolation
//
// Implements GELU, Sigmoid and Tanh activation functions using
// a shared 32-entry lookup table + piecewise-linear interpolator.
//
// Architecture:
//   Cycle 0 — register inputs (func_sel, x_in)
//   Cycle 1 — index computation, LUT read, interpolation, output
//
// Index:  idx  = (x_in + 128) >> 3   →  0..31
//         frac = (x_in + 128)[2:0]    →  0..7
//
// Interpolation:  y = lut[idx] + (lut[idx+1] - lut[idx]) * frac / 8
//
// LUT ROMs are initialised via $readmemh from hex files generated
// by tools/codegen/generate_luts.py.
// ============================================================

module lut_engine #(
    parameter GELU_FILE    = "../synth/luts/gelu.hex",
    parameter SIGMOID_FILE = "../synth/luts/sigmoid.hex",
    parameter TANH_FILE    = "../synth/luts/tanh.hex"
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [1:0]  func_sel,             // 0=GELU, 1=Sigmoid, 2=Tanh
    input  logic [7:0]  x_in,                 // signed INT8
    input  logic        valid_in,
    output logic [7:0]  y_out,                // signed INT8
    output logic        valid_out
);

    // ------------------------------------------------------------------
    // LUT ROMs — 3 functions × 32 entries × 8-bit
    // ------------------------------------------------------------------
    logic [7:0] lut_gelu    [0:31];
    logic [7:0] lut_sigmoid [0:31];
    logic [7:0] lut_tanh    [0:31];

    initial begin
        $readmemh(GELU_FILE,    lut_gelu);
        $readmemh(SIGMOID_FILE, lut_sigmoid);
        $readmemh(TANH_FILE,    lut_tanh);
    end

    // ------------------------------------------------------------------
    // Pipeline stage 0 — input sampling
    // ------------------------------------------------------------------
    logic [1:0]  func_sel_r;
    logic [7:0]  x_r;
    logic        valid_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            func_sel_r <= 2'd0;
            x_r        <= 8'd0;
            valid_r    <= 1'b0;
        end else begin
            func_sel_r <= func_sel;
            x_r        <= x_in;
            valid_r    <= valid_in;
        end
    end

    // ------------------------------------------------------------------
    // Index and fractional part  (combinational from x_r)
    // ------------------------------------------------------------------
    // 8-bit wrap of (x_r + 128) is intentional:
    //   x_r = 0x80 (-128 signed)  →  0x80 + 0x80 = 0x100 → 0x00  (idx=0)
    //   x_r = 0x00 (   0 signed)  →  0x00 + 0x80 = 0x80       (idx=16)
    //   x_r = 0x7f (+127 signed)  →  0x7f + 0x80 = 0xff       (idx=31)
    logic [7:0] offset;          // 0..255  (unsigned view)
    logic [4:0] idx;             // 0..31
    logic [2:0] frac;            // 0..7

    assign offset = x_r + 8'd128;
    assign idx    = offset[7:3];
    assign frac   = offset[2:0];

    // ------------------------------------------------------------------
    // LUT read  (combinational from func_sel_r, idx)
    // ------------------------------------------------------------------
    logic signed [7:0] lut_val;   // lut[idx]
    logic signed [7:0] lut_next;  // lut[min(idx+1, 31)]

    always_comb begin
        unique case (func_sel_r)
            2'd0: begin
                lut_val  = $signed(lut_gelu[idx]);
                lut_next = $signed(lut_gelu[idx < 31 ? idx + 1 : 31]);
            end
            2'd1: begin
                lut_val  = $signed(lut_sigmoid[idx]);
                lut_next = $signed(lut_sigmoid[idx < 31 ? idx + 1 : 31]);
            end
            2'd2: begin
                lut_val  = $signed(lut_tanh[idx]);
                lut_next = $signed(lut_tanh[idx < 31 ? idx + 1 : 31]);
            end
            default: begin
                lut_val  = 8'sd0;
                lut_next = 8'sd0;
            end
        endcase
    end

    // ------------------------------------------------------------------
    // Pipeline stage 1 — linear interpolation + output register
    // ------------------------------------------------------------------
    //  y = lut_val + (lut_next - lut_val) * frac / 8
    //
    // Intermediate widths:
    //   diff   : signed  9-bit  ( -255 .. 255 )
    //   prod   : signed 12-bit  ( -1785 .. 1785 )
    //   result : signed  9-bit
    // ------------------------------------------------------------------
    logic signed [8:0]  interp_diff;
    logic signed [11:0] interp_prod;
    logic signed [8:0]  interp_result;

    always_comb begin
        interp_diff   = $signed(lut_next) - $signed(lut_val);
        interp_prod   = interp_diff * $signed({4'b0, frac});
        interp_result = $signed(lut_val) + $signed(interp_prod >>> 3);
    end

    logic [7:0]  y_r;
    logic        valid_r2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_r      <= 8'd0;
            valid_r2 <= 1'b0;
        end else begin
            valid_r2 <= valid_r;
            if (valid_r) begin
                // Saturate to INT8 [-128, 127]
                // NOTE: $signed() is essential here — plain integer literals
                // 127 / -128 ensure correct signed comparison width.
                if ($signed(interp_result) > 127)
                    y_r <= 8'd127;
                else if ($signed(interp_result) < -128)
                    y_r <= 8'd128;           // 0x80 = -128 in two's complement
                else
                    y_r <= interp_result[7:0];
            end
        end
    end

    assign y_out    = y_r;
    assign valid_out = valid_r2;

endmodule
