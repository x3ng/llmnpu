// ============================================================
// sfu_top — NPU Special Function Unit (SFU) top-level
//
// Integrates lut_engine (activation functions) and
// quant_dequant (INT8 quantise/dequantise).
//
// Opcode routing  (from isa_defines.svh):
//   OP_ACT_RELU    (0x20)  →  integer ReLU
//   OP_ACT_GELU    (0x21)  →  LUT engine, func_sel=0
//   OP_ACT_SIGMOID (0x22)  →  LUT engine, func_sel=1
//   OP_ACT_TANH    (0x23)  →  LUT engine, func_sel=2
//   OP_ACT_RELU6   (0x24)  →  integer clamp to [0, 6]
//   OP_ACT_CLIP    (0x25)  →  integer clamp to [0, zp]
//   OP_QUANT       (0x30)  →  quant_dequant, mode=1
//   OP_DEQUANT     (0x31)  →  quant_dequant, mode=0
//
// Pipeline depth:
//   LUT engines       : 2 cycles (input → index+lut → interp+output)
//   quant_dequant     : 3 cycles (input → multiply → shift+clip)
//   sfu_top           : 3 cycles (adds one mux stage)
// ============================================================

`include "isa_defines.svh"

module sfu_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [7:0]  opcode,
    input  logic [7:0]  x_in,
    input  logic [7:0]  zp,
    input  logic [15:0] scale_mul,
    input  logic [7:0]  scale_shr,
    input  logic        valid_in,
    output logic [7:0]  y_out,
    output logic        valid_out
);

    // ------------------------------------------------------------------
    // LUT engine path  (activation functions, 2-cycle pipeline)
    // ------------------------------------------------------------------
    logic        lut_valid;
    logic [7:0]  lut_y;
    logic [1:0]  func_sel;
    logic        lut_valid_in;
    logic        relu_valid_in;
    logic        qd_valid_in;

    assign relu_valid_in = valid_in && ((opcode == `OP_ACT_RELU) ||
                                        (opcode == `OP_ACT_RELU6) ||
                                        (opcode == `OP_ACT_CLIP));
    assign lut_valid_in  = valid_in && ((opcode == `OP_ACT_GELU) ||
                                        (opcode == `OP_ACT_SIGMOID) ||
                                        (opcode == `OP_ACT_TANH));
    assign qd_valid_in   = valid_in && ((opcode == `OP_QUANT) ||
                                        (opcode == `OP_DEQUANT));

    always_comb begin
        unique case (opcode)
            `OP_ACT_GELU    : func_sel = 2'd0;
            `OP_ACT_SIGMOID : func_sel = 2'd1;
            `OP_ACT_TANH    : func_sel = 2'd2;
            default         : func_sel = 2'd0;
        endcase
    end

    lut_engine #(
        .GELU_FILE   ("../synth/luts/gelu.hex"),
        .SIGMOID_FILE("../synth/luts/sigmoid.hex"),
        .TANH_FILE   ("../synth/luts/tanh.hex")
    ) u_lut (
        .clk      (clk),
        .rst_n    (rst_n),
        .func_sel (func_sel),
        .x_in     (x_in),
        .valid_in (lut_valid_in),
        .y_out    (lut_y),
        .valid_out(lut_valid)
    );

    // ------------------------------------------------------------------
    // ReLU/Clip path  (integer clamp, aligned to LUT path)
    // ------------------------------------------------------------------
    logic [7:0] relu_y;
    logic       relu_valid;
    logic [7:0] relu_y_aligned;
    logic       relu_valid_aligned;
    logic [7:0] relu_clip_max;

    always_comb begin
        unique case (opcode)
            `OP_ACT_RELU6: relu_clip_max = 8'd6;
            `OP_ACT_CLIP:  relu_clip_max = zp;
            default:       relu_clip_max = 8'h7F;
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            relu_y     <= 8'd0;
            relu_valid <= 1'b0;
        end else begin
            if (x_in[7])
                relu_y <= 8'd0;
            else if (x_in > relu_clip_max)
                relu_y <= relu_clip_max;
            else
                relu_y <= x_in;
            relu_valid <= relu_valid_in;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            relu_y_aligned     <= 8'd0;
            relu_valid_aligned <= 1'b0;
        end else begin
            relu_y_aligned     <= relu_y;
            relu_valid_aligned <= relu_valid;
        end
    end

    // ------------------------------------------------------------------
    // Quant/Dequant path  (3-cycle pipeline)
    // ------------------------------------------------------------------
    logic        qd_valid;
    logic [7:0]  qd_y;
    logic        qd_mode;
    logic        qd_mode_latched;

    assign qd_mode = qd_valid_in ? (opcode == `OP_QUANT) : qd_mode_latched;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            qd_mode_latched <= 1'b0;
        else if (qd_valid_in)
            qd_mode_latched <= (opcode == `OP_QUANT);
    end

    quant_dequant u_qd (
        .clk       (clk),
        .rst_n     (rst_n),
        .mode      (qd_mode),
        .x_in      (x_in),
        .zp        (zp),
        .scale_mul (scale_mul),
        .scale_shr (scale_shr),
        .valid_in  (qd_valid_in),
        .y_out     (qd_y),
        .valid_out (qd_valid)
    );

    // ------------------------------------------------------------------
    // Output mux + alignment pipeline stage
    //
    // The LUT engine (2 cycles) and quant_dequant (3 cycles) have
    // different latencies.  We align both paths to 3 cycles by
    // inserting a 1-cycle delay on the LUT path, then mux.
    //
    // Cycle counts from valid_in high:
    //   LUT path:   cycle 0 (input) → cycle 1 (lut) → cycle 2 (align) → out
    //   QD path:    cycle 0 (input) → cycle 1 (mult) → cycle 2 (shift) → out
    // ------------------------------------------------------------------
    logic [7:0]  lut_y_aligned;
    logic        lut_valid_aligned;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lut_y_aligned    <= 8'd0;
            lut_valid_aligned <= 1'b0;
        end else begin
            lut_y_aligned    <= lut_y;
            lut_valid_aligned <= lut_valid;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_out    <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            if (relu_valid_aligned) begin
                    y_out    <= relu_y_aligned;
                    valid_out <= 1'b1;
            end else if (lut_valid_aligned) begin
                    y_out    <= lut_y_aligned;
                    valid_out <= 1'b1;
            end else if (qd_valid) begin
                    y_out    <= qd_y;
                    valid_out <= 1'b1;
            end else begin
                    y_out    <= 8'd0;
                    valid_out <= 1'b0;
            end
        end
    end

endmodule
