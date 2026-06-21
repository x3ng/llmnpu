// ============================================================
// quant_dequant — INT8 quantise / dequantise with aligned 3-stage pipeline
//
// Modes:
//   0 (dequant)  y = (x - zp) * scale_mul >> scale_shr
//   1 (quant)    y = clip(round(x * scale_mul >> scale_shr) + zp,
//                         -128, 127)
//
// Pipeline:
//   Cycle 0 — input sampling
//   Cycle 1 — multiply
//   Cycle 2 — shift + round (+ clip for quant)
//   Cycle 3 — valid alignment for registered consumers
// ============================================================

module quant_dequant (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        mode,               // 0=dequant, 1=quant
    input  logic [7:0]  x_in,               // signed INT8
    input  logic [7:0]  zp,                 // zero point  (unsigned)
    input  logic [15:0] scale_mul,          // multiplier  (signed)
    input  logic [7:0]  scale_shr,          // right-shift amount
    input  logic        valid_in,
    output logic [7:0]  y_out,              // signed INT8
    output logic        valid_out
);

    // ------------------------------------------------------------------
    // Pipeline stage 0 — input sampling
    // ------------------------------------------------------------------
    logic        mode_r;
    logic [7:0]  x_r, zp_r;
    logic [15:0] scale_mul_r;
    logic [7:0]  scale_shr_r;
    logic        valid_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_r       <= 1'b0;
            x_r          <= 8'd0;
            zp_r         <= 8'd0;
            scale_mul_r  <= 16'd0;
            scale_shr_r  <= 8'd0;
            valid_r      <= 1'b0;
        end else begin
            mode_r       <= mode;
            x_r          <= x_in;
            zp_r         <= zp;
            scale_mul_r  <= scale_mul;
            scale_shr_r  <= scale_shr;
            valid_r      <= valid_in;
        end
    end

    // ------------------------------------------------------------------
    // Pipeline stage 1 — multiply
    // ------------------------------------------------------------------
    // Dequant:  product = (x - zp) * scale_mul
    //   x - zp is sign-extended to 9 bits then zero-extended to 32 bits
    //   scale_mul is sign-extended to 32 bits
    //
    // Quant:  product = x * scale_mul
    //   x and scale_mul are sign-extended to 32 bits
    // ------------------------------------------------------------------
    logic signed [8:0]   dequant_diff;   // (x - zp), 9-bit signed
    logic signed [31:0]  next_product;

    always_comb begin
        dequant_diff = $signed({x_r[7], x_r}) - $signed({1'b0, zp_r});
        if (mode_r == 1'b0)
            next_product = dequant_diff * $signed(scale_mul_r);
        else
            next_product = $signed(x_r) * $signed(scale_mul_r);
    end

    logic signed [31:0] product;
    logic               product_mode;
    logic [7:0]         product_zp;
    logic [7:0]         product_scale_shr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product           <= 32'd0;
            product_mode      <= 1'b0;
            product_zp        <= 8'd0;
            product_scale_shr <= 8'd0;
        end else if (valid_r) begin
            product           <= next_product;
            product_mode      <= mode_r;
            product_zp        <= zp_r;
            product_scale_shr <= scale_shr_r;
        end
    end

    // ------------------------------------------------------------------
    // Pipeline stage 2 — shift + round + clip
    // ------------------------------------------------------------------
    // Combinational: compute shifted result + zero-point addition
    // ------------------------------------------------------------------
    logic signed [31:0] q_shifted;   // result after arithmetic shift
    logic signed [31:0] q_with_zp;   // result after adding zero-point

    always_comb begin
        if (product_mode == 1'b0) begin
            // Dequant:  y = product >> scale_shr  (no zp added per spec)
            q_shifted = product >>> product_scale_shr;
            q_with_zp = q_shifted;
        end else begin
            // Quant: round-to-nearest then shift
            if (product_scale_shr > 8'd0) begin
                if ($signed(product) >= 0)
                    q_shifted = (product + (32'sd1 << (product_scale_shr - 8'd1))) >>> product_scale_shr;
                else
                    q_shifted = (product + (32'sd1 << (product_scale_shr - 8'd1)) - 32'sd1) >>> product_scale_shr;
            end else begin
                q_shifted = product;
            end
            // Add zero point for quant mode
            q_with_zp = q_shifted + $signed({24'd0, product_zp});
        end
    end

    logic [7:0]  y_r;
    logic        valid_r2;
    logic        valid_r3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_r      <= 8'd0;
            valid_r2 <= 1'b0;
            valid_r3 <= 1'b0;
        end else begin
            valid_r2 <= valid_r;
            valid_r3 <= valid_r2;
            if (valid_r2) begin
                // Saturate to INT8 [-128, 127]
                if ($signed(q_with_zp) > 127)
                    y_r <= 8'd127;
                else if ($signed(q_with_zp) < -128)
                    y_r <= 8'd128;    // 0x80 = -128
                else
                    y_r <= q_with_zp[7:0];
            end
        end
    end

    // ------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------
    assign y_out    = y_r;
    assign valid_out = valid_r3;

endmodule
