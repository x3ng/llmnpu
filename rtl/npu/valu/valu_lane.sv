// valu_lane: Single 8-bit ALU slice for NPU VALU.
// CMOV uses rs1 as the per-lane condition and rs2 as the selected value.

`include "isa_defines.svh"

module valu_lane (
    input  logic [7:0]  op,
    input  logic [7:0]  a,
    input  logic [7:0]  b,
    output logic [7:0]  result
);

    always_comb begin
        unique case (op[3:0])
            4'h0: result = $signed(a) + $signed(b);
            4'h1: result = $signed(a) - $signed(b);
            4'h2: result = $signed(a) * $signed(b);
            4'h3: result = ($signed(a) < $signed(b)) ? a : b;
            4'h4: result = ($signed(a) > $signed(b)) ? a : b;
            4'h5: result = a & b;
            4'h6: result = a | b;
            4'h7: result = a ^ b;
            4'h8: result = a << b[2:0];
            4'h9: result = $signed(a) >>> b[2:0];
            4'hA: result = (a != 8'd0) ? b : 8'd0;
            default: result = 8'd0;
        endcase
    end

endmodule
