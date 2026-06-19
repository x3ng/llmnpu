// valu_lane: Single 8-bit ALU slice for NPU VALU
// Operations: 0=ADD, 1=SUB, 2=MUL (lower nibble), 3=MIN (signed), 4=MAX (signed)

module valu_lane (
    input  logic [2:0]  op,
    input  logic [7:0]  a,
    input  logic [7:0]  b,
    output logic [7:0]  result
);

    always_comb begin
        unique case (op)
            3'd0: result = $signed(a) + $signed(b);                // ADD
            3'd1: result = $signed(a) - $signed(b);                // SUB
            3'd2: result = a[3:0] * b[3:0];                        // MUL (lower nibble)
            3'd3: result = ($signed(a) < $signed(b)) ? a : b;      // MIN (signed)
            3'd4: result = ($signed(a) > $signed(b)) ? a : b;      // MAX (signed)
            default: result = 8'd0;
        endcase
    end

endmodule
