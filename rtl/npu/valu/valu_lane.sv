// valu_lane: Single 8-bit ALU slice for NPU VALU
// Operations: 0=ADD, 1=SUB, 2=MUL, 3=MIN, 4=MAX, 5=AND, 6=OR, 7=XOR

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
            3'd2: result = $signed(a) * $signed(b);                // MUL (low 8 bits)
            3'd3: result = ($signed(a) < $signed(b)) ? a : b;      // MIN (signed)
            3'd4: result = ($signed(a) > $signed(b)) ? a : b;      // MAX (signed)
            3'd5: result = a & b;                                  // AND
            3'd6: result = a | b;                                  // OR
            3'd7: result = a ^ b;                                  // XOR
            default: result = 8'd0;
        endcase
    end

endmodule
