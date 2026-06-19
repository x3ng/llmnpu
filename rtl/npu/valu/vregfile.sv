// vregfile: 32 x (64 x 8-bit) register file for NPU VALU
// Packed array logic [31:0][63:0][7:0] — 2 read ports, 1 write port

module vregfile (
    input  logic            clk,
    // Read port 1
    input  logic [4:0]      rs1_addr,
    output logic [63:0][7:0] rs1_data,
    // Read port 2
    input  logic [4:0]      rs2_addr,
    output logic [63:0][7:0] rs2_data,
    // Write port
    input  logic [4:0]      rd_addr,
    input  logic [63:0][7:0] rd_data,
    input  logic            rd_wen
);

    logic [31:0][63:0][7:0] regs;

    // Write: edge-triggered
    always_ff @(posedge clk) begin
        if (rd_wen)
            regs[rd_addr] <= rd_data;
    end

    // Read: combinational
    assign rs1_data = regs[rs1_addr];
    assign rs2_data = regs[rs2_addr];

endmodule
