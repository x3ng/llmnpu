// IFetch: Instruction Fetch unit.
// PC register (8-bit, 256 instructions max in 8KB I-SRAM).
// Sequential increment (no branches initially, simplified).
// Stalled by dispatch backpressure (stall signal).
// Reset: PC = 0.

`include "isa_defines.svh"
`include "npu_defines.svh"

module ifetch (
    input  logic        clk,
    input  logic        rst_n,

    // Stall from dispatch backpressure
    input  logic        stall,

    // Program counter / fetched instruction
    output logic [7:0]  pc,
    output logic [31:0] instr,
    output logic        instr_valid,

    // I-SRAM load port (for testbench / microcode loading)
    input  logic        load_en,
    input  logic [7:0]  load_addr,
    input  logic [31:0] load_data
);

    // Instruction memory: 256 words x 32 bits
    reg [31:0] imem [0:255];

    // Load port: synchronous write, independent of stall / reset
    always_ff @(posedge clk) begin
        if (load_en)
            imem[load_addr] <= load_data;
    end

    // PC register: sequential increment, reset to 0
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pc <= 8'd0;
        else if (!stall)
            pc <= pc + 8'd1;
    end

    // Combinational read from instruction memory.
    // instr_valid is always high -- the pipeline stage (id_valid in dispatch)
    // gates further progress when stalled.
    assign instr       = imem[pc];
    assign instr_valid = 1'b1;

endmodule
