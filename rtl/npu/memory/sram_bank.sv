// sram_bank.sv — Parameterized behavioral SRAM (yosys infers BRAM)
//
// Parameters:
//   DEPTH      — number of words
//   WIDTH      — word width in bits
//   READ_PORTS — 1 (Port A only) or 2 (Port A + Port B)
//   WRITE_PORTS — 1 (Port A only); 2 is accepted for future use
//
// Port A: read-write
// Port B: read-only, present when READ_PORTS >= 2
`timescale 1ns/1ps

module sram_bank #(
    parameter  int DEPTH       = 1024,
    parameter  int WIDTH       = 32,
    parameter  int READ_PORTS  = 1,
    parameter  int WRITE_PORTS = 1
) (
    input  logic                  clk,

    // ---- Port A (read-write) ----
    input  logic [$clog2(DEPTH)-1:0] addr,
    input  logic [WIDTH-1:0]         wdata,
    output logic [WIDTH-1:0]         rdata = 0,
    input  logic                     wen,   // write enable
    input  logic                     ren,   // read enable

    // ---- Port B (read-only, optional) ----
    input  logic [$clog2(DEPTH)-1:0] addr_b,
    output logic [WIDTH-1:0]         rdata_b = 0,
    input  logic                     ren_b
);

    localparam int ADDR_W = $clog2(DEPTH);

    // Memory array
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    // Port A: synchronous read / write
    always_ff @(posedge clk) begin
        if (wen) mem[addr] <= wdata;
        if (ren) rdata     <= mem[addr];
    end

    // Port B: synchronous read (optional second read port)
    generate
        if (READ_PORTS >= 2) begin : gen_port_b
            always_ff @(posedge clk) begin
                if (ren_b) rdata_b <= mem[addr_b];
            end
        end
    endgenerate

endmodule
