// ============================================================
// ext_mem_model.sv — Behavioral 64 MB External DRAM Model
//
// Byte-addressable.  Simple SRAM-style interface:
//   req, write, addr[23:0] (word), wdata[31:0], wstrb[3:0],
//   rdata[31:0], ready
//
// Reads are combinational (0-cycle latency).
// Writes are clocked with byte strobes.
// Loads program from .hex file at init via $readmemh.
// ============================================================

`include "npu_defines.svh"

module ext_mem_model #(
    parameter string HEX_FILE = "sim/verilog/firmware.hex",
    parameter int    MEM_WORDS = 64 * 1024 * 1024 / 4    // 16 M words
) (
    input  logic        clk,
    input  logic        rst_n,

    // ---- SRAM-style interface (CPU / crossbar) ----
    input  logic        req,
    input  logic        write,
    input  logic [23:0] addr,       // word address (0 .. 16M-1)
    input  logic [31:0] wdata,
    input  logic [ 3:0] wstrb,      // byte write strobes
    output logic [31:0] rdata,
    output logic        ready,

    // ---- AXI bridge read port (combinational) ----
    input  logic [23:0] axi_rd_addr,
    input  logic        axi_rd_en,
    output logic [31:0] axi_rd_rdata,

    // ---- AXI bridge write port (clocked, full word) ----
    input  logic [23:0] axi_wr_addr,
    input  logic [31:0] axi_wr_wdata,
    input  logic        axi_wr_en
);

    // ================================================================
    // Memory array: 16 M × 32 bits = 64 MB
    // ================================================================
    (* ram_style = "block" *) reg [31:0] mem [0:MEM_WORDS-1];

    // ================================================================
    // Initialise from hex file
    // ================================================================
    initial begin
        // Zero-fill everything first
        integer i;
        for (i = 0; i < MEM_WORDS; i = i + 1)
            mem[i] = 32'd0;

        // Load firmware — non-fatal if file missing
        $readmemh(HEX_FILE, mem);
    end

    // ================================================================
    // Write (clocked, byte-strobed)
    // ================================================================
    always_ff @(posedge clk) begin
        if (req && write) begin
            if (wstrb[0]) mem[addr][ 7: 0] <= wdata[ 7: 0];
            if (wstrb[1]) mem[addr][15: 8] <= wdata[15: 8];
            if (wstrb[2]) mem[addr][23:16] <= wdata[23:16];
            if (wstrb[3]) mem[addr][31:24] <= wdata[31:24];
        end
    end

    // ================================================================
    // Read — combinational (0-cycle latency)
    // ================================================================
    assign rdata = (req && !write) ? mem[addr] : 32'd0;

    // ================================================================
    // Ready — asserted same cycle as request
    // ================================================================
    assign ready = req;

    // ================================================================
    // AXI bridge read port — combinational (0-cycle latency)
    // ================================================================
    assign axi_rd_rdata = axi_rd_en ? mem[axi_rd_addr] : 32'd0;

    // ================================================================
    // AXI bridge write port — clocked, full 32-bit word
    // ================================================================
    always_ff @(posedge clk) begin
        if (axi_wr_en)
            mem[axi_wr_addr] <= axi_wr_wdata;
    end

endmodule
