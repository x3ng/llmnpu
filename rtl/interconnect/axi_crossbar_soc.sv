// ============================================================
// axi_crossbar_soc.sv — SoC Address Decoder + Data-Path Mux
//
// Connects PicoRV32 slave ports to:
//   Peripherals → Boot trampoline + UART
//   NPU CSR     → npu_top CSR MMIO
//   NPU I-SRAM  → internal 8 KB SRAM
//   NPU V-SRAM  → internal 64 KB SRAM
//   ExtMem      → ext_mem_model (64 MB DRAM)
// ============================================================

`include "npu_defines.svh"

module axi_crossbar_soc (
    input  logic        clk,
    input  logic        rst_n,

    // ============================================================
    // PicoRV32 slave ports (from picorv32_wrapper)
    // ============================================================

    // ---- Peripherals (0x0xxx_xxxx) ----
    input  logic        periph_req,
    input  logic        periph_write,
    input  logic [31:0] periph_addr,
    input  logic [31:0] periph_wdata,
    input  logic [ 3:0] periph_wstrb,
    output logic [31:0] periph_rdata,
    output logic        periph_ready,

    // ---- NPU MMIO CSR (0x1000_0xxx) ----
    input  logic        npu_csr_req,
    input  logic        npu_csr_write,
    input  logic [11:0] npu_csr_addr,
    input  logic [31:0] npu_csr_wdata,
    output logic [31:0] npu_csr_rdata,
    output logic        npu_csr_ready,

    // ---- NPU I-SRAM (0x1001_xxxx, 8 KB) ----
    input  logic        npu_isram_req,
    input  logic        npu_isram_write,
    input  logic [12:0] npu_isram_addr,
    input  logic [31:0] npu_isram_wdata,
    input  logic [ 3:0] npu_isram_wstrb,
    output logic [31:0] npu_isram_rdata,
    output logic        npu_isram_ready,

    // ---- NPU V-SRAM (0x1002_xxxx, 64 KB) ----
    input  logic        npu_vsram_req,
    input  logic        npu_vsram_write,
    input  logic [15:0] npu_vsram_addr,
    input  logic [31:0] npu_vsram_wdata,
    input  logic [ 3:0] npu_vsram_wstrb,
    output logic [31:0] npu_vsram_rdata,
    output logic        npu_vsram_ready,

    // ---- ExtMem DRAM (0x40xx_xxxx) ----
    input  logic        extmem_req,
    input  logic        extmem_write,
    input  logic [31:0] extmem_addr,
    input  logic [31:0] extmem_wdata,
    input  logic [ 3:0] extmem_wstrb,
    output logic [31:0] extmem_rdata,
    output logic        extmem_ready,

    // ============================================================
    // NPU Top CSR interface (driven by crossbar)
    // ============================================================
    output logic [11:0] csr_addr,
    output logic [31:0] csr_wdata,
    output logic        csr_we,
    output logic        csr_re,
    input  logic [31:0] csr_rdata,

    // ============================================================
    // ExtMem model interface (driven by crossbar)
    // ============================================================
    output logic        mem_req,
    output logic        mem_write,
    output logic [23:0] mem_addr,     // word address (0 .. 16M-1)
    output logic [31:0] mem_wdata,
    output logic [ 3:0] mem_wstrb,
    input  logic [31:0] mem_rdata,
    input  logic        mem_ready,

    // ============================================================
    // UART TX debug output
    // ============================================================
    output logic [7:0]  uart_tx
);

    // ============================================================
    // Peripherals — Boot trampoline + UART
    // ============================================================
    // Boot trampoline: maps the two initial instructions so the core
    // jumps from 0x00000000 to 0x40000000 (ExtMem).
    //   lui  ra, 0x40000   → 0x400000B7
    //   jalr zero, 0(ra)   → 0x00008067
    // UART:  addr 0x08 = TX data (write), addr 0x0C = TX status (read)

    always_comb begin
        periph_ready = 1'b0;
        periph_rdata = 32'd0;

        if (periph_req) begin
            unique case (periph_addr)
                32'h00000000: begin
                    periph_rdata = 32'h400000B7;
                    periph_ready = 1'b1;
                end
                32'h00000004: begin
                    periph_rdata = 32'h00008067;
                    periph_ready = 1'b1;
                end
                32'h00000008: begin
                    // UART TX data register: read returns 0
                    periph_rdata = 32'd0;
                    periph_ready = 1'b1;
                end
                32'h0000000C: begin
                    // UART TX status: always ready
                    periph_rdata = 32'h00000001;
                    periph_ready = 1'b1;
                end
                default: begin
                    // Unmapped periph region — respond 0
                    periph_rdata = 32'd0;
                    periph_ready = 1'b1;
                end
            endcase
        end
    end

    // UART TX capture
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            uart_tx <= 8'd0;
        else if (periph_req && periph_write && (periph_addr == 32'h00000008))
            uart_tx <= periph_wdata[7:0];
    end

    // ============================================================
    // NPU CSR Bridge
    //
    // picorv32_wrapper uses req/write; npu_top uses we/re.
    //   csr_we = req &&  write
    //   csr_re = req && !write
    // ============================================================
    assign csr_addr  = npu_csr_addr;
    assign csr_wdata = npu_csr_wdata;
    assign csr_we    = npu_csr_req &&  npu_csr_write;
    assign csr_re    = npu_csr_req && !npu_csr_write;

    // CSR reads are combinational — respond immediately
    assign npu_csr_rdata = csr_rdata;
    assign npu_csr_ready = npu_csr_req;   // 0-cycle response

    // ============================================================
    // NPU I-SRAM — 8 KB internal SRAM (2048 × 32-bit words)
    // ============================================================
    localparam int ISRAM_WORDS = `ISRAM_SIZE / 4;   // 2048

    (* ram_style = "block" *) reg [31:0] isram [0:ISRAM_WORDS-1];

    // Write (clocked, byte-strobed)
    always_ff @(posedge clk) begin
        if (npu_isram_req && npu_isram_write) begin
            if (npu_isram_wstrb[0])
                isram[npu_isram_addr[12:2]][ 7: 0] <= npu_isram_wdata[ 7: 0];
            if (npu_isram_wstrb[1])
                isram[npu_isram_addr[12:2]][15: 8] <= npu_isram_wdata[15: 8];
            if (npu_isram_wstrb[2])
                isram[npu_isram_addr[12:2]][23:16] <= npu_isram_wdata[23:16];
            if (npu_isram_wstrb[3])
                isram[npu_isram_addr[12:2]][31:24] <= npu_isram_wdata[31:24];
        end
    end

    // Read — combinational (0-cycle latency)
    assign npu_isram_rdata = (npu_isram_req && !npu_isram_write)
                           ? isram[npu_isram_addr[12:2]] : 32'd0;
    assign npu_isram_ready = npu_isram_req;

    // ============================================================
    // NPU V-SRAM — 64 KB internal SRAM (16384 × 32-bit words)
    // ============================================================
    localparam int VSRAM_WORDS = `WSRAM_SIZE / 4;   // 16384

    (* ram_style = "block" *) reg [31:0] vsram [0:VSRAM_WORDS-1];

    // Write (clocked, byte-strobed)
    always_ff @(posedge clk) begin
        if (npu_vsram_req && npu_vsram_write) begin
            if (npu_vsram_wstrb[0])
                vsram[npu_vsram_addr[15:2]][ 7: 0] <= npu_vsram_wdata[ 7: 0];
            if (npu_vsram_wstrb[1])
                vsram[npu_vsram_addr[15:2]][15: 8] <= npu_vsram_wdata[15: 8];
            if (npu_vsram_wstrb[2])
                vsram[npu_vsram_addr[15:2]][23:16] <= npu_vsram_wdata[23:16];
            if (npu_vsram_wstrb[3])
                vsram[npu_vsram_addr[15:2]][31:24] <= npu_vsram_wdata[31:24];
        end
    end

    // Read — combinational (0-cycle latency)
    assign npu_vsram_rdata = (npu_vsram_req && !npu_vsram_write)
                           ? vsram[npu_vsram_addr[15:2]] : 32'd0;
    assign npu_vsram_ready = npu_vsram_req;

    // ============================================================
    // ExtMem DRAM — Route to ext_mem_model with address translation
    //
    // CPU address  0x40000000 → DRAM word address 0
    // CPU address  0x40000004 → DRAM word address 1
    // ...
    // CPU address  0x43FFFFFC → DRAM word address 0x00FFFFFF
    // ============================================================
    wire [31:0] extmem_offset = extmem_addr - 32'h4000_0000;
    wire [23:0] extmem_word   = extmem_offset[25:2];  // 24-bit word address

    assign mem_req   = extmem_req;
    assign mem_write = extmem_write;
    assign mem_addr  = extmem_word;
    assign mem_wdata = extmem_wdata;
    assign mem_wstrb = extmem_wstrb;

    assign extmem_rdata = mem_rdata;
    assign extmem_ready = mem_ready;

endmodule
