// ============================================================
// top_soc.sv — NPU SoC Top-Level Integration
//
// Instantiates:
//   picorv32_wrapper     — RISC-V control processor (RV32IM)
//   npu_top              — NPU datapath (GEMM, VALU, SFU, DMA)
//   axi_crossbar_soc     — Address decoder + bus router
//   ext_mem_model        — 64 MB behavioral DRAM
//
// Debug outputs:
//   uart_tx[7:0]         — 'P' on pass, 'F' on fail
// ============================================================

`include "npu_defines.svh"
`include "isa_defines.svh"

module top_soc (
    input  logic        clk,
    input  logic        rst_n,

    // ---- UART TX debug output ----
    output logic [7:0]  uart_tx
);

    // ============================================================
    // PicoRV32 ↔ Crossbar signals
    // ============================================================
    logic        periph_req,    periph_write;
    logic [31:0] periph_addr,   periph_wdata;
    logic [ 3:0] periph_wstrb;
    logic [31:0] periph_rdata;
    logic        periph_ready;

    logic        npu_csr_req,   npu_csr_write;
    logic [11:0] npu_csr_addr;
    logic [31:0] npu_csr_wdata;
    logic [31:0] npu_csr_rdata;
    logic        npu_csr_ready;

    logic        npu_isram_req,  npu_isram_write;
    logic [12:0] npu_isram_addr;
    logic [31:0] npu_isram_wdata;
    logic [ 3:0] npu_isram_wstrb;
    logic [31:0] npu_isram_rdata;
    logic        npu_isram_ready;

    logic        npu_vsram_req,  npu_vsram_write;
    logic [15:0] npu_vsram_addr;
    logic [31:0] npu_vsram_wdata;
    logic [ 3:0] npu_vsram_wstrb;
    logic [31:0] npu_vsram_rdata;
    logic        npu_vsram_ready;

    logic        extmem_req,    extmem_write;
    logic [31:0] extmem_addr,   extmem_wdata;
    logic [ 3:0] extmem_wstrb;
    logic [31:0] extmem_rdata;
    logic        extmem_ready;

    // ============================================================
    // NPU signals
    // ============================================================
    logic [11:0] npu_csr_mmio_addr;
    logic [31:0] npu_csr_mmio_wdata;
    logic        npu_csr_mmio_we;
    logic        npu_csr_mmio_re;
    logic [31:0] npu_csr_mmio_rdata;
    logic        npu_irq;

    // ============================================================
    // ExtMem model signals (from crossbar)
    // ============================================================
    logic        mem_req,       mem_write;
    logic [23:0] mem_addr;
    logic [31:0] mem_wdata;
    logic [ 3:0] mem_wstrb;
    logic [31:0] mem_rdata;
    logic        mem_ready;

    // ============================================================
    // IRQ wiring: NPU irq → CPU irq_in[0]
    // ============================================================
    logic [31:0] cpu_irq;
    assign cpu_irq = {31'd0, npu_irq};

    // ============================================================
    // RISC-V Control Processor
    // ============================================================
    picorv32_wrapper u_cpu (
        .clk,
        .rst_n,
        .irq_in         (cpu_irq),

        .periph_req     (periph_req),
        .periph_write   (periph_write),
        .periph_addr    (periph_addr),
        .periph_wdata   (periph_wdata),
        .periph_wstrb   (periph_wstrb),
        .periph_rdata   (periph_rdata),
        .periph_ready   (periph_ready),

        .npu_csr_req    (npu_csr_req),
        .npu_csr_write  (npu_csr_write),
        .npu_csr_addr   (npu_csr_addr),
        .npu_csr_wdata  (npu_csr_wdata),
        .npu_csr_rdata  (npu_csr_rdata),
        .npu_csr_ready  (npu_csr_ready),

        .npu_isram_req  (npu_isram_req),
        .npu_isram_write(npu_isram_write),
        .npu_isram_addr (npu_isram_addr),
        .npu_isram_wdata(npu_isram_wdata),
        .npu_isram_wstrb(npu_isram_wstrb),
        .npu_isram_rdata(npu_isram_rdata),
        .npu_isram_ready(npu_isram_ready),

        .npu_vsram_req  (npu_vsram_req),
        .npu_vsram_write(npu_vsram_write),
        .npu_vsram_addr (npu_vsram_addr),
        .npu_vsram_wdata(npu_vsram_wdata),
        .npu_vsram_wstrb(npu_vsram_wstrb),
        .npu_vsram_rdata(npu_vsram_rdata),
        .npu_vsram_ready(npu_vsram_ready),

        .extmem_req     (extmem_req),
        .extmem_write   (extmem_write),
        .extmem_addr    (extmem_addr),
        .extmem_wdata   (extmem_wdata),
        .extmem_wstrb   (extmem_wstrb),
        .extmem_rdata   (extmem_rdata),
        .extmem_ready   (extmem_ready),

        .trap           ()      // unused
    );

    // ============================================================
    // DMA external memory bypass signals
    // ============================================================
    logic [63:0] npu_dma_ext_rdata;
    logic [31:0] npu_dma_ext_addr;
    logic        npu_dma_ext_re;
    logic        npu_dma_ext_we;
    logic [63:0] npu_dma_ext_wdata;

    // ============================================================
    // NPU Accelerator Top
    // ============================================================
    npu_top u_npu (
        .clk,
        .rst_n,

        .csr_addr       (npu_csr_mmio_addr),
        .csr_wdata      (npu_csr_mmio_wdata),
        .csr_we         (npu_csr_mmio_we),
        .csr_re         (npu_csr_mmio_re),
        .csr_rdata      (npu_csr_mmio_rdata),

        .dbg_imem_we    (npu_isram_req && npu_isram_write),
        .dbg_imem_addr  (npu_isram_addr[9:2]),
        .dbg_imem_wdata (npu_isram_wdata),

        .dbg_valu_wen       (1'b0),
        .dbg_valu_waddr     (5'd0),
        .dbg_valu_wdata_flat(512'd0),
        .dbg_valu_raddr     (5'd0),
        .dbg_valu_rdata_flat(),

        .debug_pc       (),
        .debug_instr    (),
        .debug_stall    (),

        .irq            (npu_irq),

        .dma_ext_rdata  (npu_dma_ext_rdata),
        .dma_ext_addr   (npu_dma_ext_addr),
        .dma_ext_re     (npu_dma_ext_re),
        .dma_ext_we     (npu_dma_ext_we),
        .dma_ext_wdata  (npu_dma_ext_wdata)
    );

    // ============================================================
    // SoC Crossbar (address decode + bus routing)
    // ============================================================
    axi_crossbar_soc u_crossbar (
        .clk,
        .rst_n,

        // PicoRV32 slave ports
        .periph_req     (periph_req),
        .periph_write   (periph_write),
        .periph_addr    (periph_addr),
        .periph_wdata   (periph_wdata),
        .periph_wstrb   (periph_wstrb),
        .periph_rdata   (periph_rdata),
        .periph_ready   (periph_ready),

        .npu_csr_req    (npu_csr_req),
        .npu_csr_write  (npu_csr_write),
        .npu_csr_addr   (npu_csr_addr),
        .npu_csr_wdata  (npu_csr_wdata),
        .npu_csr_rdata  (npu_csr_rdata),
        .npu_csr_ready  (npu_csr_ready),

        .npu_isram_req  (npu_isram_req),
        .npu_isram_write(npu_isram_write),
        .npu_isram_addr (npu_isram_addr),
        .npu_isram_wdata(npu_isram_wdata),
        .npu_isram_wstrb(npu_isram_wstrb),
        .npu_isram_rdata(npu_isram_rdata),
        .npu_isram_ready(npu_isram_ready),

        .npu_vsram_req  (npu_vsram_req),
        .npu_vsram_write(npu_vsram_write),
        .npu_vsram_addr (npu_vsram_addr),
        .npu_vsram_wdata(npu_vsram_wdata),
        .npu_vsram_wstrb(npu_vsram_wstrb),
        .npu_vsram_rdata(npu_vsram_rdata),
        .npu_vsram_ready(npu_vsram_ready),

        .extmem_req     (extmem_req),
        .extmem_write   (extmem_write),
        .extmem_addr    (extmem_addr),
        .extmem_wdata   (extmem_wdata),
        .extmem_wstrb   (extmem_wstrb),
        .extmem_rdata   (extmem_rdata),
        .extmem_ready   (extmem_ready),

        // NPU CSR interface
        .csr_addr       (npu_csr_mmio_addr),
        .csr_wdata      (npu_csr_mmio_wdata),
        .csr_we         (npu_csr_mmio_we),
        .csr_re         (npu_csr_mmio_re),
        .csr_rdata      (npu_csr_mmio_rdata),

        // ExtMem model interface
        .mem_req        (mem_req),
        .mem_write      (mem_write),
        .mem_addr       (mem_addr),
        .mem_wdata      (mem_wdata),
        .mem_wstrb      (mem_wstrb),
        .mem_rdata      (mem_rdata),
        .mem_ready      (mem_ready),

        // UART
        .uart_tx        (uart_tx)
    );

    // ============================================================
    // External DRAM Model (64 MB)
    // ============================================================
    ext_mem_model #(
        .HEX_FILE("sim/verilog/firmware.hex")
    ) u_dram (
        .clk,
        .rst_n,
        .req        (mem_req),
        .write      (mem_write),
        .addr       (mem_addr),
        .wdata      (mem_wdata),
        .wstrb      (mem_wstrb),
        .rdata      (mem_rdata),
        .ready      (mem_ready)
    );

    // ============================================================
    // DMA External Memory Bridge
    //
    // Translates DMA 64-bit byte-addressed accesses to
    // ext_mem_model 32-bit word-addressed storage.
    //
    // Same address translation as the crossbar:
    //   CPU 0x40000000 → DRAM word 0
    // ============================================================
    wire [31:0] dma_offset   = npu_dma_ext_addr - 32'h4000_0000;
    wire [23:0] dma_word_addr = dma_offset[25:2];

    // Read: combinational — DMA wrapper samples on posedge
    assign npu_dma_ext_rdata = {
        u_dram.mem[dma_word_addr + 1],
        u_dram.mem[dma_word_addr]
    };

    // Write: clocked — DMA wrapper asserts bypass_we during S_XFER
    always_ff @(posedge clk) begin
        if (npu_dma_ext_we) begin
            u_dram.mem[dma_word_addr]     <= npu_dma_ext_wdata[31:0];
            u_dram.mem[dma_word_addr + 1] <= npu_dma_ext_wdata[63:32];
        end
    end

endmodule
