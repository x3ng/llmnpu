// ============================================================
// top_soc.sv — NPU SoC Top-Level Integration
//
// Instantiates:
//   picorv32_wrapper     — RISC-V control processor (RV32IM)
//   npu_top              — NPU datapath (GEMM, VALU, SFU, DMA)
//   axi_crossbar_soc     — Address decoder + bus router
//   ext_mem_model        — 64 MB behavioral DRAM
//
// AXI-to-extmem bridge converts DMA's 64-bit AXI4 master to
// ext_mem_model's 32-bit word interface.
//
// Debug outputs:
//   uart_tx[7:0]         — 'P' on pass, 'F' on fail
// ============================================================

`include "npu_defines.svh"
`include "isa_defines.svh"

module top_soc (
    input  logic        clk,
    input  logic        rst_n,

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

    // AXI bridge → ext_mem_model direct access ports
    logic [23:0] axi_rd_addr;
    logic        axi_rd_en;
    logic [31:0] axi_rd_rdata;
    logic [23:0] axi_wr_addr;
    logic [31:0] axi_wr_wdata;
    logic        axi_wr_en;

    logic [31:0] cpu_irq;
    assign cpu_irq = {31'd0, npu_irq};

    // ============================================================
    // RISC-V Control Processor
    // ============================================================
    // trap latch: captures PicoRV32 trap assertion (e.g. illegal insn)
    logic cpu_trap;
    logic trap_latched;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            trap_latched <= 1'b0;
        else if (cpu_trap)
            trap_latched <= 1'b1;
    end

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
        .trap           (cpu_trap)
    );

    // ============================================================
    // DMA AXI signals (from npu_top)
    // ============================================================
    logic [31:0] dma_axi_araddr;
    logic [7:0]  dma_axi_arlen;
    logic        dma_axi_arvalid;
    logic        dma_axi_arready;

    logic [63:0] dma_axi_rdata;
    logic [1:0]  dma_axi_rresp;
    logic        dma_axi_rlast;
    logic        dma_axi_rvalid;
    logic        dma_axi_rready;

    logic [31:0] dma_axi_awaddr;
    logic [7:0]  dma_axi_awlen;
    logic        dma_axi_awvalid;
    logic        dma_axi_awready;

    logic [63:0] dma_axi_wdata;
    logic [7:0]  dma_axi_wstrb;
    logic        dma_axi_wlast;
    logic        dma_axi_wvalid;
    logic        dma_axi_wready;

    logic [1:0]  dma_axi_bresp;
    logic        dma_axi_bvalid;
    logic        dma_axi_bready;

    // ============================================================
    // NPU Accelerator Top (DMA_STANDALONE=0: AXI → external)
    // ============================================================
    npu_top #(
        .DMA_STANDALONE (0)
    ) u_npu (
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
        .dma_axi_araddr  (dma_axi_araddr),
        .dma_axi_arlen   (dma_axi_arlen),
        .dma_axi_arvalid (dma_axi_arvalid),
        .dma_axi_arready (dma_axi_arready),
        .dma_axi_rdata   (dma_axi_rdata),
        .dma_axi_rresp   (dma_axi_rresp),
        .dma_axi_rlast   (dma_axi_rlast),
        .dma_axi_rvalid  (dma_axi_rvalid),
        .dma_axi_rready  (dma_axi_rready),
        .dma_axi_awaddr  (dma_axi_awaddr),
        .dma_axi_awlen   (dma_axi_awlen),
        .dma_axi_awvalid (dma_axi_awvalid),
        .dma_axi_awready (dma_axi_awready),
        .dma_axi_wdata   (dma_axi_wdata),
        .dma_axi_wstrb   (dma_axi_wstrb),
        .dma_axi_wlast   (dma_axi_wlast),
        .dma_axi_wvalid  (dma_axi_wvalid),
        .dma_axi_wready  (dma_axi_wready),
        .dma_axi_bresp   (dma_axi_bresp),
        .dma_axi_bvalid  (dma_axi_bvalid),
        .dma_axi_bready  (dma_axi_bready)
    );

    // ============================================================
    // SoC Crossbar
    // ============================================================
    axi_crossbar_soc u_crossbar (
        .clk,
        .rst_n,
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
        .csr_addr       (npu_csr_mmio_addr),
        .csr_wdata      (npu_csr_mmio_wdata),
        .csr_we         (npu_csr_mmio_we),
        .csr_re         (npu_csr_mmio_re),
        .csr_rdata      (npu_csr_mmio_rdata),
        .mem_req        (mem_req),
        .mem_write      (mem_write),
        .mem_addr       (mem_addr),
        .mem_wdata      (mem_wdata),
        .mem_wstrb      (mem_wstrb),
        .mem_rdata      (mem_rdata),
        .mem_ready      (mem_ready),
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
        .ready      (mem_ready),
        .axi_rd_addr,
        .axi_rd_en,
        .axi_rd_rdata,
        .axi_wr_addr,
        .axi_wr_wdata,
        .axi_wr_en
    );

    // ================================================================
    // AXI-to-ExtMem Bridge
    //
    // Converts DMA's 64-bit AXI4 to ext_mem_model's 32-bit words.
    // Address translation: CPU 0x40000000 → DRAM word 0
    //   word_addr = (axi_byte_addr - 0x40000000) / 4
    //
    // Reads:  ext_mem reads are combinational (mem[addr] available
    //   immediately).  Two reads per 64-bit beat (low word, high word).
    // Writes: ext_mem writes are clocked.  Two writes per 64-bit beat.
    //   Writes use axi_wr_* ports (ext_mem_model latches on axi_wr_en).
    // ================================================================

    // ---- Address translation helper (pure combinational) ----
    function automatic [23:0] to_word_addr(input [31:0] byte_addr);
        logic [31:0] off;
        off = byte_addr - 32'h4000_0000;
        to_word_addr = off[25:2];
    endfunction

    // ================================================================
    // AXI Bridge — ext_mem_model port control (uses proper read/write
    //   ports instead of hierarchical references)
    // ================================================================

    // ---- AXI Read Bridge state type and variables ----
    typedef enum logic [1:0] {
        AR_IDLE  = 2'd0,
        AR_LOW   = 2'd1,
        AR_HIGH  = 2'd2,
        AR_WAIT  = 2'd3
    } ar_state_t;

    ar_state_t ar_state, ar_next;

    // ---- AXI Write Bridge state type and variables ----
    typedef enum logic [2:0] {
        AW_IDLE   = 3'd0,
        AW_WAIT_W = 3'd1,
        AW_WR_LO  = 3'd2,
        AW_WR_HI  = 3'd3,
        AW_NEXT   = 3'd4,
        AW_RESP   = 3'd5
    } aw_state_t;

    aw_state_t aw_state, aw_next;

    // ================================================================
    // AXI Read Bridge
    //
    // States: IDLE → RD_LOW (read low word) → RD_HIGH (read high,
    //   present rdata) → (wait rready, loop to RD_LOW for next beat)
    // ================================================================
    logic [7:0]  ar_beat;
    logic [7:0]  ar_len;
    logic [31:0] ar_addr;
    logic [31:0] ar_lo;   // captured low 32-bit word

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_state <= AR_IDLE;
            ar_beat  <= 8'd0;
            ar_len   <= 8'd0;
            ar_addr  <= 32'd0;
            ar_lo    <= 32'd0;
        end else begin
            ar_state <= ar_next;
            case (ar_state)
                AR_IDLE: begin
                    if (dma_axi_arvalid && dma_axi_arready) begin
                        ar_addr <= dma_axi_araddr;
                        ar_len  <= dma_axi_arlen;
                        ar_beat <= 8'd0;
                    end
                end
                AR_LOW: begin
                    // Read low word via axi_rd_rdata (combinational port)
                    ar_lo <= axi_rd_rdata;
                end
                AR_HIGH: begin
                    // Read high word (combinational), rdata presented via
                    // separate register (see below).  State advances to WAIT.
                end
                AR_WAIT: begin
                    if (dma_axi_rvalid && dma_axi_rready) begin
                        ar_addr <= ar_addr + 32'd8;
                        ar_beat <= ar_beat + 8'd1;
                    end
                end
            endcase
        end
    end

    always_comb begin
        ar_next = ar_state;
        case (ar_state)
            AR_IDLE:  if (dma_axi_arvalid && dma_axi_arready) ar_next = AR_LOW;
            AR_LOW:   ar_next = AR_HIGH;
            AR_HIGH:  ar_next = AR_WAIT;
            AR_WAIT: begin
                if (dma_axi_rvalid && dma_axi_rready) begin
                    if (ar_beat == ar_len)
                        ar_next = AR_IDLE;
                    else
                        ar_next = AR_LOW;
                end
            end
            default:  ar_next = AR_IDLE;
        endcase
    end

    assign dma_axi_arready = (ar_state == AR_IDLE);

    // R channel register
    reg        ar_rvalid;
    reg [63:0] ar_rdata;
    reg        ar_rlast;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_rvalid <= 1'b0;
            ar_rdata  <= 64'd0;
            ar_rlast  <= 1'b0;
        end else begin
            if (ar_state == AR_HIGH) begin
                // Assemble: {high_word from axi port, low_word from ar_lo}
                ar_rdata  <= {axi_rd_rdata, ar_lo};
                ar_rlast  <= (ar_beat == ar_len);
                ar_rvalid <= 1'b1;
            end else if (ar_rvalid && dma_axi_rready) begin
                ar_rvalid <= 1'b0;
            end
        end
    end

    assign dma_axi_rdata  = ar_rdata;
    assign dma_axi_rvalid = ar_rvalid;
    assign dma_axi_rlast  = ar_rlast;
    assign dma_axi_rresp  = 2'b00;

    // ================================================================
    // AXI Write Bridge
    //
    // Accepts AXI writes via axi_wr_* ports into ext_mem_model.
    // Two 32-bit writes per 64-bit beat.
    // ================================================================
    logic [7:0]  aw_beat;
    logic [7:0]  aw_len;
    logic [31:0] aw_addr;
    logic [63:0] aw_buf;   // captured wdata for current beat
    logic        aw_wlast;  // captured wlast for current beat

    // ================================================================
    // AXI Bridge — ext_mem_model port control (uses proper read/write
    //   ports instead of hierarchical references)
    // ================================================================

    // ---- Read port control (combinational) ----
    always_comb begin
        axi_rd_en   = 1'b0;
        axi_rd_addr = 24'd0;
        if (ar_state == AR_LOW) begin
            axi_rd_en   = 1'b1;
            axi_rd_addr = to_word_addr(ar_addr);
        end else if (ar_state == AR_HIGH) begin
            axi_rd_en   = 1'b1;
            axi_rd_addr = to_word_addr(ar_addr) + 24'd1;
        end
    end

    // ---- Write port control (combinational) ----
    always_comb begin
        axi_wr_en    = 1'b0;
        axi_wr_addr  = 24'd0;
        axi_wr_wdata = 32'd0;
        if (aw_state == AW_WR_LO) begin
            axi_wr_en    = 1'b1;
            axi_wr_addr  = to_word_addr(aw_addr);
            axi_wr_wdata = aw_buf[31:0];
        end else if (aw_state == AW_WR_HI) begin
            axi_wr_en    = 1'b1;
            axi_wr_addr  = to_word_addr(aw_addr) + 24'd1;
            axi_wr_wdata = aw_buf[63:32];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_state  <= AW_IDLE;
            aw_beat   <= 8'd0;
            aw_len    <= 8'd0;
            aw_addr   <= 32'd0;
            aw_buf    <= 64'd0;
            aw_wlast  <= 1'b0;
        end else begin
            aw_state <= aw_next;
            case (aw_state)
                AW_IDLE: begin
                    if (dma_axi_awvalid && dma_axi_awready) begin
                        aw_addr <= dma_axi_awaddr;
                        aw_len  <= dma_axi_awlen;
                        aw_beat <= 8'd0;
                    end
                end
                AW_WAIT_W: begin
                    if (dma_axi_wvalid && dma_axi_wready) begin
                        aw_buf   <= dma_axi_wdata;
                        aw_wlast <= dma_axi_wlast;
                    end
                end
                AW_WR_LO: begin
                    // Write low 32 bits via axi_wr_* port (driven by
                    // combinational block above, latched in ext_mem_model)
                end
                AW_WR_HI: begin
                    // Write high 32 bits via axi_wr_* port
                end
                AW_NEXT: begin
                    // Advance for next beat
                    aw_addr <= aw_addr + 32'd8;
                    aw_beat <= aw_beat + 8'd1;
                end
                AW_RESP: begin
                    if (dma_axi_bvalid && dma_axi_bready) begin
                        // done
                    end
                end
            endcase
        end
    end

    always_comb begin
        aw_next = aw_state;
        case (aw_state)
            AW_IDLE:   if (dma_axi_awvalid && dma_axi_awready)    aw_next = AW_WAIT_W;
            AW_WAIT_W: if (dma_axi_wvalid && dma_axi_wready)       aw_next = AW_WR_LO;
            AW_WR_LO:                                              aw_next = AW_WR_HI;
            AW_WR_HI:                                              aw_next = AW_NEXT;
            AW_NEXT:   if (aw_wlast)                               aw_next = AW_RESP;
                       else                                        aw_next = AW_WAIT_W;
            AW_RESP:   if (dma_axi_bvalid && dma_axi_bready)       aw_next = AW_IDLE;
            default:                                               aw_next = AW_IDLE;
        endcase
    end

    assign dma_axi_awready = (aw_state == AW_IDLE);
    assign dma_axi_wready  = (aw_state == AW_WAIT_W);

    // B channel
    reg aw_bvalid;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_bvalid <= 1'b0;
        end else begin
            if (aw_state == AW_RESP && !aw_bvalid)
                aw_bvalid <= 1'b1;
            else if (dma_axi_bvalid && dma_axi_bready)
                aw_bvalid <= 1'b0;
        end
    end
    assign dma_axi_bvalid = aw_bvalid;
    assign dma_axi_bresp  = 2'b00;

endmodule
