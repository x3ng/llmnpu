// ============================================================
// picorv32_wrapper.sv — NPU Control Processor Wrapper
//
// Instantiates picorv32_adapted (RV32IM) with:
//   - Memory-map address decoder → 5 slaves
//   - IRQ input from NPU
//   - Per-slave request/response ports
//
// Memory Map (spec §3.1):
//   Peripherals:   0x0000_0000 – 0x0FFF_FFFF  (UART, Timer, ...)
//   NPU MMIO CSR:  0x1000_0000 – 0x1000_0FFF  (4 KB)
//   NPU I-SRAM:    0x1001_0000 – 0x1001_1FFF  (8 KB)
//   NPU V-SRAM:    0x1002_0000 – 0x1002_FFFF  (64 KB)
//   ExtMem DRAM:   0x4000_0000 – 0x43FF_FFFF  (64 MB)
// ============================================================

module picorv32_wrapper (
    input  logic        clk,
    input  logic        rst_n,

    // ---- IRQ from NPU (connected to CSR irq output) ----
    input  logic [31:0] irq_in,

    // ============================================================
    // Per-Slave Memory Ports
    // ============================================================

    // Peripherals (UART at 0x0, timer, etc.)
    output logic        periph_req,
    output logic        periph_write,
    output logic [31:0] periph_addr,
    output logic [31:0] periph_wdata,
    output logic [ 3:0] periph_wstrb,
    input  logic [31:0] periph_rdata,
    input  logic        periph_ready,

    // NPU MMIO CSR
    output logic        npu_csr_req,
    output logic        npu_csr_write,
    output logic [11:0] npu_csr_addr,
    output logic [31:0] npu_csr_wdata,
    input  logic [31:0] npu_csr_rdata,
    input  logic        npu_csr_ready,

    // NPU I-SRAM (instruction SRAM, 8 KB)
    output logic        npu_isram_req,
    output logic        npu_isram_write,
    output logic [12:0] npu_isram_addr,
    output logic [31:0] npu_isram_wdata,
    output logic [ 3:0] npu_isram_wstrb,
    input  logic [31:0] npu_isram_rdata,
    input  logic        npu_isram_ready,

    // NPU V-SRAM (vector/weight SRAM, 64 KB)
    output logic        npu_vsram_req,
    output logic        npu_vsram_write,
    output logic [15:0] npu_vsram_addr,
    output logic [31:0] npu_vsram_wdata,
    output logic [ 3:0] npu_vsram_wstrb,
    input  logic [31:0] npu_vsram_rdata,
    input  logic        npu_vsram_ready,

    // External DRAM
    output logic        extmem_req,
    output logic        extmem_write,
    output logic [31:0] extmem_addr,
    output logic [31:0] extmem_wdata,
    output logic [ 3:0] extmem_wstrb,
    input  logic [31:0] extmem_rdata,
    input  logic        extmem_ready,

    // ---- Trap (debug) ----
    output logic        trap
);

    // ============================================================
    // PicoRV32 core signals
    // ============================================================
    logic        cpu_mem_valid;
    logic        cpu_mem_instr;
    logic        cpu_mem_ready;
    logic [31:0] cpu_mem_addr;
    logic [31:0] cpu_mem_wdata;
    logic [ 3:0] cpu_mem_wstrb;
    logic [31:0] cpu_mem_rdata;

    logic [31:0] cpu_eoi;        // end-of-interrupt (unused)

    // ============================================================
    // PicoRV32_adapted instantiation (RV32IM)
    // ============================================================
    picorv32_adapted #(
        .ENABLE_MUL      (1),
        .ENABLE_FAST_MUL (1),
        .ENABLE_DIV      (1),
        .ENABLE_IRQ      (1),
        .BARREL_SHIFTER  (1),
        .ENABLE_REGS_16_31 (1),
        .ENABLE_REGS_DUALPORT (1),
        .ENABLE_COUNTERS (1),
        .ENABLE_COUNTERS64 (1),
        .CATCH_MISALIGN  (1),
        .CATCH_ILLINSN   (1),
        .PROGADDR_RESET  (32'h0000_0000),
        .PROGADDR_IRQ    (32'h0000_0010),
        .STACKADDR       (32'h4400_0000),    // top of ExtMem DRAM (FW overrides in _start)
        .MASKED_IRQ      (32'h0000_0000),
        .LATCHED_IRQ     (32'hffff_ffff)
    ) u_picorv32 (
        .clk        (clk),
        .resetn     (rst_n),
        .trap       (trap),

        .mem_valid  (cpu_mem_valid),
        .mem_instr  (cpu_mem_instr),
        .mem_ready  (cpu_mem_ready),
        .mem_addr   (cpu_mem_addr),
        .mem_wdata  (cpu_mem_wdata),
        .mem_wstrb  (cpu_mem_wstrb),
        .mem_rdata  (cpu_mem_rdata),

        // Look-ahead (unused — tie off)
        .mem_la_read  (),
        .mem_la_write (),
        .mem_la_addr  (),
        .mem_la_wdata (),
        .mem_la_wstrb (),

        // PCPI (unused — tie off)
        .pcpi_valid (),
        .pcpi_insn  (),
        .pcpi_rs1   (),
        .pcpi_rs2   (),
        .pcpi_wr    (1'b0),
        .pcpi_rd    (32'd0),
        .pcpi_wait  (1'b0),
        .pcpi_ready (1'b0),

        // IRQ
        .irq    (irq_in),
        .eoi    (cpu_eoi),

        // Trace (unused)
        .trace_valid (),
        .trace_data  ()
    );

    // ============================================================
    // Address Decoder
    // ============================================================
    // Slave select encoding (one-hot)
    //   [4] = Peripherals  (0x0xxx_xxxx)
    //   [3] = NPU CSR      (0x1000_0xxx)
    //   [2] = NPU I-SRAM   (0x1001_xxxx)
    //   [1] = NPU V-SRAM   (0x1002_xxxx)
    //   [0] = ExtMem DRAM  (0x40xx_xxxx)
    // ============================================================

    logic [4:0] slave_sel;

    always_comb begin
        slave_sel = 5'd0;
        // ExtMem DRAM: 0x4000_0000 – 0x43FF_FFFF
        if (cpu_mem_addr[31:26] == 6'b010000)  // 0x40.. – 0x43..
            slave_sel = 5'b00001;
        // NPU V-SRAM: 0x1002_0000 – 0x1002_FFFF
        else if (cpu_mem_addr[31:16] == 16'h1002)
            slave_sel = 5'b00010;
        // NPU I-SRAM: 0x1001_0000 – 0x1001_1FFF
        else if (cpu_mem_addr[31:13] == 19'h08008)
            slave_sel = 5'b00100;
        // NPU MMIO CSR: 0x1000_0000 – 0x1000_0FFF
        else if (cpu_mem_addr[31:12] == 20'h10000)
            slave_sel = 5'b01000;
        // Peripherals: 0x0000_0000 – 0x0FFF_FFFF
        else if (cpu_mem_addr[31:28] == 4'h0)
            slave_sel = 5'b10000;
        // Default: no slave selected (returns 0, no ready)
    end

    // ============================================================
    // Request routing (combinational)
    // ============================================================
    wire is_write = |cpu_mem_wstrb;   // write if any byte strobe asserted

    always_comb begin
        // Defaults: all slaves inactive
        periph_req      = 1'b0;
        periph_write    = 1'b0;
        periph_addr     = cpu_mem_addr;
        periph_wdata    = cpu_mem_wdata;
        periph_wstrb    = cpu_mem_wstrb;

        npu_csr_req     = 1'b0;
        npu_csr_write   = 1'b0;
        npu_csr_addr    = cpu_mem_addr[11:0];
        npu_csr_wdata   = cpu_mem_wdata;

        npu_isram_req   = 1'b0;
        npu_isram_write = 1'b0;
        npu_isram_addr  = cpu_mem_addr[12:0];
        npu_isram_wdata = cpu_mem_wdata;
        npu_isram_wstrb = cpu_mem_wstrb;

        npu_vsram_req   = 1'b0;
        npu_vsram_write = 1'b0;
        npu_vsram_addr  = cpu_mem_addr[15:0];
        npu_vsram_wdata = cpu_mem_wdata;
        npu_vsram_wstrb = cpu_mem_wstrb;

        extmem_req      = 1'b0;
        extmem_write    = 1'b0;
        extmem_addr     = cpu_mem_addr;
        extmem_wdata    = cpu_mem_wdata;
        extmem_wstrb    = cpu_mem_wstrb;

        // Activate selected slave
        if (cpu_mem_valid) begin
            casez (slave_sel)
                5'b10000: begin  // Peripherals
                    periph_req   = 1'b1;
                    periph_write = is_write;
                end
                5'b01000: begin  // NPU CSR
                    npu_csr_req   = 1'b1;
                    npu_csr_write = is_write;
                end
                5'b00100: begin  // NPU I-SRAM
                    npu_isram_req   = 1'b1;
                    npu_isram_write = is_write;
                end
                5'b00010: begin  // NPU V-SRAM
                    npu_vsram_req   = 1'b1;
                    npu_vsram_write = is_write;
                end
                5'b00001: begin  // ExtMem DRAM
                    extmem_req   = 1'b1;
                    extmem_write = is_write;
                end
                default: ;  // no slave selected
            endcase
        end
    end

    // ============================================================
    // Read data + ready multiplexer (back to PicoRV32)
    // ============================================================
    always_comb begin
        cpu_mem_ready = 1'b0;
        cpu_mem_rdata = 32'd0;

        casez (slave_sel)
            5'b10000: begin
                cpu_mem_ready = periph_ready;
                cpu_mem_rdata = periph_rdata;
            end
            5'b01000: begin
                cpu_mem_ready = npu_csr_ready;
                cpu_mem_rdata = npu_csr_rdata;
            end
            5'b00100: begin
                cpu_mem_ready = npu_isram_ready;
                cpu_mem_rdata = npu_isram_rdata;
            end
            5'b00010: begin
                cpu_mem_ready = npu_vsram_ready;
                cpu_mem_rdata = npu_vsram_rdata;
            end
            5'b00001: begin
                cpu_mem_ready = extmem_ready;
                cpu_mem_rdata = extmem_rdata;
            end
            default: begin
                // No slave selected — return 0 immediately to avoid hang
                cpu_mem_ready = cpu_mem_valid;
                cpu_mem_rdata = 32'd0;
            end
        endcase
    end

endmodule
