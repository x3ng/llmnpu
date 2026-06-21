// csr.sv — 32-bit MMIO Control/Status Register file
// Base address: 0x1000_0000
// Address decode: 12-bit byte address, word-aligned (addr[11:2])

`include "npu_defines.svh"

module csr (
    input  logic        clk,
    input  logic        rst_n,

    // MMIO
    input  logic [11:0] addr,
    input  logic [31:0] wdata,
    input  logic        we,
    output logic [31:0] rdata,

    // Status inputs from NPU
    input  logic        npu_busy,
    input  logic        npu_going_idle,
    input  logic        dma_err_event,
    input  logic        ill_insn_event,
    input  logic        perf_busy,
    input  logic        perf_gemm_busy,
    input  logic        perf_valu_busy,
    input  logic        perf_sfu_busy,
    input  logic        perf_dma_busy,
    input  logic [7:0]  current_pc,

    // Debug signals (exposed read-only via DEBUG register)
    input  logic [31:0] debug_signals,

    // Control outputs to NPU
    output logic        npu_start,
    output logic        npu_rst,
    output logic        npu_halt,
    output logic        pc_we,
    output logic [7:0]  pc_wdata,
    output logic [7:0]  issue_opcode,

    // DMA CSR register outputs (for top-level DMA wiring)
    output logic [31:0] dma_ext_addr,
    output logic [15:0] dma_sram_addr,
    output logic [15:0] dma_length,
    output logic [15:0] dma_row_count,
    output logic [15:0] dma_row_bytes,
    output logic [15:0] dma_ext_stride,
    output logic [15:0] dma_sram_stride,
    output logic        dma_csr_start,
    output logic        dma_csr_is_store,
    output logic        dma_csr_is_2d,

    // Descriptor pointer
    output logic [31:0] desc_ptr,

    // Interrupt
    output logic        irq
);

    // ----------------------------------------------------------------
    // Register word addresses
    // ----------------------------------------------------------------
    localparam logic [9:0] A_CTRL      = 10'h00;
    localparam logic [9:0] A_STATUS    = 10'h01;
    localparam logic [9:0] A_PC        = 10'h02;
    localparam logic [9:0] A_DESC_PTR  = 10'h04;
    // DMA CSR word addresses (byte offsets: 0x20, 0x28, 0x30, 0x38)
    localparam logic [9:0] A_DMA_CSR0  = 10'h08;   // byte 0x20
    localparam logic [9:0] A_DMA_CSR1  = 10'h0A;   // byte 0x28
    localparam logic [9:0] A_DMA_CSR2  = 10'h0C;   // byte 0x30
    localparam logic [9:0] A_DMA_CSR3  = 10'h0E;   // byte 0x38
    localparam logic [9:0] A_IRQ_EN    = 10'h10;
    localparam logic [9:0] A_IRQ_STAT  = 10'h11;
    localparam logic [9:0] A_PERF_CYC  = 10'h20;
    localparam logic [9:0] A_PERF_BUSY = 10'h21;
    localparam logic [9:0] A_PERF_GEMM = 10'h22;
    localparam logic [9:0] A_PERF_VALU = 10'h23;
    localparam logic [9:0] A_PERF_SFU  = 10'h24;
    localparam logic [9:0] A_PERF_DMA  = 10'h25;
    localparam logic [9:0] A_DEBUG     = 10'h18;   // byte 0x60

    // ----------------------------------------------------------------
    // Individual registers (no unpacked arrays for iverilog compat)
    // ----------------------------------------------------------------
    logic [31:0] ctrl_reg;
    logic [31:0] status_reg;
    logic [31:0] desc_ptr_reg;
    logic [31:0] dma_csr0, dma_csr1, dma_csr2, dma_csr3;
    logic [31:0] irq_en_reg;
    logic [31:0] irq_stat_reg;
    logic [31:0] perf_cycle_reg;
    logic [31:0] perf_busy_reg;
    logic [31:0] perf_gemm_reg;
    logic [31:0] perf_valu_reg;
    logic [31:0] perf_sfu_reg;
    logic [31:0] perf_dma_reg;

    wire [9:0] waddr = addr[11:2];

    // ----------------------------------------------------------------
    // Write (sequential) — with integrated irq_stat auto-set
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg       <= 32'd0;
            desc_ptr_reg   <= 32'd0;
            dma_csr0       <= 32'd0;
            dma_csr1       <= 32'd0;
            dma_csr2       <= 32'd0;
            dma_csr3       <= 32'd0;
            irq_en_reg     <= 32'd0;
            irq_stat_reg   <= 32'd0;
            perf_cycle_reg <= 32'd0;
            perf_busy_reg  <= 32'd0;
            perf_gemm_reg  <= 32'd0;
            perf_valu_reg  <= 32'd0;
            perf_sfu_reg   <= 32'd0;
            perf_dma_reg   <= 32'd0;
        end else begin
            // Performance counters.  Writes below can seed or clear any
            // counter; otherwise active sources increment once per cycle.
            perf_cycle_reg <= perf_cycle_reg + 32'd1;
            if (perf_busy)
                perf_busy_reg <= perf_busy_reg + 32'd1;
            if (perf_gemm_busy)
                perf_gemm_reg <= perf_gemm_reg + 32'd1;
            if (perf_valu_busy)
                perf_valu_reg <= perf_valu_reg + 32'd1;
            if (perf_sfu_busy)
                perf_sfu_reg <= perf_sfu_reg + 32'd1;
            if (perf_dma_busy)
                perf_dma_reg <= perf_dma_reg + 32'd1;

            // irq_stat[0] set when any busy engine is about to go idle
            // (npu_going_idle pulses one cycle before busy falls)
            if (npu_going_idle)
                irq_stat_reg[0] <= 1'b1;
            if (dma_err_event)
                irq_stat_reg[1] <= 1'b1;
            if (ill_insn_event)
                irq_stat_reg[2] <= 1'b1;

            if (we) begin
                case (waddr)
                    A_CTRL:      ctrl_reg     <= wdata;
                    A_DESC_PTR:  desc_ptr_reg <= wdata;
                    A_DMA_CSR0:  dma_csr0     <= wdata;
                    A_DMA_CSR1:  dma_csr1     <= wdata;
                    A_DMA_CSR2:  dma_csr2     <= wdata;
                    A_DMA_CSR3:  dma_csr3     <= wdata;
                    A_IRQ_EN:    irq_en_reg   <= wdata;
                    A_IRQ_STAT:  irq_stat_reg <= irq_stat_reg & ~wdata;  // W1C
                    A_PERF_CYC:  perf_cycle_reg <= wdata;
                    A_PERF_BUSY: perf_busy_reg  <= wdata;
                    A_PERF_GEMM: perf_gemm_reg  <= wdata;
                    A_PERF_VALU: perf_valu_reg  <= wdata;
                    A_PERF_SFU:  perf_sfu_reg   <= wdata;
                    A_PERF_DMA:  perf_dma_reg   <= wdata;
                    default: ;
                endcase
            end
        end
    end

    // ----------------------------------------------------------------
    // Read (combinational)
    // ----------------------------------------------------------------
    always_comb begin
        rdata = 32'd0;
        case (waddr)
            A_CTRL:      rdata = ctrl_reg;
            A_STATUS:    rdata = status_reg;
            A_PC:        rdata = {24'd0, current_pc};
            A_DESC_PTR:  rdata = desc_ptr_reg;
            A_DMA_CSR0:  rdata = dma_csr0;
            A_DMA_CSR1:  rdata = dma_csr1;
            A_DMA_CSR2:  rdata = dma_csr2;
            A_DMA_CSR3:  rdata = dma_csr3;
            A_IRQ_EN:    rdata = irq_en_reg;
            A_IRQ_STAT:  rdata = irq_stat_reg;
            A_PERF_CYC:  rdata = perf_cycle_reg;
            A_PERF_BUSY: rdata = perf_busy_reg;
            A_PERF_GEMM: rdata = perf_gemm_reg;
            A_PERF_VALU: rdata = perf_valu_reg;
            A_PERF_SFU:  rdata = perf_sfu_reg;
            A_PERF_DMA:  rdata = perf_dma_reg;
            A_DEBUG:     rdata = debug_signals;
            default:     rdata = 32'd0;
        endcase
    end

    // ----------------------------------------------------------------
    // DMA CSR outputs (combinational — top samples when needed)
    // ----------------------------------------------------------------
    // 1D: CSR0=ext_addr, CSR1[15:0]=sram_off, CSR2[15:0]=length,
    //     CSR3[1:0]=dir/start.
    // 2D: CSR1[31:16]=sram_stride, CSR2[31:16]=rows,
    //     CSR2[15:0]=row_bytes, CSR3[31:16]=ext_stride, CSR3[2]=2D.
    assign dma_ext_addr   = dma_csr0;
    assign dma_sram_addr  = dma_csr1[15:0];
    assign dma_length     = dma_csr2[15:0];
    assign dma_row_count  = dma_csr2[31:16];
    assign dma_row_bytes  = dma_csr2[15:0];
    assign dma_ext_stride = (we && (waddr == A_DMA_CSR3)) ? wdata[31:16] :
                                                                dma_csr3[31:16];
    assign dma_sram_stride= dma_csr1[31:16];

    // dma_csr_start: pulse when DMA_CSR3 written with bit[0] set
    assign dma_csr_start    = we && (waddr == A_DMA_CSR3) && wdata[0];
    assign dma_csr_is_store = (we && (waddr == A_DMA_CSR3)) ? wdata[1] : dma_csr3[1];
    assign dma_csr_is_2d    = (we && (waddr == A_DMA_CSR3)) ? wdata[2] : dma_csr3[2];

    // desc_ptr: from CSR register for descriptor fetch
    assign desc_ptr = desc_ptr_reg;

    // ----------------------------------------------------------------
    // Control outputs
    // ----------------------------------------------------------------
    // npu_start: single-cycle pulse whenever software writes CTRL[0].
    // CTRL is a stored debug/control register, but START behaves like a strobe.
    assign npu_start = we && (waddr == A_CTRL) && wdata[0];

    assign npu_rst = ctrl_reg[1];
    assign npu_halt = ctrl_reg[2];
    assign pc_we = we && (waddr == A_PC);
    assign pc_wdata = wdata[7:0];
    assign issue_opcode = (we && (waddr == A_CTRL)) ? wdata[15:8] : ctrl_reg[15:8];

    // ----------------------------------------------------------------
    // Status register
    //   bit0 = busy (RO, direct from npu_busy)
    //   bit1 = irq_pend (RO, any pending IRQ source)
    // ----------------------------------------------------------------
    assign status_reg = {30'd0, |irq_stat_reg, npu_busy};

    // ----------------------------------------------------------------
    // Interrupt generation
    // ----------------------------------------------------------------
    // IRQ asserted when any irq_en bit AND corresponding irq_stat bit are set
    assign irq = |(irq_en_reg & irq_stat_reg);

endmodule
