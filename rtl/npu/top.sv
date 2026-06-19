// npu_top.sv — NPU Top-Level Integration
//
// Instantiates and wires:
//   IF/ID → Dispatch → {GEMM, VALU, SFU, DMA} → Crossbar → SRAM banks
//   CSR register file, Ping-pong controllers

`include "npu_defines.svh"
`include "isa_defines.svh"

module npu_top (
    input  logic        clk,
    input  logic        rst_n,

    // ---- CSR MMIO (32-bit, 12-bit byte address) ----
    input  logic [11:0] csr_addr,
    input  logic [31:0] csr_wdata,
    input  logic        csr_we,
    input  logic        csr_re,
    output logic [31:0] csr_rdata,

    // ---- Debug: IF/ID instruction memory access ----
    input  logic        dbg_imem_we,
    input  logic [7:0]  dbg_imem_addr,
    input  logic [31:0] dbg_imem_wdata,

    // ---- Debug: VALU register file access (512-bit flat) ----
    input  logic        dbg_valu_wen,
    input  logic [4:0]  dbg_valu_waddr,
    input  logic [511:0] dbg_valu_wdata_flat,
    input  logic [4:0]  dbg_valu_raddr,
    output logic [511:0] dbg_valu_rdata_flat,

    // ---- Debug outputs ----
    output logic [7:0]  debug_pc,
    output logic [31:0] debug_instr,
    output logic        debug_stall,

    // ---- Interrupt ----
    output logic        irq
);

    // ================================================================
    // CSR signals
    // ================================================================
    logic        csr_start;
    logic        csr_rst;
    logic [31:0] csr_dma_ext_addr;
    logic [31:0] csr_dma_sram_len;

    // ================================================================
    // Running flag: set by csr_start pulse, cleared by csr_rst
    // ================================================================
    logic running;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            running <= 1'b0;
        else if (csr_rst)
            running <= 1'b0;
        else if (csr_start)
            running <= 1'b1;
    end

    // Combined reset for NPU datapath
    wire dp_rst_n = rst_n && !csr_rst;

    // ================================================================
    // CSR instance
    // ================================================================
    logic npu_busy;
    logic npu_going_idle;

    csr u_csr (
        .clk,
        .rst_n,
        .addr          (csr_addr),
        .wdata         (csr_wdata),
        .we            (csr_we),
        .rdata         (csr_rdata),
        .npu_busy      (npu_busy),
        .npu_going_idle(npu_going_idle),
        .npu_start     (csr_start),
        .npu_rst       (csr_rst),
        .dma_ext_addr  (csr_dma_ext_addr),
        .dma_sram_len  (csr_dma_sram_len),
        .irq           (irq)
    );

    // ================================================================
    // IF/ID dispatch signals
    // ================================================================
    logic        gemm_cmd_valid, valu_cmd_valid, sfu_cmd_valid, dma_cmd_valid;
    logic [31:0] gemm_cmd, valu_cmd, sfu_cmd, dma_cmd;
    logic        gemm_busy, valu_busy, sfu_busy, dma_busy;
    logic        if_stall;

    // ================================================================
    // IF/ID Pipeline
    // ================================================================
    if_id_top u_if_id (
        .clk,
        .rst_n          (dp_rst_n),
        .mem_we         (dbg_imem_we),
        .mem_addr       (dbg_imem_addr),
        .mem_wdata      (dbg_imem_wdata),
        .gemm_busy      (gemm_busy),
        .valu_busy      (valu_busy),
        .sfu_busy       (sfu_busy),
        .dma_busy       (dma_busy),
        .gemm_cmd_valid (gemm_cmd_valid),
        .valu_cmd_valid (valu_cmd_valid),
        .sfu_cmd_valid  (sfu_cmd_valid),
        .dma_cmd_valid  (dma_cmd_valid),
        .gemm_cmd       (gemm_cmd),
        .valu_cmd       (valu_cmd),
        .sfu_cmd        (sfu_cmd),
        .dma_cmd        (dma_cmd),
        .debug_pc       (debug_pc),
        .debug_instr    (debug_instr),
        .stall_if       (if_stall)
    );

    assign debug_stall = if_stall;

    // ================================================================
    // GEMM — Systolic Array
    // ================================================================
    logic               gemm_start;
    logic [7:0]         gemm_k_count;
    logic               gemm_done;
    logic [15:0][7:0]   gemm_a_in, gemm_b_in;
    logic [15:0][15:0][15:0] gemm_psum;
    logic               gemm_psum_valid;

    assign gemm_start   = gemm_cmd_valid;
    assign gemm_k_count = gemm_cmd[7:0];

    // Data path not connected yet — a_in/b_in from SRAM ping-pong (future)
    assign gemm_a_in = '0;
    assign gemm_b_in = '0;

    systolic_array #(
        .ROWS(`GEMM_ROWS), .COLS(`GEMM_COLS), .K_MAX(`GEMM_K_MAX)
    ) u_gemm (
        .clk,
        .rst_n       (dp_rst_n),
        .a_in        (gemm_a_in),
        .a_valid     (1'b0),
        .b_in        (gemm_b_in),
        .start       (gemm_start),
        .k_count     (gemm_k_count),
        .busy        (gemm_busy),
        .done        (gemm_done),
        .psum_out    (gemm_psum),
        .psum_valid  (gemm_psum_valid)
    );

    // ================================================================
    // VALU — Vector ALU
    // ================================================================
    logic [7:0]   valu_opcode, valu_opt;
    logic [4:0]   valu_rs1, valu_rs2, valu_rd;
    logic         valu_cmd_gated;
    logic         valu_done;
    logic [63:0][7:0] valu_test_wdata;
    logic [63:0][7:0] valu_test_rdata;

    // Decode 32-bit VALU instruction
    assign valu_opcode = valu_cmd[31:24];
    assign valu_opt    = valu_cmd[27:20];
    assign valu_rd     = valu_cmd[23:16];
    assign valu_rs1    = valu_cmd[15:8];
    assign valu_rs2    = valu_cmd[7:0];
    assign valu_cmd_gated = valu_cmd_valid;

    // Reshape 512-bit flat ↔ 64×8-bit packed for VALU test interface
    genvar vi;
    generate
        for (vi = 0; vi < 64; vi++) begin : gen_valu_pack
            assign valu_test_wdata[vi] = dbg_valu_wdata_flat[vi*8 +: 8];
            assign dbg_valu_rdata_flat[vi*8 +: 8] = valu_test_rdata[vi];
        end
    endgenerate

    valu_top u_valu (
        .clk,
        .rst_n       (dp_rst_n),
        .cmd_valid   (valu_cmd_gated),
        .opcode      (valu_opcode),
        .opt         (valu_opt),
        .rs1         (valu_rs1),
        .rs2         (valu_rs2),
        .rd          (valu_rd),
        .busy        (valu_busy),
        .done        (valu_done),
        .test_wen    (dbg_valu_wen),
        .test_waddr  (dbg_valu_waddr),
        .test_wdata  (valu_test_wdata),
        .test_raddr  (dbg_valu_raddr),
        .test_rdata  (valu_test_rdata)
    );

    // ================================================================
    // SFU — Special Function Unit
    // ================================================================
    logic [7:0]  sfu_opcode;
    logic        sfu_valid_in_gated;
    logic [7:0]  sfu_x_in;
    logic        sfu_valid_out;
    logic [7:0]  sfu_y_out;

    assign sfu_opcode = sfu_cmd[31:24];
    assign sfu_x_in   = sfu_cmd[7:0];
    assign sfu_valid_in_gated = sfu_cmd_valid;

    sfu_top u_sfu (
        .clk,
        .rst_n       (dp_rst_n),
        .opcode      (sfu_opcode),
        .x_in        (sfu_x_in),
        .zp          (8'd0),
        .scale_mul   (16'd1),
        .scale_shr   (8'd0),
        .valid_in    (sfu_valid_in_gated),
        .y_out       (sfu_y_out),
        .valid_out   (sfu_valid_out)
    );

    // SFU busy tracking: 3-cycle pipeline shift register
    logic [2:0] sfu_pipe;
    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            sfu_pipe <= 3'd0;
        else
            sfu_pipe <= {sfu_pipe[1:0], sfu_valid_in_gated};
    end
    assign sfu_busy = |sfu_pipe;

    // ================================================================
    // DMA — Direct Memory Access
    // ================================================================
    logic        dma_start;
    logic [7:0]  dma_opcode;
    logic        dma_done;

    assign dma_start  = dma_cmd_valid;
    assign dma_opcode = dma_cmd[31:24];

    npu_dma u_dma (
        .clk,
        .rst_n         (dp_rst_n),
        .start         (dma_start),
        .opcode        (dma_opcode),
        .ext_addr      (csr_dma_ext_addr),
        .sram_addr     (csr_dma_sram_len[31:16]),
        .length        (csr_dma_sram_len[15:0]),
        .busy          (dma_busy),
        .done          (dma_done),
        .pp_bank       (),
        .pp_ready      (),
        .sim_sram_en   (1'b0),
        .sim_sram_we   (1'b0),
        .sim_sram_addr (16'd0),
        .sim_sram_wdata(64'd0),
        .sim_sram_rdata(),
        .sim_ext_en    (1'b0),
        .sim_ext_we    (1'b0),
        .sim_ext_addr  (32'd0),
        .sim_ext_wdata (64'd0),
        .sim_ext_rdata ()
    );

    // ================================================================
    // Crossbar — 3-master x 4-slave interconnect to SRAM banks
    // ================================================================
    // Master 0: DMA (future)
    // Master 1: GEMM (future)
    // Master 2: VALU/SFU (future)
    logic [31:0] xbar_m0_rdata, xbar_m1_rdata, xbar_m2_rdata;
    logic        xbar_m0_grant, xbar_m1_grant, xbar_m2_grant;

    crossbar u_crossbar (
        .clk,
        .rst_n       (dp_rst_n),
        .m0_req      (1'b0),
        .m0_addr     (16'd0),
        .m0_wdata    (32'd0),
        .m0_wen      (1'b0),
        .m0_rdata    (xbar_m0_rdata),
        .m0_grant    (xbar_m0_grant),
        .m1_req      (1'b0),
        .m1_addr     (16'd0),
        .m1_wdata    (32'd0),
        .m1_wen      (1'b0),
        .m1_rdata    (xbar_m1_rdata),
        .m1_grant    (xbar_m1_grant),
        .m2_req      (1'b0),
        .m2_addr     (16'd0),
        .m2_wdata    (32'd0),
        .m2_wen      (1'b0),
        .m2_rdata    (xbar_m2_rdata),
        .m2_grant    (xbar_m2_grant)
    );

    // ================================================================
    // Ping-pong buffer controllers
    // ================================================================
    // GEMM A buffer: 256B × 2 banks
    logic pp_gemm_a_fill, pp_gemm_a_consume;
    logic pp_gemm_a_fill_bank, pp_gemm_a_active_bank, pp_gemm_a_ready;

    pingpong #(.BUF_SIZE(`PP_GEMM_A_SIZE)) u_pp_gemm_a (
        .clk,
        .rst_n        (dp_rst_n),
        .fill_done    (pp_gemm_a_fill),
        .consume_done (pp_gemm_a_consume),
        .fill_bank    (pp_gemm_a_fill_bank),
        .active_bank  (pp_gemm_a_active_bank),
        .ready        (pp_gemm_a_ready)
    );

    // GEMM B buffer: 256B × 2 banks
    logic pp_gemm_b_fill, pp_gemm_b_consume;
    logic pp_gemm_b_fill_bank, pp_gemm_b_active_bank, pp_gemm_b_ready;

    pingpong #(.BUF_SIZE(`PP_GEMM_B_SIZE)) u_pp_gemm_b (
        .clk,
        .rst_n        (dp_rst_n),
        .fill_done    (pp_gemm_b_fill),
        .consume_done (pp_gemm_b_consume),
        .fill_bank    (pp_gemm_b_fill_bank),
        .active_bank  (pp_gemm_b_active_bank),
        .ready        (pp_gemm_b_ready)
    );

    // GEMM P (partial sum) buffer: 512B × 2 banks
    logic pp_gemm_p_fill, pp_gemm_p_consume;
    logic pp_gemm_p_fill_bank, pp_gemm_p_active_bank, pp_gemm_p_ready;

    pingpong #(.BUF_SIZE(`PP_GEMM_P_SIZE)) u_pp_gemm_p (
        .clk,
        .rst_n        (dp_rst_n),
        .fill_done    (pp_gemm_p_fill),
        .consume_done (pp_gemm_p_consume),
        .fill_bank    (pp_gemm_p_fill_bank),
        .active_bank  (pp_gemm_p_active_bank),
        .ready        (pp_gemm_p_ready)
    );

    // VALU buffer: 256B × 2 banks
    logic pp_valu_fill, pp_valu_consume;
    logic pp_valu_fill_bank, pp_valu_active_bank, pp_valu_ready;

    pingpong #(.BUF_SIZE(`PP_VALU_SIZE)) u_pp_valu (
        .clk,
        .rst_n        (dp_rst_n),
        .fill_done    (pp_valu_fill),
        .consume_done (pp_valu_consume),
        .fill_bank    (pp_valu_fill_bank),
        .active_bank  (pp_valu_active_bank),
        .ready        (pp_valu_ready)
    );

    // SFU buffer: 256B × 2 banks
    logic pp_sfu_fill, pp_sfu_consume;
    logic pp_sfu_fill_bank, pp_sfu_active_bank, pp_sfu_ready;

    pingpong #(.BUF_SIZE(`PP_SFU_SIZE)) u_pp_sfu (
        .clk,
        .rst_n        (dp_rst_n),
        .fill_done    (pp_sfu_fill),
        .consume_done (pp_sfu_consume),
        .fill_bank    (pp_sfu_fill_bank),
        .active_bank  (pp_sfu_active_bank),
        .ready        (pp_sfu_ready)
    );

    // Ping-pong control signals tied off until load/store units exist
    assign pp_gemm_a_fill    = 1'b0;
    assign pp_gemm_a_consume = gemm_done;
    assign pp_gemm_b_fill    = 1'b0;
    assign pp_gemm_b_consume = gemm_done;
    assign pp_gemm_p_fill    = gemm_psum_valid;
    assign pp_gemm_p_consume = 1'b0;
    assign pp_valu_fill      = 1'b0;
    assign pp_valu_consume   = valu_done;
    assign pp_sfu_fill       = 1'b0;
    assign pp_sfu_consume    = sfu_valid_out;

    // ================================================================
    // Global busy aggregation
    // ================================================================
    assign npu_busy = gemm_busy || valu_busy || sfu_busy || dma_busy;

    // going_idle: pulsed when any engine that is currently busy asserts done,
    // meaning it will go idle on the next cycle.  Used by CSR for IRQ.
    assign npu_going_idle = (gemm_busy && gemm_done)  ||
                            (valu_busy && valu_done)  ||
                            (dma_busy  && dma_done)   ||
                            (sfu_busy  && sfu_valid_out);

endmodule
