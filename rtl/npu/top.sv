// npu_top.sv — NPU Top-Level Integration
//
// Instantiates and wires:
//   IF/ID → Dispatch → {GEMM, VALU, SFU, DMA} → Crossbar → SRAM banks
//   CSR register file, Ping-pong controllers
//
// DMA AXI master interface passes through to the SoC level for
// connection to ext_mem_model.

`include "npu_defines.svh"
`include "isa_defines.svh"

module npu_top #(
    parameter bit DMA_STANDALONE = 0
) (
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
    output logic        irq,

    // ---- DMA AXI4 Master Interface (passthrough to SoC) ----
    // Read address
    output logic [31:0] dma_axi_araddr,
    output logic [7:0]  dma_axi_arlen,
    output logic        dma_axi_arvalid,
    input  logic        dma_axi_arready,
    // Read data
    input  logic [63:0] dma_axi_rdata,
    input  logic [1:0]  dma_axi_rresp,
    input  logic        dma_axi_rlast,
    input  logic        dma_axi_rvalid,
    output logic        dma_axi_rready,
    // Write address
    output logic [31:0] dma_axi_awaddr,
    output logic [7:0]  dma_axi_awlen,
    output logic        dma_axi_awvalid,
    input  logic        dma_axi_awready,
    // Write data
    output logic [63:0] dma_axi_wdata,
    output logic [7:0]  dma_axi_wstrb,
    output logic        dma_axi_wlast,
    output logic        dma_axi_wvalid,
    input  logic        dma_axi_wready,
    // Write response
    input  logic [1:0]  dma_axi_bresp,
    input  logic        dma_axi_bvalid,
    output logic        dma_axi_bready
);

    // ================================================================
    // CSR signals
    // ================================================================
    logic        csr_start;
    logic        csr_rst;
    logic [31:0] csr_dma_ext_addr;
    logic [15:0] csr_dma_sram_addr;
    logic [15:0] csr_dma_length;
    logic [15:0] csr_dma_row_count;
    logic [15:0] csr_dma_row_bytes;
    logic [15:0] csr_dma_ext_stride;
    logic [15:0] csr_dma_sram_stride;
    logic        csr_dma_start;
    logic        csr_dma_is_store;
    logic        csr_dma_is_2d;
    logic [31:0] csr_desc_ptr;
    logic [7:0]  csr_issue_opcode;
    logic        csr_halt;
    logic        csr_pc_we;
    logic [7:0]  csr_pc_wdata;
    logic [7:0]  if_current_pc;
    logic        illegal_cmd_valid;
    logic        dma_error;

    // ================================================================
    // Running flag
    // ================================================================
    logic running;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            running <= 1'b0;
        else if (illegal_cmd_valid)
            running <= 1'b0;
        else if (csr_start)
            running <= 1'b1;
        else if (csr_rst)
            running <= 1'b0;
    end

    wire dp_rst_n = rst_n && !csr_rst;

    // ================================================================
    // CSR instance
    // ================================================================
    logic npu_busy;
    logic npu_going_idle;
    logic perf_gemm_busy;
    logic perf_valu_busy;
    logic perf_sfu_busy;
    logic perf_dma_busy;

    // ================================================================
    // Debug signal pack for CSR DEBUG register (0x60)
    // (declaration up here so it is visible to the CSR instance;
    //  the actual assignment lives at the bottom after all sub-signals)
    // ================================================================
    logic [31:0] debug_signals;

    csr u_csr (
        .clk,
        .rst_n,
        .addr            (csr_addr),
        .wdata           (csr_wdata),
        .we              (csr_we),
        .rdata           (csr_rdata),
        .npu_busy        (npu_busy),
        .npu_going_idle  (npu_going_idle),
        .dma_err_event   (dma_error),
        .ill_insn_event  (illegal_cmd_valid),
        .perf_busy       (npu_busy),
        .perf_gemm_busy  (perf_gemm_busy),
        .perf_valu_busy  (perf_valu_busy),
        .perf_sfu_busy   (perf_sfu_busy),
        .perf_dma_busy   (perf_dma_busy),
        .current_pc      (if_current_pc),
        .debug_signals   (debug_signals),
        .npu_start       (csr_start),
        .npu_rst         (csr_rst),
        .npu_halt        (csr_halt),
        .pc_we           (csr_pc_we),
        .pc_wdata        (csr_pc_wdata),
        .issue_opcode    (csr_issue_opcode),
        .dma_ext_addr    (csr_dma_ext_addr),
        .dma_sram_addr   (csr_dma_sram_addr),
        .dma_length      (csr_dma_length),
        .dma_row_count   (csr_dma_row_count),
        .dma_row_bytes   (csr_dma_row_bytes),
        .dma_ext_stride  (csr_dma_ext_stride),
        .dma_sram_stride (csr_dma_sram_stride),
        .dma_csr_start   (csr_dma_start),
        .dma_csr_is_store(csr_dma_is_store),
        .dma_csr_is_2d   (csr_dma_is_2d),
        .desc_ptr        (csr_desc_ptr),
        .irq             (irq)
    );

    // ================================================================
    // IF/ID dispatch signals
    // ================================================================
    logic        gemm_cmd_valid, valu_cmd_valid, sfu_cmd_valid, dma_cmd_valid;
    logic [31:0] gemm_cmd, valu_cmd, sfu_cmd, dma_cmd;
    logic        gemm_busy, valu_busy, sfu_busy, dma_busy;
    logic        ifid_gemm_busy;
    logic        ifid_dma_busy;
    logic        if_stall;
    logic        gemm_issue_valid;
    logic [31:0] gemm_issue_cmd;
    logic [31:0] gemm_issue_cmd_latched;
    logic        if_refill_req;
    logic [31:0] if_refill_ext_addr;
    logic        if_refill_busy;
    logic        if_refill_valid;
    logic [31:0] if_refill_data;

    // ================================================================
    // IF/ID Pipeline
    // ================================================================
    if_id_top u_if_id (
        .clk,
        .rst_n          (dp_rst_n),
        .mem_we         (dbg_imem_we),
        .mem_addr       (dbg_imem_addr),
        .mem_wdata      (dbg_imem_wdata),
        .instr_base_addr (csr_desc_ptr),
        .pc_we          (csr_pc_we),
        .pc_wdata       (csr_pc_wdata),
        .halt           (csr_halt || !running),
        .gemm_busy      (ifid_gemm_busy),
        .valu_busy      (valu_busy),
        .sfu_busy       (sfu_busy),
        .dma_busy       (ifid_dma_busy),
        .refill_req     (if_refill_req),
        .refill_ext_addr(if_refill_ext_addr),
        .refill_valid   (if_refill_valid),
        .refill_data    (if_refill_data),
        .refill_busy    (if_refill_busy),
        .gemm_cmd_valid (gemm_cmd_valid),
        .valu_cmd_valid (valu_cmd_valid),
        .sfu_cmd_valid  (sfu_cmd_valid),
        .dma_cmd_valid  (dma_cmd_valid),
        .illegal_cmd_valid(illegal_cmd_valid),
        .gemm_cmd       (gemm_cmd),
        .valu_cmd       (valu_cmd),
        .sfu_cmd        (sfu_cmd),
        .dma_cmd        (dma_cmd),
        .debug_pc       (debug_pc),
        .current_pc     (if_current_pc),
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
    logic signed [15:0][8:0] gemm_a_in, gemm_b_in;
    logic [15:0][15:0][15:0] gemm_psum;
    logic               gemm_psum_valid;

    logic        gpl_gemm_start;
    logic        gpl_feeding;
    logic        csr_gemm_start;
    logic        if_gemm_desc_start;
    logic        gemm_issue_desc_fetch;
    logic [15:0] gemm_issue_desc_ptr;

    assign csr_gemm_start   = csr_start && (csr_issue_opcode == `OP_GEMM);
    assign if_gemm_desc_start = gemm_cmd_valid &&
                                ((gemm_cmd[31:24] == `OP_GEMM) ||
                                 (gemm_cmd[31:24] == `OP_GEMM_SCALE));
    assign gemm_issue_valid = gemm_cmd_valid || csr_gemm_start;
    assign gemm_issue_cmd   = csr_gemm_start ? {`OP_GEMM, 16'd0, 8'd16} : gemm_cmd;
    assign gemm_issue_desc_fetch = csr_gemm_start || if_gemm_desc_start;
    assign gemm_issue_desc_ptr = csr_gemm_start
                               ? csr_desc_ptr[15:0]
                               : (csr_desc_ptr[15:0] + {gemm_cmd[21:8], 2'b00});
    assign gemm_start       = gpl_gemm_start;

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            gemm_issue_cmd_latched <= 32'd0;
        else if (gemm_issue_valid)
            gemm_issue_cmd_latched <= gemm_issue_cmd;
    end

    systolic_array #(
        .ROWS(`GEMM_ROWS), .COLS(`GEMM_COLS), .K_MAX(`GEMM_K_MAX)
    ) u_gemm (
        .clk,
        .rst_n       (dp_rst_n),
        .a_in        (gemm_a_in),
        .a_valid     (gpl_feeding && gemm_busy),
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

    assign valu_opcode = valu_cmd[31:24];
    assign valu_opt    = valu_cmd[27:20];
    assign valu_rd     = valu_cmd[23:16];
    assign valu_rs1    = valu_cmd[15:8];
    assign valu_rs2    = valu_cmd[7:0];
    assign valu_cmd_gated = valu_cmd_valid;

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

    logic [3:0] sfu_pipe;
    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            sfu_pipe <= 4'd0;
        else
            sfu_pipe <= {sfu_pipe[2:0], sfu_valid_in_gated};
    end
    assign sfu_busy = |sfu_pipe;

    // ================================================================
    // DMA — Direct Memory Access
    // ================================================================
    logic        dma_start;
    logic [7:0]  dma_opcode;
    logic [7:0]  dma_opcode_latched;
    logic        dma_done;
    logic        dma_load_inflight;
    wire         dma_cmd_is_store = dma_cmd_valid && (dma_cmd[31:24] == `OP_DMA_ST);

    // Bridge state type
    typedef enum logic [1:0] {
        DMA_BR_IDLE   = 2'd0,
        DMA_BR_COPY   = 2'd1,
        DMA_BR_PREFILL= 2'd2
    } dma_br_state_t;

    dma_br_state_t dma_br_state, dma_br_next;

    wire prefill_entering = (dma_br_state == DMA_BR_IDLE) &&
                             ((csr_dma_start && csr_dma_is_store) || dma_cmd_is_store);
    wire prefill_active   = (dma_br_state == DMA_BR_PREFILL);
    wire prefill_done     = prefill_active && (dma_br_next == DMA_BR_IDLE);

    logic dma_restart;
    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            dma_restart <= 1'b0;
        else
            dma_restart <= prefill_done;
    end

    assign dma_start  = (dma_cmd_valid && ~prefill_entering)
                      || (csr_dma_start && ~prefill_entering)
                      || dma_restart;

    assign dma_opcode = dma_cmd_valid  ? dma_cmd[31:24] :
                        dma_restart    ? `OP_DMA_ST :
                        csr_dma_start  ? (csr_dma_is_store ? `OP_DMA_ST :
                                          (csr_dma_is_2d ? `OP_DMA_2D : `OP_DMA_LD)) :
                        8'd0;

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            dma_opcode_latched <= 8'd0;
        else if (dma_start)
            dma_opcode_latched <= dma_opcode;
    end

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            dma_load_inflight <= 1'b0;
        else if (dma_error)
            dma_load_inflight <= 1'b0;
        else if ((csr_dma_start && !csr_dma_is_store) ||
                 (dma_cmd_valid && (dma_cmd[31:24] != `OP_DMA_ST)))
            dma_load_inflight <= 1'b1;
        else if ((dma_br_state == DMA_BR_COPY) && (dma_br_next == DMA_BR_IDLE))
            dma_load_inflight <= 1'b0;
    end

    // DMA sim port wires (for DMA→Crossbar bridge)
    logic        dma_sim_sram_en;
    logic        dma_sim_sram_we;
    logic [15:0] dma_sim_sram_addr;
    logic [63:0] dma_sim_sram_wdata;
    logic [63:0] dma_sim_sram_rdata;
    logic        dma_sim_ext_en;
    logic        dma_sim_ext_we;
    logic [31:0] dma_sim_ext_addr;
    logic [63:0] dma_sim_ext_wdata;
    logic [63:0] dma_sim_ext_rdata;
    logic [31:0] dma_m_axi_araddr;
    logic [7:0]  dma_m_axi_arlen;
    logic        dma_m_axi_arvalid;
    logic        dma_m_axi_arready;
    logic [63:0] dma_m_axi_rdata;
    logic [1:0]  dma_m_axi_rresp;
    logic        dma_m_axi_rlast;
    logic        dma_m_axi_rvalid;
    logic        dma_m_axi_rready;

    npu_dma #(
        .STANDALONE (DMA_STANDALONE)
    ) u_dma (
        .clk,
        .rst_n         (dp_rst_n),
        .start         (dma_start),
        .opcode        (dma_opcode),
        .ext_addr      (csr_dma_ext_addr),
        .sram_addr     (csr_dma_sram_addr),
        .length        (csr_dma_length),
        .row_count     (csr_dma_row_count),
        .row_bytes     (csr_dma_row_bytes),
        .ext_stride    (csr_dma_ext_stride),
        .sram_stride   (csr_dma_sram_stride),
        .busy          (dma_busy),
        .done          (dma_done),
        .error         (dma_error),
        .pp_bank       (),
        .pp_ready      (),

        // AXI passthrough
        .m_axi_araddr  (dma_m_axi_araddr),
        .m_axi_arlen   (dma_m_axi_arlen),
        .m_axi_arvalid (dma_m_axi_arvalid),
        .m_axi_arready (dma_m_axi_arready),
        .m_axi_rdata   (dma_m_axi_rdata),
        .m_axi_rresp   (dma_m_axi_rresp),
        .m_axi_rlast   (dma_m_axi_rlast),
        .m_axi_rvalid  (dma_m_axi_rvalid),
        .m_axi_rready  (dma_m_axi_rready),
        .m_axi_awaddr  (dma_axi_awaddr),
        .m_axi_awlen   (dma_axi_awlen),
        .m_axi_awvalid (dma_axi_awvalid),
        .m_axi_awready (dma_axi_awready),
        .m_axi_wdata   (dma_axi_wdata),
        .m_axi_wstrb   (dma_axi_wstrb),
        .m_axi_wlast   (dma_axi_wlast),
        .m_axi_wvalid  (dma_axi_wvalid),
        .m_axi_wready  (dma_axi_wready),
        .m_axi_bresp   (dma_axi_bresp),
        .m_axi_bvalid  (dma_axi_bvalid),
        .m_axi_bready  (dma_axi_bready),

        // Debug
        .sim_sram_en   (dma_sim_sram_en),
        .sim_sram_we   (dma_sim_sram_we),
        .sim_sram_addr (dma_sim_sram_addr),
        .sim_sram_wdata(dma_sim_sram_wdata),
        .sim_sram_rdata(dma_sim_sram_rdata),
        .sim_ext_en    (dma_sim_ext_en),
        .sim_ext_we    (dma_sim_ext_we),
        .sim_ext_addr  (dma_sim_ext_addr),
        .sim_ext_wdata (dma_sim_ext_wdata),
        .sim_ext_rdata (dma_sim_ext_rdata)
    );

    // IF/ID instruction refill uses the AXI read channel when the DMA engine
    // is not already issuing a read.  One 32-instruction refill block maps to
    // sixteen 64-bit AXI beats.
    typedef enum logic [2:0] {
        IFR_IDLE   = 3'd0,
        IFR_AR     = 3'd1,
        IFR_R_LO   = 3'd2,
        IFR_OUT_LO = 3'd3,
        IFR_OUT_HI = 3'd4
    } ifr_state_t;

    ifr_state_t ifr_state;
    logic [31:0] ifr_addr;
    logic [4:0]  ifr_word_idx;
    logic [63:0] ifr_beat_data;
    wire         ifr_active = (ifr_state != IFR_IDLE);
    wire         ifr_can_start = if_refill_req &&
                                 !ifr_active &&
                                 !dma_busy &&
                                 !dma_load_inflight &&
                                 (dma_br_state == DMA_BR_IDLE) &&
                                 !dma_start &&
                                 !dma_m_axi_arvalid;

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n) begin
            ifr_state        <= IFR_IDLE;
            ifr_addr         <= 32'd0;
            ifr_word_idx     <= 5'd0;
            ifr_beat_data    <= 64'd0;
        end else begin
            case (ifr_state)
                IFR_IDLE: begin
                    if (ifr_can_start) begin
                        ifr_addr     <= if_refill_ext_addr;
                        ifr_word_idx <= 5'd0;
                        ifr_state    <= IFR_AR;
                    end
                end

                IFR_AR: begin
                    if (dma_axi_arready)
                        ifr_state <= IFR_R_LO;
                end

                IFR_R_LO: begin
                    if (dma_axi_rvalid) begin
                        ifr_beat_data   <= dma_axi_rdata;
                        ifr_state       <= IFR_OUT_LO;
                    end
                end

                IFR_OUT_LO: begin
                    ifr_state <= IFR_OUT_HI;
                end

                IFR_OUT_HI: begin
                    if (ifr_word_idx == 5'd30) begin
                        ifr_word_idx <= 5'd0;
                        ifr_state    <= IFR_IDLE;
                    end else begin
                        ifr_word_idx <= ifr_word_idx + 5'd2;
                        ifr_state    <= IFR_R_LO;
                    end
                end

                default: begin
                    ifr_state <= IFR_IDLE;
                end
            endcase
        end
    end

    assign if_refill_valid = (ifr_state == IFR_OUT_LO) ||
                             (ifr_state == IFR_OUT_HI);
    assign if_refill_data  = (ifr_state == IFR_OUT_LO)
                           ? ifr_beat_data[31:0]
                           : ifr_beat_data[63:32];

    assign dma_axi_araddr  = ifr_active ? ifr_addr : dma_m_axi_araddr;
    assign dma_axi_arlen   = ifr_active ? 8'd15 : dma_m_axi_arlen;
    assign dma_axi_arvalid = (ifr_state == IFR_AR) ? 1'b1 : dma_m_axi_arvalid;
    assign dma_m_axi_arready = !ifr_active && dma_axi_arready;

    assign dma_m_axi_rdata  = dma_axi_rdata;
    assign dma_m_axi_rresp  = dma_axi_rresp;
    assign dma_m_axi_rlast  = dma_axi_rlast;
    assign dma_m_axi_rvalid = !ifr_active && dma_axi_rvalid;
    assign dma_axi_rready   = ifr_active ? (ifr_state == IFR_R_LO) : dma_m_axi_rready;

    // ================================================================
    // Crossbar — 3-master x 4-slave interconnect to SRAM banks
    // ================================================================
    logic [31:0] xbar_m0_rdata, xbar_m1_rdata, xbar_m2_rdata;
    logic        xbar_m0_grant, xbar_m1_grant, xbar_m2_grant;

    logic        m0_req, m1_req, m2_req;
    logic [15:0] m0_addr, m1_addr, m2_addr;
    logic [31:0] m0_wdata, m1_wdata, m2_wdata;
    logic        m0_wen,  m1_wen,  m2_wen;

    crossbar u_crossbar (
        .clk,
        .rst_n       (dp_rst_n),
        .m0_req      (m0_req),
        .m0_addr     (m0_addr),
        .m0_wdata    (m0_wdata),
        .m0_wen      (m0_wen),
        .m0_rdata    (xbar_m0_rdata),
        .m0_grant    (xbar_m0_grant),
        .m1_req      (m1_req),
        .m1_addr     (m1_addr),
        .m1_wdata    (m1_wdata),
        .m1_wen      (m1_wen),
        .m1_rdata    (xbar_m1_rdata),
        .m1_grant    (xbar_m1_grant),
        .m2_req      (m2_req),
        .m2_addr     (m2_addr),
        .m2_wdata    (m2_wdata),
        .m2_wen      (m2_wen),
        .m2_rdata    (xbar_m2_rdata),
        .m2_grant    (xbar_m2_grant)
    );

    // ================================================================
    // DMA → Crossbar Copy Bridge
    //
    // After a DMA LOAD completes, this FSM reads data from the DMA's
    // internal SRAM (via sim_sram) and writes it into the crossbar's
    // SRAM banks (via M0) so GEMM/VALU can access it.
    //
    // For a DMA STORE, the FSM reads from crossbar SRAM and writes
    // into DMA's internal SRAM before DMA STORE starts.
    // ================================================================
	    logic [15:0]   dma_br_cnt;
        logic [15:0]   dma_br_row;
	    logic          dma_br_phase;
	    logic          dma_br_word_active;
        logic [15:0]   dma_br_rows;
        logic [15:0]   dma_br_row_bytes;
        logic [15:0]   dma_br_sram_stride;
        logic [31:0]   dma_br_sram_addr_calc;

    always_comb begin
        dma_br_rows = (csr_dma_is_2d && csr_dma_row_count != 16'd0)
                    ? csr_dma_row_count : 16'd1;
        dma_br_row_bytes = csr_dma_is_2d ? csr_dma_row_bytes : csr_dma_length;
        dma_br_sram_stride = (csr_dma_is_2d && csr_dma_sram_stride != 16'd0)
                           ? csr_dma_sram_stride : dma_br_row_bytes;
        dma_br_sram_addr_calc = {16'd0, csr_dma_sram_addr}
                              + ({16'd0, dma_br_row} * {16'd0, dma_br_sram_stride})
                              + {16'd0, dma_br_cnt};
        dma_br_word_active = (dma_br_row < dma_br_rows) &&
                             (dma_br_cnt < dma_br_row_bytes);
    end

    always_ff @(posedge clk or negedge dp_rst_n) begin
	        if (!dp_rst_n) begin
	            dma_br_state <= DMA_BR_IDLE;
	            dma_br_cnt   <= 16'd0;
                dma_br_row   <= 16'd0;
	            dma_br_phase <= 1'b0;
	        end else begin
	            dma_br_state <= dma_br_next;
	            case (dma_br_state)
	                DMA_BR_IDLE: begin
	                    dma_br_cnt <= 16'd0;
                        dma_br_row <= 16'd0;
	                    dma_br_phase <= 1'b0;
	                end
	                DMA_BR_PREFILL: begin
	                    if (!dma_br_word_active) begin
	                        dma_br_phase <= 1'b0;
	                    end else if (!dma_br_phase) begin
	                        if (xbar_m0_grant)
	                            dma_br_phase <= 1'b1;
	                    end else begin
                            if (dma_br_cnt + 16'd4 >= dma_br_row_bytes) begin
                                dma_br_cnt <= 16'd0;
                                dma_br_row <= dma_br_row + 16'd1;
                            end else begin
	                            dma_br_cnt <= dma_br_cnt + 16'd4;
                            end
	                        dma_br_phase <= 1'b0;
	                    end
                end
                DMA_BR_COPY: begin
                    if (!dma_br_word_active) begin
                        dma_br_phase <= 1'b0;
                    end else if (!dma_br_phase) begin
                        dma_br_phase <= 1'b1;
                    end else if (xbar_m0_grant) begin
                        if (dma_br_cnt + 16'd4 >= dma_br_row_bytes) begin
                            dma_br_cnt <= 16'd0;
                            dma_br_row <= dma_br_row + 16'd1;
                        end else begin
                            dma_br_cnt <= dma_br_cnt + 16'd4;
                        end
                        dma_br_phase <= 1'b0;
                    end
                end
            endcase
        end
    end

    always_comb begin
        dma_br_next = dma_br_state;
        case (dma_br_state)
            DMA_BR_IDLE: begin
                if (dma_done && dma_load_inflight)
                    dma_br_next = DMA_BR_COPY;
                else if ((csr_dma_start && csr_dma_is_store) || dma_cmd_is_store)
                    dma_br_next = DMA_BR_PREFILL;
            end
            DMA_BR_COPY: begin
                if (dma_br_row >= dma_br_rows)
                    dma_br_next = DMA_BR_IDLE;
            end
            DMA_BR_PREFILL: begin
                if (dma_br_row >= dma_br_rows)
                    dma_br_next = DMA_BR_IDLE;
            end
        endcase
    end

    assign dma_sim_sram_en   = dma_br_word_active &&
                                (((dma_br_state == DMA_BR_COPY) && !dma_br_phase) ||
                                 ((dma_br_state == DMA_BR_PREFILL) && dma_br_phase));
    assign dma_sim_sram_we   = (dma_br_state == DMA_BR_PREFILL) && dma_br_phase;
    assign dma_sim_sram_addr = dma_br_sram_addr_calc[15:0];
	    assign dma_sim_sram_wdata= (dma_br_state == DMA_BR_PREFILL) ? {32'd0, xbar_m0_rdata} : 64'd0;

    // M0 (DMA) driven by bridge FSM
	    assign m0_req   = dma_br_word_active &&
	                      (((dma_br_state == DMA_BR_COPY) && dma_br_phase) ||
	                       (dma_br_state == DMA_BR_PREFILL));
    assign m0_addr  = dma_br_sram_addr_calc[15:0];
    assign m0_wdata = (dma_br_state == DMA_BR_COPY) ? dma_sim_sram_rdata[31:0] : 32'd0;
    assign m0_wen   = (dma_br_state == DMA_BR_COPY) && dma_br_phase;

    // DMA sim_ext ports — only used in standalone DMA tests to access
    // wrapper's internal AXI RAM.  When DMA_STANDALONE=0, the wrapper
    // ignores sim_ram_* and uses external AXI instead.
    assign dma_sim_ext_en   = 1'b0;
    assign dma_sim_ext_we   = 1'b0;
    assign dma_sim_ext_addr = 32'd0;
    assign dma_sim_ext_wdata= 64'd0;

    // ================================================================
    // GEMM Data Preloader
    // ================================================================
    typedef enum logic [3:0] {
        GPL_IDLE      = 4'd0,
        GPL_LOAD_B0   = 4'd1,
        GPL_LOAD_B1   = 4'd2,
        GPL_LOAD_B2   = 4'd3,
        GPL_LOAD_B3   = 4'd4,
        GPL_LOAD_A    = 4'd5,
        GPL_START     = 4'd6,
        GPL_WAIT      = 4'd7,
        GPL_DESC0     = 4'd8,
        GPL_DESC1     = 4'd9,
        GPL_DESC2     = 4'd10,
        GPL_DESC3     = 4'd11,
        GPL_DESC4     = 4'd12
    } gpl_state_t;

    gpl_state_t  gpl_state, gpl_next;
    logic [7:0]  gpl_row;
    logic [5:0]  gpl_word;
    logic [127:0] gpl_b_row [0:255];
    logic [2047:0] gpl_a_row [0:15];

    logic [7:0]  gpl_feed_k;
    logic [10:0] gpl_feed_byte;
    logic [1:0]  gpl_feed_phase;
    logic [127:0] gpl_b_feed_row;
    logic [1:0]  gpl_b_word;
    logic        gpl_capture_valid;
    logic        gpl_capture_is_b;
    logic [7:0]  gpl_capture_row;
    logic [5:0]  gpl_capture_word;
    logic        gpl_desc_capture_valid;
    logic [2:0]  gpl_desc_capture_word;
    logic [15:0] gpl_desc_ptr_latched;
    logic [15:0] gpl_a_base;
    logic [15:0] gpl_b_base;
    logic [15:0] gpl_o_base;
    logic [7:0]  gpl_k_count;
    logic [4:0]  gpl_k_tiles;
    logic signed [7:0] gpl_a_zp;
    logic signed [7:0] gpl_b_zp;
    logic [4:0]  gpl_out_scale_shr;
    logic signed [15:0] gpl_out_scale_mul;
    logic        gpl_relu;
    logic signed [7:0] gpl_out_zp;
    logic        gpl_desc_valid;

    assign gemm_k_count = gpl_k_count;

    logic [15:0] gpl_read_addr;
    logic [8:0]  gpl_k_bytes;
    logic [5:0]  gpl_a_word_last;
    logic        gpl_feed_last_k;
    logic        gpl_load_last_k;

    always_comb begin
        gpl_k_bytes = {gpl_k_tiles, 4'd0};
        gpl_a_word_last = gpl_k_bytes[7:2] - 6'd1;
        gpl_feed_last_k = (gpl_k_count == 8'd0) ? (gpl_feed_k == 8'hFF) :
                                                   (gpl_feed_k == (gpl_k_count - 8'd1));
        gpl_load_last_k = (gpl_k_count == 8'd0) ? (gpl_row == 8'hFF) :
                                                  (gpl_row == (gpl_k_count - 8'd1));
    end

    always_comb begin
        case (gpl_state)
            GPL_LOAD_B0: gpl_b_word = 2'd0;
            GPL_LOAD_B1: gpl_b_word = 2'd1;
            GPL_LOAD_B2: gpl_b_word = 2'd2;
            GPL_LOAD_B3: gpl_b_word = 2'd3;
            default:     gpl_b_word = 2'd0;
        endcase
    end

    always_comb begin
        gpl_read_addr = 16'd0;
        case (gpl_state)
            GPL_LOAD_B0, GPL_LOAD_B1, GPL_LOAD_B2, GPL_LOAD_B3:
                gpl_read_addr = gpl_b_base
                              + {4'd0, gpl_row, 4'd0}
                              + {10'd0, gpl_b_word, 2'd0};
            GPL_LOAD_A:
                gpl_read_addr = gpl_a_base
                              + ({8'd0, gpl_row[3:0]} * {7'd0, gpl_k_bytes})
                              + {8'd0, gpl_word, 2'd0};
            GPL_DESC0: gpl_read_addr = gpl_desc_ptr_latched;
            GPL_DESC1: gpl_read_addr = gpl_desc_ptr_latched + 16'd4;
            GPL_DESC2: gpl_read_addr = gpl_desc_ptr_latched + 16'd8;
            GPL_DESC3: gpl_read_addr = gpl_desc_ptr_latched + 16'd12;
            GPL_DESC4: gpl_read_addr = gpl_desc_ptr_latched + 16'd16;
            default: gpl_read_addr = 16'd0;
        endcase
    end

    always_comb begin
        gpl_feed_byte = {gpl_feed_k, 3'b000};
        gpl_b_feed_row = gpl_b_row[gpl_feed_k];
        for (int r = 0; r < 16; r++) begin
            logic signed [7:0] a_byte;
            a_byte = gpl_a_row[r][gpl_feed_byte +: 8];
            gemm_a_in[r] = {a_byte[7], a_byte} - {gpl_a_zp[7], gpl_a_zp};
        end
        for (int c = 0; c < 16; c++) begin
            logic signed [7:0] b_byte;
            b_byte = gpl_b_feed_row[c*8 +: 8];
            gemm_b_in[c] = {b_byte[7], b_byte} - {gpl_b_zp[7], gpl_b_zp};
        end
    end

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n) begin
            gpl_state     <= GPL_IDLE;
            gpl_row       <= 8'd0;
            gpl_word      <= 6'd0;
            gpl_gemm_start<= 1'b0;
            gpl_feed_k    <= 8'd0;
            gpl_feed_phase<= 2'd0;
            gpl_feeding   <= 1'b0;
            gpl_capture_valid <= 1'b0;
            gpl_capture_is_b  <= 1'b0;
            gpl_capture_row   <= 8'd0;
            gpl_capture_word  <= 6'd0;
            gpl_desc_capture_valid <= 1'b0;
            gpl_desc_capture_word  <= 3'd0;
            gpl_desc_ptr_latched   <= `DSRAM_BASE;
            gpl_a_base             <= `ASRAM_BASE;
            gpl_b_base             <= `WSRAM_BASE;
            gpl_o_base             <= `OSRAM_BASE;
            gpl_k_count            <= 8'd16;
            gpl_k_tiles            <= 5'd1;
            gpl_a_zp               <= 8'sd0;
            gpl_b_zp               <= 8'sd0;
            gpl_out_scale_shr      <= 5'd0;
            gpl_out_scale_mul      <= 16'sd1;
            gpl_relu               <= 1'b0;
            gpl_out_zp             <= 8'sd0;
            gpl_desc_valid         <= 1'b0;
            for (int r = 0; r < 16; r++) begin
                gpl_a_row[r] <= 128'd0;
            end
            for (int k = 0; k < 256; k++)
                gpl_b_row[k] <= 128'd0;
        end else begin
            gpl_state <= gpl_next;
            gpl_gemm_start <= 1'b0;

            // Crossbar SRAM banks are synchronous-read.  A grant issues
            // the read this cycle; the corresponding rdata is captured
            // on the next clock with this delayed metadata.
            if (gpl_capture_valid) begin
                if (gpl_capture_is_b)
                    gpl_b_row[gpl_capture_row][{gpl_capture_word, 5'b00000} +: 32] <= xbar_m1_rdata;
                else
                    gpl_a_row[gpl_capture_row][{gpl_capture_word, 5'b00000} +: 32] <= xbar_m1_rdata;
            end
            gpl_capture_valid <= 1'b0;

            if (gpl_desc_capture_valid) begin
                unique case (gpl_desc_capture_word)
                    3'd1: begin
                        if (xbar_m1_rdata[15:0] != 16'd0) begin
                            gpl_k_count <= (xbar_m1_rdata[15:0] >= 16'd16) ? 8'd0 :
                                           {xbar_m1_rdata[3:0], 4'd0};
                            gpl_k_tiles <= (xbar_m1_rdata[15:0] >= 16'd16) ? 5'd16 :
                                           {1'b0, xbar_m1_rdata[3:0]};
                            gpl_a_base   <= {xbar_m1_rdata[19:16], 12'd0};
                            gpl_b_base   <= {xbar_m1_rdata[27:24], 12'd0};
                            gpl_desc_valid <= 1'b1;
                        end
                    end
                    3'd2: begin
                        if (gpl_desc_valid) begin
                            gpl_o_base <= {xbar_m1_rdata[3:0], 12'd0};
                            gpl_a_zp <= xbar_m1_rdata[15:8];
                            gpl_b_zp <= xbar_m1_rdata[23:16];
                        end
                    end
                    3'd3: begin
                        if (gpl_desc_valid) begin
                            gpl_out_scale_shr <= (xbar_m1_rdata[23:8] > 16'd31) ? 5'd31 :
                                                 xbar_m1_rdata[12:8];
                            gpl_out_scale_mul[7:0] <= xbar_m1_rdata[31:24];
                        end
                    end
                    3'd4: begin
                        if (gpl_desc_valid) begin
                            gpl_out_scale_mul[15:8] <= xbar_m1_rdata[7:0];
                            gpl_relu <= |xbar_m1_rdata[15:8];
                            gpl_out_zp <= xbar_m1_rdata[23:16];
                        end
                    end
                    default: begin
                    end
                endcase
            end
            gpl_desc_capture_valid <= 1'b0;

            case (gpl_state)
                GPL_IDLE: begin
                    gpl_row   <= 8'd0;
                    gpl_word  <= 6'd0;
                    gpl_feeding <= 1'b0;
                    gpl_feed_k <= 8'd0;
                    gpl_feed_phase <= 2'd0;
                    if (gemm_issue_valid) begin
                        gpl_desc_ptr_latched <= gemm_issue_desc_ptr;
                        gpl_a_base           <= `ASRAM_BASE;
                        gpl_b_base           <= `WSRAM_BASE;
                        gpl_o_base           <= `OSRAM_BASE;
                        gpl_k_count          <= (gemm_issue_cmd[7:0] == 8'd0) ? 8'd16 :
                                                gemm_issue_cmd[7:0];
                        gpl_k_tiles          <= 5'd1;
                        gpl_a_zp             <= 8'sd0;
                        gpl_b_zp             <= 8'sd0;
                        gpl_out_scale_shr    <= 5'd0;
                        gpl_out_scale_mul    <= 16'sd1;
                        gpl_relu             <= 1'b0;
                        gpl_out_zp           <= 8'sd0;
                        gpl_desc_valid       <= 1'b0;
                    end
                end
                GPL_DESC0, GPL_DESC1, GPL_DESC2, GPL_DESC3, GPL_DESC4: begin
                    if (xbar_m1_grant) begin
                        gpl_desc_capture_valid <= 1'b1;
                        unique case (gpl_state)
                            GPL_DESC0: gpl_desc_capture_word <= 3'd0;
                            GPL_DESC1: gpl_desc_capture_word <= 3'd1;
                            GPL_DESC2: gpl_desc_capture_word <= 3'd2;
                            GPL_DESC3: gpl_desc_capture_word <= 3'd3;
                            default:   gpl_desc_capture_word <= 3'd4;
                        endcase
                    end
                end
                GPL_LOAD_B0, GPL_LOAD_B1, GPL_LOAD_B2, GPL_LOAD_B3: begin
                    if (xbar_m1_grant) begin
                        gpl_capture_valid <= 1'b1;
                        gpl_capture_is_b  <= 1'b1;
                        gpl_capture_row   <= gpl_row;
                        gpl_capture_word  <= {4'd0, gpl_b_word};
                        if (gpl_state == GPL_LOAD_B3) begin
                            if (!gpl_load_last_k)
                                gpl_row <= gpl_row + 8'd1;
                            else
                                gpl_row <= 8'd0;
                        end
                    end
                end
                GPL_LOAD_A: begin
                    if (xbar_m1_grant) begin
                        gpl_capture_valid <= 1'b1;
                        gpl_capture_is_b  <= 1'b0;
                        gpl_capture_row   <= gpl_row;
                        gpl_capture_word  <= gpl_word;
                        if (gpl_word == gpl_a_word_last) begin
                            if (gpl_row < 8'd15)
                                gpl_row <= gpl_row + 8'd1;
                            gpl_word <= 6'd0;
                        end else begin
                            gpl_word <= gpl_word + 6'd1;
                        end
                    end
                end
                GPL_START: begin
                    gpl_gemm_start <= 1'b1;
                    gpl_feeding    <= 1'b1;
                    gpl_feed_k     <= 8'd0;
                    gpl_feed_phase <= 2'd0;
                end
                GPL_WAIT: begin
                    if (gemm_busy) begin
                        if (gpl_feed_phase < 2'd1)
                            gpl_feed_phase <= gpl_feed_phase + 2'd1;
                        else if (!gpl_feed_last_k)
                            gpl_feed_k <= gpl_feed_k + 8'd1;
                    end
                    if (gemm_done)
                        gpl_feeding <= 1'b0;
                end
            endcase
        end
    end

    always_comb begin
        gpl_next = gpl_state;
        case (gpl_state)
            GPL_IDLE:    if (gemm_issue_valid) begin
                              if (gemm_issue_desc_fetch)
                                  gpl_next = GPL_DESC0;
                              else
                                  gpl_next = GPL_LOAD_B0;
                          end
            GPL_LOAD_B0: if (xbar_m1_grant)                gpl_next = GPL_LOAD_B1;
            GPL_LOAD_B1: if (xbar_m1_grant)                gpl_next = GPL_LOAD_B2;
            GPL_LOAD_B2: if (xbar_m1_grant)                gpl_next = GPL_LOAD_B3;
            GPL_LOAD_B3: if (xbar_m1_grant) begin
                              if (gpl_load_last_k)
                                  gpl_next = GPL_LOAD_A;
                              else
                                  gpl_next = GPL_LOAD_B0;
                          end
            GPL_LOAD_A:  if (xbar_m1_grant && gpl_word == gpl_a_word_last && gpl_row == 8'd15)
                                                            gpl_next = GPL_START;
            GPL_START:                                     gpl_next = GPL_WAIT;
            GPL_WAIT:    if (gemm_done)                     gpl_next = GPL_IDLE;
            GPL_DESC0:   if (xbar_m1_grant)                 gpl_next = GPL_DESC1;
            GPL_DESC1:   if (xbar_m1_grant)                 gpl_next = GPL_DESC2;
            GPL_DESC2:   if (xbar_m1_grant)                 gpl_next = GPL_DESC3;
            GPL_DESC3:   if (xbar_m1_grant)                 gpl_next = GPL_DESC4;
            GPL_DESC4:   if (xbar_m1_grant)                 gpl_next = GPL_LOAD_B0;
            default:     gpl_next = GPL_IDLE;
        endcase
    end

    assign m1_req   = (gpl_state == GPL_DESC0 || gpl_state == GPL_DESC1 ||
                       gpl_state == GPL_DESC2 || gpl_state == GPL_DESC3 ||
                       gpl_state == GPL_DESC4 ||
                       gpl_state == GPL_LOAD_B0 || gpl_state == GPL_LOAD_B1 ||
                       gpl_state == GPL_LOAD_B2 || gpl_state == GPL_LOAD_B3 ||
                       gpl_state == GPL_LOAD_A);
    assign m1_addr  = gpl_read_addr;
    assign m1_wdata = 32'd0;
    assign m1_wen   = 1'b0;

    // ================================================================
    // GEMM PSUM → OSRAM Writeback
    // ================================================================
    logic        gemm_wb_active;
    logic [6:0]  gemm_wb_cnt;
    logic [31:0] gemm_wb_wdata;
    logic [15:0] gemm_wb_addr;

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n) begin
            gemm_wb_active <= 1'b0;
            gemm_wb_cnt    <= 7'd0;
        end else begin
            if (gemm_psum_valid) begin
                gemm_wb_active <= 1'b1;
                gemm_wb_cnt    <= 7'd0;
            end else if (gemm_wb_active && xbar_m2_grant) begin
                if (gemm_wb_cnt == 7'd127)
                    gemm_wb_active <= 1'b0;
                else
                    gemm_wb_cnt <= gemm_wb_cnt + 7'd1;
            end
        end
    end

    logic [15:0][15:0][15:0] wb_psum_buf;
    always_ff @(posedge clk) begin
        if (gemm_psum_valid)
            wb_psum_buf <= gemm_psum;
    end

    function automatic logic [15:0] gemm_postprocess(input logic [15:0] psum);
        logic signed [15:0] psum_s;
        logic signed [31:0] scaled;
        logic signed [31:0] shifted;
        begin
            psum_s = psum;
            scaled = psum_s * gpl_out_scale_mul;
            shifted = scaled >>> gpl_out_scale_shr;
            if (gpl_relu && shifted < 32'sd0)
                shifted = 32'sd0;
            shifted = shifted + {{24{gpl_out_zp[7]}}, gpl_out_zp};

            if (shifted > 32'sd32767)
                gemm_postprocess = 16'h7FFF;
            else if (shifted < -32'sd32768)
                gemm_postprocess = 16'h8000;
            else
                gemm_postprocess = shifted[15:0];
        end
    endfunction

    wire [31:0] wb_word_arr [0:127];
    genvar wi, wj;
    generate
        for (wi = 0; wi < 16; wi++) begin : wb_row_gen
            for (wj = 0; wj < 8; wj++) begin : wb_col_gen
                assign wb_word_arr[wi*8 + wj] = {
                    gemm_postprocess(wb_psum_buf[wi][wj*2 + 1]),
                    gemm_postprocess(wb_psum_buf[wi][wj*2])
                };
            end
        end
    endgenerate

    assign gemm_wb_addr  = gpl_o_base + {gemm_wb_cnt, 2'b00};
    assign gemm_wb_wdata = wb_word_arr[gemm_wb_cnt];

    // M2 is exclusively used by GEMM writeback. VALU/SFU have no
    // read-data path from M2; including them in m2_req causes useless
    // OSRAM reads that contend with GEMM writeback and DMA bridge.
    assign m2_req   = gemm_wb_active;
    assign m2_addr  = gemm_wb_addr;
    assign m2_wdata = gemm_wb_wdata;
    assign m2_wen   = gemm_wb_active;

    // ================================================================
    // Ping-pong buffer controllers
    // ================================================================
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

    wire dma_bridge_copy_done = (dma_br_state == DMA_BR_COPY) &&
                                (dma_br_next == DMA_BR_IDLE);
    wire dma_target_asram = (csr_dma_sram_addr[15:12] == 4'h0);
    wire dma_target_wsram = (csr_dma_sram_addr[15:12] == 4'h1);
    wire dma_target_osram = (csr_dma_sram_addr[15:12] == 4'h2);

    assign pp_gemm_a_fill    = dma_bridge_copy_done && dma_target_asram;
    assign pp_gemm_a_consume = gemm_done;
    assign pp_gemm_b_fill    = dma_bridge_copy_done && dma_target_wsram;
    assign pp_gemm_b_consume = gemm_done;
    assign pp_gemm_p_fill    = gemm_psum_valid;
    assign pp_gemm_p_consume = prefill_done && dma_target_osram;
    assign pp_valu_fill      = 1'b0;
    assign pp_valu_consume   = valu_done;
    assign pp_sfu_fill       = 1'b0;
    assign pp_sfu_consume    = sfu_valid_out;

    // ================================================================
    // Global busy aggregation
    // ================================================================
    // The DMA bridge FSM (COPY/PREFILL) runs after the DMA engine's
    // dma_done pulse to copy data between DMA-internal SRAM and
    // crossbar SRAM banks.  It must be reflected in npu_busy so that
    // firmware does not start the next DMA before the bridge finishes.
    logic bridge_busy;
    logic gemm_preload_busy;
    assign bridge_busy = (dma_br_state != DMA_BR_IDLE);
    assign gemm_preload_busy = (gpl_state != GPL_IDLE);
    assign ifid_gemm_busy = gemm_cmd_valid || gemm_busy || gemm_preload_busy ||
                            gemm_psum_valid || gemm_wb_active;
    assign ifid_dma_busy = dma_cmd_valid || dma_busy || bridge_busy ||
                           dma_load_inflight || dma_restart || ifr_active;

    assign npu_busy = gemm_busy || valu_busy || sfu_busy || dma_busy ||
                      bridge_busy || dma_load_inflight || gemm_preload_busy ||
                      gemm_psum_valid || gemm_wb_active || ifr_active ||
                      if_refill_req;
    assign perf_gemm_busy = gemm_busy || gemm_preload_busy ||
                            gemm_psum_valid || gemm_wb_active;
    assign perf_valu_busy = valu_busy;
    assign perf_sfu_busy  = sfu_busy;
    assign perf_dma_busy  = dma_busy || bridge_busy || dma_load_inflight ||
                            ifr_active;

    assign npu_going_idle = (valu_busy && valu_done)  ||
                            (dma_busy  && dma_done)   ||
                            (sfu_busy  && sfu_valid_out) ||
                            (bridge_busy && dma_br_next == DMA_BR_IDLE) ||
                            (gemm_wb_active && xbar_m2_grant && gemm_wb_cnt == 7'd127);

    // Debug signal pack (assigned here after all sub-signals are declared)
    assign debug_signals = {
        16'd0,                              // [31:16] reserved
        ifr_active,                         // [15] IF refill AXI read active
        if_refill_busy,                     // [14] IF refill stream active
        if_refill_req,                      // [13] IF refill request pending
        ^if_refill_ext_addr,                // [12] consume refill address for lint
        npu_busy,                           // [11] aggregated busy
        gemm_wb_active,                     // [10] GEMM writeback FSM active
        dma_br_state,                       // [9:8] DMA bridge FSM state
        gpl_state[2:0],                     // [7:5] GEMM preloader FSM state
        bridge_busy,                        // [4] DMA bridge busy
        dma_busy,                           // [3] DMA engine busy
        sfu_busy,                           // [2] SFU busy
        valu_busy,                          // [1] VALU busy
        gemm_busy                           // [0] GEMM busy
    };

endmodule
