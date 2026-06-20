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
    logic        csr_dma_start;
    logic        csr_dma_is_store;
    logic [31:0] csr_desc_ptr;

    // ================================================================
    // Running flag
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

    wire dp_rst_n = rst_n && !csr_rst;

    // ================================================================
    // CSR instance
    // ================================================================
    logic npu_busy;
    logic npu_going_idle;

    csr u_csr (
        .clk,
        .rst_n,
        .addr            (csr_addr),
        .wdata           (csr_wdata),
        .we              (csr_we),
        .rdata           (csr_rdata),
        .npu_busy        (npu_busy),
        .npu_going_idle  (npu_going_idle),
        .npu_start       (csr_start),
        .npu_rst         (csr_rst),
        .dma_ext_addr    (csr_dma_ext_addr),
        .dma_sram_addr   (csr_dma_sram_addr),
        .dma_length      (csr_dma_length),
        .dma_csr_start   (csr_dma_start),
        .dma_csr_is_store(csr_dma_is_store),
        .desc_ptr        (csr_desc_ptr),
        .irq             (irq)
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

    logic        gpl_gemm_start;
    logic        gpl_feeding;

    assign gemm_start   = gpl_gemm_start;
    assign gemm_k_count = gemm_cmd[7:0];

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
    logic [7:0]  dma_opcode_latched;
    logic        dma_done;

    // Bridge state type
    typedef enum logic [1:0] {
        DMA_BR_IDLE   = 2'd0,
        DMA_BR_COPY   = 2'd1,
        DMA_BR_PREFILL= 2'd2
    } dma_br_state_t;

    dma_br_state_t dma_br_state, dma_br_next;

    wire prefill_entering = (dma_br_state == DMA_BR_IDLE) && (csr_dma_start && csr_dma_is_store);
    wire prefill_active   = (dma_br_state == DMA_BR_PREFILL);
    wire prefill_done     = prefill_active && (dma_br_next == DMA_BR_IDLE);

    logic dma_restart;
    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            dma_restart <= 1'b0;
        else
            dma_restart <= prefill_done;
    end

    assign dma_start  = dma_cmd_valid
                      || (csr_dma_start && ~prefill_entering)
                      || dma_restart;

    assign dma_opcode = dma_cmd_valid  ? dma_cmd[31:24] :
                        dma_restart    ? `OP_DMA_ST :
                        csr_dma_start  ? (csr_dma_is_store ? `OP_DMA_ST : `OP_DMA_LD) :
                        8'd0;

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n)
            dma_opcode_latched <= 8'd0;
        else if (dma_start)
            dma_opcode_latched <= dma_opcode;
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
        .busy          (dma_busy),
        .done          (dma_done),
        .pp_bank       (),
        .pp_ready      (),

        // AXI passthrough
        .m_axi_araddr  (dma_axi_araddr),
        .m_axi_arlen   (dma_axi_arlen),
        .m_axi_arvalid (dma_axi_arvalid),
        .m_axi_arready (dma_axi_arready),
        .m_axi_rdata   (dma_axi_rdata),
        .m_axi_rresp   (dma_axi_rresp),
        .m_axi_rlast   (dma_axi_rlast),
        .m_axi_rvalid  (dma_axi_rvalid),
        .m_axi_rready  (dma_axi_rready),
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

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n) begin
            dma_br_state <= DMA_BR_IDLE;
            dma_br_cnt   <= 16'd0;
        end else begin
            dma_br_state <= dma_br_next;
            case (dma_br_state)
                DMA_BR_IDLE: begin
                    dma_br_cnt <= 16'd0;
                end
                DMA_BR_PREFILL: begin
                    if (xbar_m0_grant)
                        dma_br_cnt <= dma_br_cnt + 16'd4;
                end
                DMA_BR_COPY: begin
                    if (xbar_m0_grant)
                        dma_br_cnt <= dma_br_cnt + 16'd4;
                end
            endcase
        end
    end

    always_comb begin
        dma_br_next = dma_br_state;
        case (dma_br_state)
            DMA_BR_IDLE: begin
                if (dma_done && dma_opcode_latched == `OP_DMA_LD)
                    dma_br_next = DMA_BR_COPY;
                else if (csr_dma_start && csr_dma_is_store)
                    dma_br_next = DMA_BR_PREFILL;
            end
            DMA_BR_COPY: begin
                if (dma_br_cnt >= csr_dma_length)
                    dma_br_next = DMA_BR_IDLE;
            end
            DMA_BR_PREFILL: begin
                if (dma_br_cnt >= csr_dma_length)
                    dma_br_next = DMA_BR_IDLE;
            end
        endcase
    end

    assign dma_sim_sram_en   = (dma_br_state == DMA_BR_COPY) || (dma_br_state == DMA_BR_PREFILL);
    assign dma_sim_sram_we   = (dma_br_state == DMA_BR_PREFILL);
    assign dma_sim_sram_addr = csr_dma_sram_addr + dma_br_cnt;
    assign dma_sim_sram_wdata= (dma_br_state == DMA_BR_PREFILL) ? {32'd0, xbar_m0_rdata} : 64'd0;

    // M0 (DMA) driven by bridge FSM
    assign m0_req   = (dma_br_state == DMA_BR_COPY) || (dma_br_state == DMA_BR_PREFILL);
    assign m0_addr  = csr_dma_sram_addr + dma_br_cnt;
    assign m0_wdata = (dma_br_state == DMA_BR_COPY) ? dma_sim_sram_rdata[31:0] : 32'd0;
    assign m0_wen   = (dma_br_state == DMA_BR_COPY);

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
    typedef enum logic [2:0] {
        GPL_IDLE      = 3'd0,
        GPL_LOAD_B0   = 3'd1,
        GPL_LOAD_B1   = 3'd2,
        GPL_LOAD_B2   = 3'd3,
        GPL_LOAD_B3   = 3'd4,
        GPL_LOAD_A    = 3'd5,
        GPL_START     = 3'd6,
        GPL_WAIT      = 3'd7
    } gpl_state_t;

    gpl_state_t  gpl_state, gpl_next;
    logic [3:0]  gpl_row;
    logic [1:0]  gpl_word;
    logic [127:0] gpl_b_buf;

    logic [127:0] gpl_a_row0, gpl_a_row1,  gpl_a_row2,  gpl_a_row3;
    logic [127:0] gpl_a_row4, gpl_a_row5,  gpl_a_row6,  gpl_a_row7;
    logic [127:0] gpl_a_row8, gpl_a_row9,  gpl_a_row10, gpl_a_row11;
    logic [127:0] gpl_a_row12,gpl_a_row13, gpl_a_row14, gpl_a_row15;

    logic [3:0]  gpl_feed_row;

    logic [15:0] gpl_read_addr;
    always_comb begin
        gpl_read_addr = 16'd0;
        case (gpl_state)
            GPL_LOAD_B0, GPL_LOAD_B1, GPL_LOAD_B2, GPL_LOAD_B3:
                gpl_read_addr = `WSRAM_BASE + {12'd0, gpl_word, 2'd0};
            GPL_LOAD_A:
                gpl_read_addr = `ASRAM_BASE
                              + {gpl_row, 4'd0}
                              + {10'd0, gpl_word, 2'd0};
            default: gpl_read_addr = 16'd0;
        endcase
    end

    always_ff @(posedge clk) begin
        if (gpl_state == GPL_LOAD_A && xbar_m1_grant) begin
            case (gpl_row)
                4'd0:  gpl_a_row0[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd1:  gpl_a_row1[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd2:  gpl_a_row2[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd3:  gpl_a_row3[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd4:  gpl_a_row4[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd5:  gpl_a_row5[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd6:  gpl_a_row6[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd7:  gpl_a_row7[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd8:  gpl_a_row8[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd9:  gpl_a_row9[gpl_word*32 +: 32]  <= xbar_m1_rdata;
                4'd10: gpl_a_row10[gpl_word*32 +: 32] <= xbar_m1_rdata;
                4'd11: gpl_a_row11[gpl_word*32 +: 32] <= xbar_m1_rdata;
                4'd12: gpl_a_row12[gpl_word*32 +: 32] <= xbar_m1_rdata;
                4'd13: gpl_a_row13[gpl_word*32 +: 32] <= xbar_m1_rdata;
                4'd14: gpl_a_row14[gpl_word*32 +: 32] <= xbar_m1_rdata;
                4'd15: gpl_a_row15[gpl_word*32 +: 32] <= xbar_m1_rdata;
            endcase
        end
    end

    always_comb begin
        case (gpl_feed_row)
            4'd0:  gemm_a_in = gpl_a_row0;
            4'd1:  gemm_a_in = gpl_a_row1;
            4'd2:  gemm_a_in = gpl_a_row2;
            4'd3:  gemm_a_in = gpl_a_row3;
            4'd4:  gemm_a_in = gpl_a_row4;
            4'd5:  gemm_a_in = gpl_a_row5;
            4'd6:  gemm_a_in = gpl_a_row6;
            4'd7:  gemm_a_in = gpl_a_row7;
            4'd8:  gemm_a_in = gpl_a_row8;
            4'd9:  gemm_a_in = gpl_a_row9;
            4'd10: gemm_a_in = gpl_a_row10;
            4'd11: gemm_a_in = gpl_a_row11;
            4'd12: gemm_a_in = gpl_a_row12;
            4'd13: gemm_a_in = gpl_a_row13;
            4'd14: gemm_a_in = gpl_a_row14;
            4'd15: gemm_a_in = gpl_a_row15;
        endcase
    end

    always_ff @(posedge clk or negedge dp_rst_n) begin
        if (!dp_rst_n) begin
            gpl_state     <= GPL_IDLE;
            gpl_row       <= 4'd0;
            gpl_word      <= 2'd0;
            gpl_b_buf     <= 128'd0;
            gpl_gemm_start<= 1'b0;
            gpl_feed_row  <= 4'd0;
            gpl_feeding   <= 1'b0;
            gpl_a_row0    <= 128'd0; gpl_a_row1  <= 128'd0;
            gpl_a_row2    <= 128'd0; gpl_a_row3  <= 128'd0;
            gpl_a_row4    <= 128'd0; gpl_a_row5  <= 128'd0;
            gpl_a_row6    <= 128'd0; gpl_a_row7  <= 128'd0;
            gpl_a_row8    <= 128'd0; gpl_a_row9  <= 128'd0;
            gpl_a_row10   <= 128'd0; gpl_a_row11 <= 128'd0;
            gpl_a_row12   <= 128'd0; gpl_a_row13 <= 128'd0;
            gpl_a_row14   <= 128'd0; gpl_a_row15 <= 128'd0;
        end else begin
            gpl_state <= gpl_next;
            gpl_gemm_start <= 1'b0;
            case (gpl_state)
                GPL_IDLE: begin
                    gpl_row   <= 4'd0;
                    gpl_word  <= 2'd0;
                    gpl_feeding <= 1'b0;
                    gpl_feed_row <= 4'd0;
                end
                GPL_LOAD_B0, GPL_LOAD_B1, GPL_LOAD_B2, GPL_LOAD_B3: begin
                    if (xbar_m1_grant) begin
                        case (gpl_state)
                            GPL_LOAD_B0: gpl_b_buf[31:0]   <= xbar_m1_rdata;
                            GPL_LOAD_B1: gpl_b_buf[63:32]  <= xbar_m1_rdata;
                            GPL_LOAD_B2: gpl_b_buf[95:64]  <= xbar_m1_rdata;
                            GPL_LOAD_B3: gpl_b_buf[127:96] <= xbar_m1_rdata;
                            default: ;
                        endcase
                    end
                end
                GPL_LOAD_A: begin
                    if (xbar_m1_grant) begin
                        if (gpl_word == 2'd3) begin
                            if (gpl_row < 4'd15)
                                gpl_row <= gpl_row + 4'd1;
                            gpl_word <= 2'd0;
                        end else begin
                            gpl_word <= gpl_word + 2'd1;
                        end
                    end
                end
                GPL_START: begin
                    gpl_gemm_start <= 1'b1;
                    gpl_feeding    <= 1'b1;
                    gpl_feed_row   <= 4'd0;
                end
                GPL_WAIT: begin
                    if (gemm_busy && gpl_feed_row < 4'd15)
                        gpl_feed_row <= gpl_feed_row + 4'd1;
                    if (gemm_done)
                        gpl_feeding <= 1'b0;
                end
            endcase
        end
    end

    always_comb begin
        gpl_next = gpl_state;
        case (gpl_state)
            GPL_IDLE:    if (gemm_cmd_valid)               gpl_next = GPL_LOAD_B0;
            GPL_LOAD_B0: if (xbar_m1_grant)                gpl_next = GPL_LOAD_B1;
            GPL_LOAD_B1: if (xbar_m1_grant)                gpl_next = GPL_LOAD_B2;
            GPL_LOAD_B2: if (xbar_m1_grant)                gpl_next = GPL_LOAD_B3;
            GPL_LOAD_B3: if (xbar_m1_grant)                gpl_next = GPL_LOAD_A;
            GPL_LOAD_A:  if (xbar_m1_grant && gpl_word == 2'd3 && gpl_row == 4'd15)
                                                            gpl_next = GPL_START;
            GPL_START:                                     gpl_next = GPL_WAIT;
            GPL_WAIT:    if (gemm_done)                     gpl_next = GPL_IDLE;
            default:     gpl_next = GPL_IDLE;
        endcase
    end

    assign m1_req   = (gpl_state == GPL_LOAD_B0 || gpl_state == GPL_LOAD_B1 ||
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

    wire [31:0] wb_word_arr [0:127];
    genvar wi, wj;
    generate
        for (wi = 0; wi < 16; wi++) begin : wb_row_gen
            for (wj = 0; wj < 8; wj++) begin : wb_col_gen
                assign wb_word_arr[wi*8 + wj] = {
                    wb_psum_buf[wi][wj*2 + 1],
                    wb_psum_buf[wi][wj*2]
                };
            end
        end
    endgenerate

    assign gemm_wb_addr  = `OSRAM_BASE + {gemm_wb_cnt, 2'b00};
    assign gemm_wb_wdata = wb_word_arr[gemm_wb_cnt];

    // M2 is exclusively used by GEMM writeback. VALU/SFU have no
    // read-data path from M2; including them in m2_req causes useless
    // OSRAM reads that contend with GEMM writeback and DMA bridge.
    assign m2_req   = gemm_wb_active;
    assign m2_addr  = gemm_wb_addr;
    assign m2_wdata = gemm_wb_wdata;
    assign m2_wen   = gemm_wb_active;

    assign gemm_b_in = gpl_b_buf[127:0];

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

    assign npu_going_idle = (gemm_busy && gemm_done)  ||
                            (valu_busy && valu_done)  ||
                            (dma_busy  && dma_done)   ||
                            (sfu_busy  && sfu_valid_out);

endmodule
