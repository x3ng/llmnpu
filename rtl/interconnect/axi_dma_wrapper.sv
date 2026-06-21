// ============================================================
// axi_dma_wrapper.sv — Proper AXI DMA Wrapper
//
// Uses alexforencich/verilog-axi axi_dma_rd.v (MIT) for LOAD
// and a burst AXI write FSM for STORE.
//
// STANDALONE=1 (default): internal 64 KB AXI RAM responds to
//   DMA engines — used in unit-level cocotb tests (test_dma,
//   test_dma_load) where no external memory model exists.
// STANDALONE=0: AXI master ports route to external memory
//   (ext_mem_model via a bridge in top_soc).
//
// Command interface (compatible with npu_dma):
//   start, mode (01=LOAD, 10=STORE), ext_addr[31:0],
//   length[15:0] (bytes, 8-aligned), done, xfer_active
//
// Data conduit:
//   rd_data[63:0] / rd_valid  — AXI read data presented word
//                                 at a time to npu_dma (LOAD)
//   wr_data[63:0] / wr_ready  — npu_dma presents write data
//                                 when wr_ready is asserted (STORE)
// ============================================================

`include "npu_defines.svh"

module axi_dma_wrapper #(
    parameter bit STANDALONE = 1
) (
    input  logic        clk,
    input  logic        rst_n,

    // ---- Command interface ----
    input  logic        start,
    input  logic [1:0]  mode,
    input  logic [31:0] ext_addr,
    input  logic [15:0] length,       // bytes, must be 8-aligned

    // ---- Status ----
    output logic        done,
    output logic        xfer_active,

    // ---- Data conduit to/from npu_dma ----
    output logic [63:0] rd_data,
    output logic        rd_valid,
    input  logic [63:0] wr_data,
    output logic        wr_ready,

    // ============================================================
    // AXI4 Master Interface (connected externally when STANDALONE=0)
    // ============================================================
    // Read address channel
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,

    // Read data channel
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,

    // Write address channel
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,

    // Write data channel
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,

    // Write response channel
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,

    // ---- Sim debug: direct access to internal AXI RAM (always available) ----
    input  logic        sim_ram_en,
    input  logic        sim_ram_we,
    input  logic [31:0] sim_ram_addr,
    input  logic [63:0] sim_ram_wdata,
    output logic [63:0] sim_ram_rdata
);

    // ================================================================
    // Local parameters
    // ================================================================
    localparam int AXIS_KEEP_WIDTH = `AXI_DATA_WIDTH / 8;   // 8
    localparam int LEN_WIDTH       = 16;

    // Active-high reset for reference modules
    wire rst = ~rst_n;

    // ================================================================
    // DMA-engine-side AXI bus (driven by DMA engines)
    // ================================================================
    // Master→Slave (driven by DMA engines)
    logic [31:0] dma_araddr;
    logic [7:0]  dma_arlen;
    logic        dma_arvalid;

    logic [31:0] dma_awaddr;
    logic [7:0]  dma_awlen;
    logic        dma_awvalid;

    logic [63:0] dma_wdata;
    logic [7:0]  dma_wstrb;
    logic        dma_wlast;
    logic        dma_wvalid;

    logic        dma_rready;
    logic        dma_bready;

    // Slave→Master (driven by AXI slave — either internal RAM or external)
    logic        dma_arready;
    logic [63:0] dma_rdata;
    logic [1:0]  dma_rresp;
    logic        dma_rlast;
    logic        dma_rvalid;
    logic        dma_awready;
    logic        dma_wready;
    logic [1:0]  dma_bresp;
    logic        dma_bvalid;

    // ================================================================
    // axi_dma_rd signals (read DMA engine)
    // ================================================================
    localparam int RD_DESC_ID_WIDTH   = 1;
    localparam int RD_DESC_DEST_WIDTH = 1;
    localparam int RD_DESC_USER_WIDTH = 1;

    logic [31:0]                   rd_desc_addr;
    logic [LEN_WIDTH-1:0]          rd_desc_len;
    logic [7:0]                    rd_desc_tag;
    logic [RD_DESC_ID_WIDTH-1:0]   rd_desc_id;
    logic [RD_DESC_DEST_WIDTH-1:0] rd_desc_dest;
    logic [RD_DESC_USER_WIDTH-1:0] rd_desc_user;
    logic                          rd_desc_valid;
    logic                          rd_desc_ready;

    logic [7:0]                    rd_status_tag;
    logic [3:0]                    rd_status_error;
    logic                          rd_status_valid;

    logic [`AXI_DATA_WIDTH-1:0]    rd_stream_tdata;
    logic [AXIS_KEEP_WIDTH-1:0]    rd_stream_tkeep;
    logic                          rd_stream_tvalid;
    logic                          rd_stream_tready;
    logic                          rd_stream_tlast;

    // ================================================================
    // AXI DMA Read Engine (from reference, MIT licensed)
    // ================================================================
    axi_dma_rd #(
        .AXI_DATA_WIDTH     (`AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH     (`AXI_ADDR_WIDTH),
        .AXI_STRB_WIDTH     (AXIS_KEEP_WIDTH),
        .AXI_ID_WIDTH       (1),
        .AXI_MAX_BURST_LEN  (256),
        .AXIS_DATA_WIDTH    (`AXI_DATA_WIDTH),
        .AXIS_KEEP_ENABLE   (1),
        .AXIS_KEEP_WIDTH    (AXIS_KEEP_WIDTH),
        .AXIS_LAST_ENABLE   (1),
        .AXIS_ID_ENABLE     (0),
        .AXIS_ID_WIDTH      (RD_DESC_ID_WIDTH),
        .AXIS_DEST_ENABLE   (0),
        .AXIS_DEST_WIDTH    (RD_DESC_DEST_WIDTH),
        .AXIS_USER_ENABLE   (0),
        .AXIS_USER_WIDTH    (RD_DESC_USER_WIDTH),
        .LEN_WIDTH          (LEN_WIDTH),
        .TAG_WIDTH          (8),
        .ENABLE_SG          (0),
        .ENABLE_UNALIGNED   (0)
    ) u_axi_dma_rd (
        .clk                        (clk),
        .rst                        (rst),
        .s_axis_read_desc_addr      (rd_desc_addr),
        .s_axis_read_desc_len       (rd_desc_len),
        .s_axis_read_desc_tag       (rd_desc_tag),
        .s_axis_read_desc_id        (rd_desc_id),
        .s_axis_read_desc_dest      (rd_desc_dest),
        .s_axis_read_desc_user      (rd_desc_user),
        .s_axis_read_desc_valid     (rd_desc_valid),
        .s_axis_read_desc_ready     (rd_desc_ready),
        .m_axis_read_desc_status_tag   (rd_status_tag),
        .m_axis_read_desc_status_error (rd_status_error),
        .m_axis_read_desc_status_valid (rd_status_valid),
        .m_axis_read_data_tdata     (rd_stream_tdata),
        .m_axis_read_data_tkeep     (rd_stream_tkeep),
        .m_axis_read_data_tvalid    (rd_stream_tvalid),
        .m_axis_read_data_tready    (rd_stream_tready),
        .m_axis_read_data_tlast     (rd_stream_tlast),
        .m_axis_read_data_tid       (),
        .m_axis_read_data_tdest     (),
        .m_axis_read_data_tuser     (),
        .m_axi_arid                 (),
        .m_axi_araddr               (dma_araddr),
        .m_axi_arlen                (dma_arlen),
        .m_axi_arsize               (),
        .m_axi_arburst              (),
        .m_axi_arlock               (),
        .m_axi_arcache              (),
        .m_axi_arprot               (),
        .m_axi_arvalid              (dma_arvalid),
        .m_axi_arready              (dma_arready),
        .m_axi_rid                  (1'd0),
        .m_axi_rdata                (dma_rdata),
        .m_axi_rresp                (dma_rresp),
        .m_axi_rlast                (dma_rlast),
        .m_axi_rvalid               (dma_rvalid),
        .m_axi_rready               (dma_rready),
        .enable                     (1'b1)
    );

    // Tie off unused descriptor fields
    assign rd_desc_tag  = 8'd0;
    assign rd_desc_id   = 1'd0;
    assign rd_desc_dest = 1'd0;
    assign rd_desc_user = 1'd0;

    // ================================================================
    // Wrapper FSM — coordinates read and write operations
    // ================================================================
    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_RD_START  = 3'd1,
        S_RD_XFER   = 3'd2,
        S_RD_DONE   = 3'd3,
        S_WR_AW     = 3'd4,
        S_WR_W      = 3'd5,
        S_WR_B      = 3'd6
    } state_t;

    state_t state, next;

    logic [15:0] word_cnt;
    logic [8:0]  wr_burst_words;
    logic [8:0]  wr_burst_beat;

    // Total transfer count in 64-bit words (combinational)
    wire [15:0] total_words;
    assign total_words[12:0] = length[15:3];
    assign total_words[15:13] = 3'd0;
    wire [15:0] wr_remaining_words = total_words - word_cnt;
    wire [8:0]  wr_next_burst_words =
        (wr_remaining_words >= 16'd256) ? 9'd256 :
                                          {1'b0, wr_remaining_words[7:0]};

    // ------------------------------------------------------------
    // Read descriptor drive (combinational)
    // ------------------------------------------------------------
    always_comb begin
        rd_desc_valid = 1'b0;
        rd_desc_addr  = ext_addr;
        rd_desc_len   = length;
        if (state == S_RD_START)
            rd_desc_valid = 1'b1;
    end

    // Always accept stream data (FIFO in axi_dma_rd handles buffering)
    assign rd_stream_tready = 1'b1;

    // ------------------------------------------------------------
    // Sequential FSM
    // ------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            xfer_active <= 1'b0;
            rd_data     <= 64'd0;
            rd_valid    <= 1'b0;
            word_cnt    <= 16'd0;
            wr_burst_words <= 9'd0;
            wr_burst_beat  <= 9'd0;
        end else begin
            state <= next;

            // Defaults
            done     <= 1'b0;
            rd_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    xfer_active <= 1'b0;
                    word_cnt    <= 16'd0;
                end

                S_RD_START: begin
                    xfer_active <= 1'b1;
                end

                S_RD_XFER: begin
                    xfer_active <= 1'b1;
                    if (rd_stream_tvalid) begin
                        rd_data  <= rd_stream_tdata;
                        rd_valid <= 1'b1;
                        word_cnt <= word_cnt + 16'd1;
                    end
                end

                S_RD_DONE: begin
                    xfer_active <= 1'b0;
                    done        <= 1'b1;
                end

                S_WR_AW: begin
                    xfer_active <= 1'b1;
                    if (dma_awvalid && dma_awready) begin
                        wr_burst_words <= wr_next_burst_words;
                        wr_burst_beat  <= 9'd0;
                    end
                end

                S_WR_W: begin
                    xfer_active <= 1'b1;
                    if (dma_wvalid && dma_wready) begin
                        word_cnt <= word_cnt + 16'd1;
                        if (wr_burst_beat + 9'd1 < wr_burst_words)
                            wr_burst_beat <= wr_burst_beat + 9'd1;
                    end
                end

                S_WR_B: begin
                    xfer_active <= 1'b1;
                end

                default: begin
                    xfer_active <= 1'b0;
                end
            endcase
        end
    end

    // ------------------------------------------------------------
    // Next-state logic
    // ------------------------------------------------------------
    always_comb begin
        next = state;
        case (state)
            S_IDLE: begin
                if (start && mode == 2'b01)
                    next = S_RD_START;
                else if (start && mode == 2'b10)
                    next = S_WR_AW;
            end

            S_RD_START: begin
                if (rd_desc_ready)
                    next = S_RD_XFER;
            end

            S_RD_XFER: begin
                if (rd_stream_tvalid && rd_stream_tlast)
                    next = S_RD_DONE;
            end

            S_RD_DONE: begin
                if (!start)
                    next = S_IDLE;
            end

            S_WR_AW: begin
                if (dma_awready)
                    next = S_WR_W;
            end

            S_WR_W: begin
                if (dma_wvalid && dma_wready && dma_wlast)
                    next = S_WR_B;
            end

            S_WR_B: begin
                if (dma_bvalid && dma_bready) begin
                    if (word_cnt >= total_words)
                        next = S_RD_DONE;
                    else
                        next = S_WR_AW;
                end
            end

            default: next = S_IDLE;
        endcase
    end

    // ------------------------------------------------------------
    // AXI write channel drives (combinational from state)
    // ------------------------------------------------------------
    assign dma_awaddr  = ext_addr + {word_cnt, 3'd0};
    assign dma_awlen   = wr_next_burst_words[7:0] - 8'd1;
    assign dma_awvalid = (state == S_WR_AW);

    assign dma_wdata  = wr_data;
    assign dma_wstrb  = 8'hFF;
    assign dma_wlast  = (wr_burst_beat + 9'd1 >= wr_burst_words);
    assign dma_wvalid = (state == S_WR_W);

    assign dma_bready = (state == S_WR_B);
    assign wr_ready   = (state == S_WR_W) && dma_wready;

    // ================================================================
    // Internal AXI RAM — 64 KB byte-addressable
    //
    // Has its own slave interface (ram_ar*, ram_r*, ram_aw*, ram_w*,
    // ram_b*) which is driven by:
    //   - dma_* (DMA engines) in STANDALONE mode
    //   - tied off in integrated mode (RAM used for debug only)
    // ================================================================

    // RAM slave signals
    logic [31:0] ram_araddr;
    logic [7:0]  ram_arlen;
    logic        ram_arvalid;
    logic        ram_arready;

    logic [63:0] ram_rdata;
    logic [1:0]  ram_rresp;
    logic        ram_rlast;
    logic        ram_rvalid;
    logic        ram_rready;

    logic [31:0] ram_awaddr;
    logic [7:0]  ram_awlen;
    logic        ram_awvalid;
    logic        ram_awready;

    logic [63:0] ram_wdata;
    logic [7:0]  ram_wstrb;
    logic        ram_wlast;
    logic        ram_wvalid;
    logic        ram_wready;

    logic [1:0]  ram_bresp;
    logic        ram_bvalid;
    logic        ram_bready;

    // ---- Memory array ----
    reg [7:0] axi_ram [0:65535];

    // ---- Debug write (clocked) ----
    always_ff @(posedge clk) begin
        if (sim_ram_en && sim_ram_we) begin
            axi_ram[sim_ram_addr  ] <= sim_ram_wdata[7:0];
            axi_ram[sim_ram_addr+1] <= sim_ram_wdata[15:8];
            axi_ram[sim_ram_addr+2] <= sim_ram_wdata[23:16];
            axi_ram[sim_ram_addr+3] <= sim_ram_wdata[31:24];
            axi_ram[sim_ram_addr+4] <= sim_ram_wdata[39:32];
            axi_ram[sim_ram_addr+5] <= sim_ram_wdata[47:40];
            axi_ram[sim_ram_addr+6] <= sim_ram_wdata[55:48];
            axi_ram[sim_ram_addr+7] <= sim_ram_wdata[63:56];
        end
    end

    // ---- Debug read (registered) ----
    reg [63:0] sim_ram_rdata_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sim_ram_rdata_reg <= 64'd0;
        end else if (sim_ram_en && !sim_ram_we) begin
            sim_ram_rdata_reg[7:0]   <= axi_ram[sim_ram_addr  ];
            sim_ram_rdata_reg[15:8]  <= axi_ram[sim_ram_addr+1];
            sim_ram_rdata_reg[23:16] <= axi_ram[sim_ram_addr+2];
            sim_ram_rdata_reg[31:24] <= axi_ram[sim_ram_addr+3];
            sim_ram_rdata_reg[39:32] <= axi_ram[sim_ram_addr+4];
            sim_ram_rdata_reg[47:40] <= axi_ram[sim_ram_addr+5];
            sim_ram_rdata_reg[55:48] <= axi_ram[sim_ram_addr+6];
            sim_ram_rdata_reg[63:56] <= axi_ram[sim_ram_addr+7];
        end
    end
    assign sim_ram_rdata = sim_ram_rdata_reg;

    // ---- AXI Read Slave FSM ----
    typedef enum logic [0:0] {
        RAM_R_IDLE  = 1'd0,
        RAM_R_BURST = 1'd1
    } ram_r_state_t;

    ram_r_state_t ram_r_state, ram_r_next;
    logic [7:0]   ram_r_len;
    logic [31:0]  ram_r_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ram_r_state <= RAM_R_IDLE;
            ram_r_len   <= 8'd0;
            ram_r_addr  <= 32'd0;
        end else begin
            ram_r_state <= ram_r_next;
            case (ram_r_state)
                RAM_R_IDLE: begin
                    if (ram_arvalid && ram_arready) begin
                        ram_r_addr <= ram_araddr;
                        ram_r_len  <= ram_arlen;
                    end
                end
                RAM_R_BURST: begin
                    if (ram_rvalid && ram_rready) begin
                        ram_r_addr <= ram_r_addr + 32'd8;
                        if (ram_r_len == 8'd0)
                            ram_r_state <= RAM_R_IDLE;
                        else
                            ram_r_len <= ram_r_len - 8'd1;
                    end
                end
            endcase
        end
    end

    always_comb begin
        ram_r_next = ram_r_state;
        case (ram_r_state)
            RAM_R_IDLE: begin
                if (ram_arvalid && ram_arready)
                    ram_r_next = RAM_R_BURST;
            end
            RAM_R_BURST: begin
                if (ram_rvalid && ram_rready && ram_r_len == 8'd0)
                    ram_r_next = RAM_R_IDLE;
            end
        endcase
    end

    assign ram_arready = (ram_r_state == RAM_R_IDLE);

    wire [63:0] ram_read_data;
    assign ram_read_data[7:0]   = axi_ram[ram_r_addr  ];
    assign ram_read_data[15:8]  = axi_ram[ram_r_addr+1];
    assign ram_read_data[23:16] = axi_ram[ram_r_addr+2];
    assign ram_read_data[31:24] = axi_ram[ram_r_addr+3];
    assign ram_read_data[39:32] = axi_ram[ram_r_addr+4];
    assign ram_read_data[47:40] = axi_ram[ram_r_addr+5];
    assign ram_read_data[55:48] = axi_ram[ram_r_addr+6];
    assign ram_read_data[63:56] = axi_ram[ram_r_addr+7];

    // Next-address data — used when advancing to the next burst beat.
    // ram_r_addr is updated via NBA, so in the rvalid block we use
    // this wire to read data at the UPDATED (next) address.
    wire [31:0]  ram_r_addr_next = ram_r_addr + 32'd8;
    wire [63:0] ram_read_data_next;
    assign ram_read_data_next[7:0]   = axi_ram[ram_r_addr_next  ];
    assign ram_read_data_next[15:8]  = axi_ram[ram_r_addr_next+1];
    assign ram_read_data_next[23:16] = axi_ram[ram_r_addr_next+2];
    assign ram_read_data_next[31:24] = axi_ram[ram_r_addr_next+3];
    assign ram_read_data_next[39:32] = axi_ram[ram_r_addr_next+4];
    assign ram_read_data_next[47:40] = axi_ram[ram_r_addr_next+5];
    assign ram_read_data_next[55:48] = axi_ram[ram_r_addr_next+6];
    assign ram_read_data_next[63:56] = axi_ram[ram_r_addr_next+7];

    reg        ram_rvalid_reg;
    reg [63:0] ram_rdata_reg;
    reg        ram_rlast_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ram_rvalid_reg <= 1'b0;
            ram_rdata_reg  <= 64'd0;
            ram_rlast_reg  <= 1'b0;
        end else begin
            if (ram_r_state == RAM_R_IDLE && ram_arvalid && ram_arready) begin
                ram_rvalid_reg <= 1'b1;
                // Read from the incoming AR address (ram_araddr), not ram_r_addr
                // which hasn't been updated yet (NBA race).
                ram_rdata_reg[7:0]   <= axi_ram[ram_araddr  ];
                ram_rdata_reg[15:8]  <= axi_ram[ram_araddr+1];
                ram_rdata_reg[23:16] <= axi_ram[ram_araddr+2];
                ram_rdata_reg[31:24] <= axi_ram[ram_araddr+3];
                ram_rdata_reg[39:32] <= axi_ram[ram_araddr+4];
                ram_rdata_reg[47:40] <= axi_ram[ram_araddr+5];
                ram_rdata_reg[55:48] <= axi_ram[ram_araddr+6];
                ram_rdata_reg[63:56] <= axi_ram[ram_araddr+7];
                ram_rlast_reg  <= (ram_arlen == 8'd0);
            end else if (ram_r_state == RAM_R_BURST && ram_rvalid && ram_rready && ram_r_len > 8'd0) begin
                ram_rvalid_reg <= 1'b1;
                ram_rdata_reg  <= ram_read_data_next;
                ram_rlast_reg  <= (ram_r_len == 8'd1);
            end else if (ram_rvalid && ram_rready) begin
                ram_rvalid_reg <= 1'b0;
            end
        end
    end

    assign ram_rdata  = ram_rdata_reg;
    assign ram_rvalid = ram_rvalid_reg;
    assign ram_rlast  = ram_rlast_reg;
    assign ram_rresp  = 2'b00;

    // ---- AXI Write Slave FSM ----
    typedef enum logic [1:0] {
        RAM_W_IDLE   = 2'd0,
        RAM_W_ACCEPT = 2'd1,
        RAM_W_RESP   = 2'd2
    } ram_w_state_t;

    ram_w_state_t ram_w_state, ram_w_next;
    logic [31:0]  ram_w_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ram_w_state <= RAM_W_IDLE;
            ram_w_addr  <= 32'd0;
        end else begin
            ram_w_state <= ram_w_next;
            case (ram_w_state)
                RAM_W_IDLE: begin
                    if (ram_awvalid && ram_awready) begin
                        ram_w_addr <= ram_awaddr;
                    end
                end
                RAM_W_ACCEPT: begin
                    if (ram_wvalid && ram_wready) begin
                        if (ram_wstrb[0]) axi_ram[ram_w_addr  ] <= ram_wdata[7:0];
                        if (ram_wstrb[1]) axi_ram[ram_w_addr+1] <= ram_wdata[15:8];
                        if (ram_wstrb[2]) axi_ram[ram_w_addr+2] <= ram_wdata[23:16];
                        if (ram_wstrb[3]) axi_ram[ram_w_addr+3] <= ram_wdata[31:24];
                        if (ram_wstrb[4]) axi_ram[ram_w_addr+4] <= ram_wdata[39:32];
                        if (ram_wstrb[5]) axi_ram[ram_w_addr+5] <= ram_wdata[47:40];
                        if (ram_wstrb[6]) axi_ram[ram_w_addr+6] <= ram_wdata[55:48];
                        if (ram_wstrb[7]) axi_ram[ram_w_addr+7] <= ram_wdata[63:56];
                        ram_w_addr <= ram_w_addr + 32'd8;
                        if (ram_wlast)
                            ram_w_state <= RAM_W_RESP;
                    end
                end
                RAM_W_RESP: begin
                    if (ram_bvalid && ram_bready)
                        ram_w_state <= RAM_W_IDLE;
                end
            endcase
        end
    end

    always_comb begin
        ram_w_next = ram_w_state;
        case (ram_w_state)
            RAM_W_IDLE: begin
                if (ram_awvalid && ram_awready)
                    ram_w_next = RAM_W_ACCEPT;
            end
            RAM_W_ACCEPT: begin
                if (ram_wvalid && ram_wready && ram_wlast)
                    ram_w_next = RAM_W_RESP;
            end
            RAM_W_RESP: begin
                if (ram_bvalid && ram_bready)
                    ram_w_next = RAM_W_IDLE;
            end
        endcase
    end

    assign ram_awready = (ram_w_state == RAM_W_IDLE);
    assign ram_wready  = (ram_w_state == RAM_W_ACCEPT);

    reg ram_bvalid_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ram_bvalid_reg <= 1'b0;
        end else begin
            if (ram_w_state == RAM_W_ACCEPT && ram_wvalid && ram_wready && ram_wlast)
                ram_bvalid_reg <= 1'b1;
            else if (ram_bvalid && ram_bready)
                ram_bvalid_reg <= 1'b0;
        end
    end
    assign ram_bvalid = ram_bvalid_reg;
    assign ram_bresp  = 2'b00;

    // ================================================================
    // Standalone / Integrated mux
    //
    // In standalone mode: DMA engines ↔ internal RAM
    // In integrated mode: DMA engines ↔ external ports,
    //   internal RAM is accessible only through debug port
    // ================================================================
    generate
        if (STANDALONE) begin : g_standalone
            // DMA read address → RAM
            assign ram_araddr  = dma_araddr;
            assign ram_arlen   = dma_arlen;
            assign ram_arvalid = dma_arvalid;
            assign dma_arready = ram_arready;

            // RAM read data → DMA
            assign dma_rdata  = ram_rdata;
            assign dma_rresp  = ram_rresp;
            assign dma_rlast  = ram_rlast;
            assign dma_rvalid = ram_rvalid;
            assign ram_rready = dma_rready;

            // DMA write address → RAM
            assign ram_awaddr  = dma_awaddr;
            assign ram_awlen   = dma_awlen;
            assign ram_awvalid = dma_awvalid;
            assign dma_awready = ram_awready;

            // DMA write data → RAM
            assign ram_wdata  = dma_wdata;
            assign ram_wstrb  = dma_wstrb;
            assign ram_wlast  = dma_wlast;
            assign ram_wvalid = dma_wvalid;
            assign dma_wready = ram_wready;

            // RAM write response → DMA
            assign dma_bresp  = ram_bresp;
            assign dma_bvalid = ram_bvalid;
            assign ram_bready = dma_bready;

            // External ports tied off
            assign m_axi_araddr  = 32'd0;
            assign m_axi_arlen   = 8'd0;
            assign m_axi_arvalid = 1'b0;
            assign m_axi_rready  = 1'b0;
            assign m_axi_awaddr  = 32'd0;
            assign m_axi_awlen   = 8'd0;
            assign m_axi_awvalid = 1'b0;
            assign m_axi_wdata   = 64'd0;
            assign m_axi_wstrb   = 8'd0;
            assign m_axi_wlast   = 1'b0;
            assign m_axi_wvalid  = 1'b0;
            assign m_axi_bready  = 1'b0;
        end else begin : g_integrated
            // DMA read address → external
            assign m_axi_araddr  = dma_araddr;
            assign m_axi_arlen   = dma_arlen;
            assign m_axi_arvalid = dma_arvalid;
            assign dma_arready   = m_axi_arready;

            // External read data → DMA
            assign dma_rdata  = m_axi_rdata;
            assign dma_rresp  = m_axi_rresp;
            assign dma_rlast  = m_axi_rlast;
            assign dma_rvalid = m_axi_rvalid;
            assign m_axi_rready = dma_rready;

            // DMA write address → external
            assign m_axi_awaddr  = dma_awaddr;
            assign m_axi_awlen   = dma_awlen;
            assign m_axi_awvalid = dma_awvalid;
            assign dma_awready   = m_axi_awready;

            // DMA write data → external
            assign m_axi_wdata  = dma_wdata;
            assign m_axi_wstrb  = dma_wstrb;
            assign m_axi_wlast  = dma_wlast;
            assign m_axi_wvalid = dma_wvalid;
            assign dma_wready   = m_axi_wready;

            // External write response → DMA
            assign dma_bresp  = m_axi_bresp;
            assign dma_bvalid = m_axi_bvalid;
            assign m_axi_bready = dma_bready;

            // Internal RAM idle
            assign ram_araddr  = 32'd0;
            assign ram_arlen   = 8'd0;
            assign ram_arvalid = 1'b0;
            assign ram_rready  = 1'b0;
            assign ram_awaddr  = 32'd0;
            assign ram_awlen   = 8'd0;
            assign ram_awvalid = 1'b0;
            assign ram_wdata   = 64'd0;
            assign ram_wstrb   = 8'd0;
            assign ram_wlast   = 1'b0;
            assign ram_wvalid  = 1'b0;
            assign ram_bready  = 1'b0;
        end
    endgenerate

endmodule
