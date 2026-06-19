// ============================================================
// npu_dma.sv — NPU DMA Controller
//
// Coordinates 1D/2D data transfers between external memory and
// NPU SRAM.  Wraps axi_dma_wrapper (proper AXI DMA engine).
//
// FSM: IDLE → XFER → DONE
//
// Feature set (initial version):
//   - 1D linear LOAD (OP_DMA_LD)  ext→sram
//   - 1D linear STORE (OP_DMA_ST) sram→ext
//   - 2D placeholder (OP_DMA_2D)  treated as 1D LOAD
//   - Ping-pong buffer coordination via bank flip
// ============================================================

`include "npu_defines.svh"
`include "isa_defines.svh"

module npu_dma #(
    parameter bit STANDALONE = 1
) (
    input  logic        clk,
    input  logic        rst_n,

    // Command interface (from dispatcher)
    input  logic        start,
    input  logic [7:0]  opcode,
    input  logic [31:0] ext_addr,
    input  logic [15:0] sram_addr,
    input  logic [15:0] length,

    // Status
    output logic        busy,
    output logic        done,

    // Ping-pong buffer coordination
    output logic        pp_bank,
    output logic        pp_ready,

    // ---- AXI4 Master Interface (passthrough from wrapper) ----
    // Read address
    output logic [31:0] m_axi_araddr,
    output logic [7:0]  m_axi_arlen,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    // Read data
    input  logic [63:0] m_axi_rdata,
    input  logic [1:0]  m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready,
    // Write address
    output logic [31:0] m_axi_awaddr,
    output logic [7:0]  m_axi_awlen,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    // Write data
    output logic [63:0] m_axi_wdata,
    output logic [7:0]  m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    // Write response
    input  logic [1:0]  m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,

    // --------------------------------------------------------
    // Simulation debug: direct SRAM access
    // --------------------------------------------------------
    input  logic        sim_sram_en,
    input  logic        sim_sram_we,
    input  logic [15:0] sim_sram_addr,
    input  logic [63:0] sim_sram_wdata,
    output logic [63:0] sim_sram_rdata,

    // Simulation debug: direct ExtMem access (routed to wrapper AXI RAM)
    input  logic        sim_ext_en,
    input  logic        sim_ext_we,
    input  logic [31:0] sim_ext_addr,
    input  logic [63:0] sim_ext_wdata,
    output logic [63:0] sim_ext_rdata
);

    // --------------------------------------------------------
    // SRAM memory model — 64 KB, byte-addressable
    // --------------------------------------------------------
    reg [7:0] sram [0:65535];

    // SRAM simulation debug write (sequential — clocked)
    always_ff @(posedge clk) begin
        if (sim_sram_en && sim_sram_we) begin
            sram[sim_sram_addr  ] <= sim_sram_wdata[7:0];
            sram[sim_sram_addr+1] <= sim_sram_wdata[15:8];
            sram[sim_sram_addr+2] <= sim_sram_wdata[23:16];
            sram[sim_sram_addr+3] <= sim_sram_wdata[31:24];
            sram[sim_sram_addr+4] <= sim_sram_wdata[39:32];
            sram[sim_sram_addr+5] <= sim_sram_wdata[47:40];
            sram[sim_sram_addr+6] <= sim_sram_wdata[55:48];
            sram[sim_sram_addr+7] <= sim_sram_wdata[63:56];
        end
    end

    // SRAM simulation debug read (registered — updated after NBA settles)
    reg [63:0] sram_rdata_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sram_rdata_reg <= 64'd0;
        end else if (sim_sram_en && !sim_sram_we) begin
            sram_rdata_reg[7:0]   <= sram[sim_sram_addr  ];
            sram_rdata_reg[15:8]  <= sram[sim_sram_addr+1];
            sram_rdata_reg[23:16] <= sram[sim_sram_addr+2];
            sram_rdata_reg[31:24] <= sram[sim_sram_addr+3];
            sram_rdata_reg[39:32] <= sram[sim_sram_addr+4];
            sram_rdata_reg[47:40] <= sram[sim_sram_addr+5];
            sram_rdata_reg[55:48] <= sram[sim_sram_addr+6];
            sram_rdata_reg[63:56] <= sram[sim_sram_addr+7];
        end
    end
    assign sim_sram_rdata = sram_rdata_reg;

    // --------------------------------------------------------
    // Wrapper signals
    // --------------------------------------------------------
    logic        wrapper_start;
    logic [1:0]  wrapper_mode;
    logic [31:0] wrapper_ext_addr;
    logic [15:0] wrapper_length;
    logic        wrapper_done;
    logic        wrapper_xfer_active;

    logic [63:0] wrapper_rd_data;
    logic        wrapper_rd_valid;

    logic [63:0] wrapper_wr_data;
    logic        wrapper_wr_ready;

    // --------------------------------------------------------
    // Command registers (captured on start)
    // --------------------------------------------------------
    reg [1:0]  r_mode;
    reg [31:0] r_ext_addr;
    reg [15:0] r_sram_addr;
    reg [15:0] r_length;
    reg [15:0] xfer_cnt;
    reg [15:0] total_words;     // length / 8

    // wr_data driven combinationally from sram — avoids NBA race
    // with wrapper's capture cycle.
    assign wrapper_wr_data[7:0]   = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt  ] : 8'd0;
    assign wrapper_wr_data[15:8]  = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+1] : 8'd0;
    assign wrapper_wr_data[23:16] = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+2] : 8'd0;
    assign wrapper_wr_data[31:24] = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+3] : 8'd0;
    assign wrapper_wr_data[39:32] = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+4] : 8'd0;
    assign wrapper_wr_data[47:40] = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+5] : 8'd0;
    assign wrapper_wr_data[55:48] = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+6] : 8'd0;
    assign wrapper_wr_data[63:56] = (r_mode == 2'b10) ? sram[r_sram_addr + xfer_cnt+7] : 8'd0;

    // --------------------------------------------------------
    // Wrapper instantiation — proper AXI DMA
    // --------------------------------------------------------
    axi_dma_wrapper #(
        .STANDALONE (STANDALONE)
    ) wrapper (
        .clk,
        .rst_n,
        .start         (wrapper_start),
        .mode          (wrapper_mode),
        .ext_addr      (wrapper_ext_addr),
        .length        (wrapper_length),
        .done          (wrapper_done),
        .xfer_active   (wrapper_xfer_active),
        .rd_data       (wrapper_rd_data),
        .rd_valid      (wrapper_rd_valid),
        .wr_data       (wrapper_wr_data),
        .wr_ready      (wrapper_wr_ready),

        // AXI passthrough
        .m_axi_araddr  (m_axi_araddr),
        .m_axi_arlen   (m_axi_arlen),
        .m_axi_arvalid (m_axi_arvalid),
        .m_axi_arready (m_axi_arready),
        .m_axi_rdata   (m_axi_rdata),
        .m_axi_rresp   (m_axi_rresp),
        .m_axi_rlast   (m_axi_rlast),
        .m_axi_rvalid  (m_axi_rvalid),
        .m_axi_rready  (m_axi_rready),
        .m_axi_awaddr  (m_axi_awaddr),
        .m_axi_awlen   (m_axi_awlen),
        .m_axi_awvalid (m_axi_awvalid),
        .m_axi_awready (m_axi_awready),
        .m_axi_wdata   (m_axi_wdata),
        .m_axi_wstrb   (m_axi_wstrb),
        .m_axi_wlast   (m_axi_wlast),
        .m_axi_wvalid  (m_axi_wvalid),
        .m_axi_wready  (m_axi_wready),
        .m_axi_bresp   (m_axi_bresp),
        .m_axi_bvalid  (m_axi_bvalid),
        .m_axi_bready  (m_axi_bready),

        // Debug: sim_ext_* routed to wrapper's sim_ram_*
        .sim_ram_en    (sim_ext_en),
        .sim_ram_we    (sim_ext_we),
        .sim_ram_addr  (sim_ext_addr),
        .sim_ram_wdata (sim_ext_wdata),
        .sim_ram_rdata (sim_ext_rdata)
    );

    // Ping-pong bank
    reg pp_bank_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pp_bank_reg <= 1'b0;
        else if (done && busy)  pp_bank_reg <= ~pp_bank_reg;
    end
    assign pp_bank  = pp_bank_reg;
    assign pp_ready = ~busy;

    // --------------------------------------------------------
    // DMA FSM — IDLE → XFER → DONE
    // --------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE = 2'b00,
        S_XFER = 2'b01,
        S_DONE = 2'b10
    } state_t;

    state_t state, next;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            wrapper_start <= 1'b0;
            xfer_cnt      <= 16'd0;
            total_words   <= 16'd0;
        end else begin
            state <= next;

            case (state)
                S_IDLE: begin
                    done          <= 1'b0;
                    wrapper_start <= 1'b0;
                    if (start) begin
                        r_ext_addr  <= ext_addr;
                        r_sram_addr <= sram_addr;
                        r_length    <= length;
                        total_words <= length[15:3];  // bytes / 8
                        xfer_cnt    <= 16'd0;

                        // Decode opcode → wrapper mode
                        case (opcode)
                            `OP_DMA_LD:  r_mode <= 2'b01;
                            `OP_DMA_ST:  r_mode <= 2'b10;
                            `OP_DMA_2D:  r_mode <= 2'b01;
                            default:     r_mode <= 2'b00;
                        endcase

                        // Arm wrapper
                        wrapper_start     <= 1'b1;
                        wrapper_mode      <= (opcode == `OP_DMA_ST) ? 2'b10 : 2'b01;
                        wrapper_ext_addr  <= ext_addr;
                        wrapper_length    <= length;
                    end
                end

                S_XFER: begin
                    wrapper_start <= 1'b0;

                    if (r_mode == 2'b01) begin
                        // LOAD: capture wrapper rd_data into SRAM
                        // when rd_valid is asserted
                        if (wrapper_rd_valid) begin
                            sram[r_sram_addr + xfer_cnt  ] <= wrapper_rd_data[7:0];
                            sram[r_sram_addr + xfer_cnt+1] <= wrapper_rd_data[15:8];
                            sram[r_sram_addr + xfer_cnt+2] <= wrapper_rd_data[23:16];
                            sram[r_sram_addr + xfer_cnt+3] <= wrapper_rd_data[31:24];
                            sram[r_sram_addr + xfer_cnt+4] <= wrapper_rd_data[39:32];
                            sram[r_sram_addr + xfer_cnt+5] <= wrapper_rd_data[47:40];
                            sram[r_sram_addr + xfer_cnt+6] <= wrapper_rd_data[55:48];
                            sram[r_sram_addr + xfer_cnt+7] <= wrapper_rd_data[63:56];
                            xfer_cnt <= xfer_cnt + 16'd8;
                        end
                    end else if (r_mode == 2'b10) begin
                        // STORE: wr_data is driven combinationally (wire above).
                        // Advance xfer_cnt when wrapper accepts current word.
                        if (wrapper_wr_ready) begin
                            xfer_cnt <= xfer_cnt + 16'd8;
                        end
                    end
                end

                S_DONE: begin
                    done          <= 1'b1;
                    wrapper_start <= 1'b0;
                end
            endcase
        end
    end

    // --------------------------------------------------------
    // Next-state logic
    // --------------------------------------------------------
    always_comb begin
        next = state;
        case (state)
            S_IDLE: if (start)               next = S_XFER;
            S_XFER: if (wrapper_done)        next = S_DONE;
            S_DONE: if (~start)              next = S_IDLE;
        endcase
    end

    assign busy = (state != S_IDLE);

endmodule
