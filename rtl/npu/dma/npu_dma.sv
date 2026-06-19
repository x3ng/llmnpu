// ============================================================
// npu_dma.sv — NPU DMA Controller
//
// Coordinates 1D/2D data transfers between external memory and
// NPU SRAM.  Drives the behavioral axi_dma_wrapper and manages
// ping-pong buffer ping-pong handshake.
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

module npu_dma (
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
    output logic        pp_bank,         // current ping-pong bank (toggles)
    output logic        pp_ready,        // buffer ready for consumer

    // --------------------------------------------------------
    // Simulation debug: direct SRAM access
    // --------------------------------------------------------
    input  logic        sim_sram_en,
    input  logic        sim_sram_we,
    input  logic [15:0] sim_sram_addr,
    input  logic [63:0] sim_sram_wdata,
    output logic [63:0] sim_sram_rdata,

    // Simulation debug: direct ExtMem access (routed to wrapper)
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
    // Wrapper instantiation
    // --------------------------------------------------------
    logic        wrapper_start;
    logic [1:0]  wrapper_mode;
    logic [31:0] wrapper_ext_addr;
    logic [15:0] wrapper_sram_addr;
    logic [15:0] wrapper_length;
    logic        wrapper_done;
    logic [63:0] wrapper_rd_data;
    logic [63:0] wrapper_wr_data;
    logic        wrapper_xfer_active;

    axi_dma_wrapper wrapper (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (wrapper_start),
        .mode        (wrapper_mode),
        .ext_addr    (wrapper_ext_addr),
        .sram_addr   (wrapper_sram_addr),
        .length      (wrapper_length),
        .done        (wrapper_done),
        .rd_data     (wrapper_rd_data),
        .wr_data     (wrapper_wr_data),
        .xfer_active (wrapper_xfer_active),
        .sim_en      (sim_ext_en),
        .sim_we      (sim_ext_we),
        .sim_addr    (sim_ext_addr),
        .sim_wdata   (sim_ext_wdata),
        .sim_rdata   (sim_ext_rdata)
    );

    // --------------------------------------------------------
    // Command registers (captured on start)
    // --------------------------------------------------------
    reg [1:0]  r_mode;          // decoded opcode → wrapper mode
    reg [31:0] r_ext_addr;
    reg [15:0] r_sram_addr;
    reg [15:0] r_length;
    reg [15:0] xfer_cnt;        // bytes transferred in current DMA op

    // Ping-pong bank (toggles after each transfer)
    reg pp_bank_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pp_bank_reg <= 1'b0;
        else if (done && busy)  pp_bank_reg <= ~pp_bank_reg;  // flip at completion
    end
    assign pp_bank  = pp_bank_reg;
    assign pp_ready = ~busy;    // buffer ready when DMA is idle

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
            wrapper_wr_data <= 64'd0;
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
                        xfer_cnt    <= 16'd0;

                        // Decode opcode → wrapper mode
                        case (opcode)
                            `OP_DMA_LD:  r_mode <= 2'b01;
                            `OP_DMA_ST:  r_mode <= 2'b10;
                            `OP_DMA_2D:  r_mode <= 2'b01;  // treat as load for now
                            default:     r_mode <= 2'b00;
                        endcase

                        // Arm wrapper
                        wrapper_start     <= 1'b1;
                        wrapper_mode      <= (opcode == `OP_DMA_ST) ? 2'b10 : 2'b01;
                        wrapper_ext_addr  <= ext_addr;
                        wrapper_sram_addr <= sram_addr;
                        wrapper_length    <= length;
                    end
                end

                S_XFER: begin
                    wrapper_start <= 1'b0;

                    // Data path
                    if (r_mode == 2'b01) begin
                        // LOAD: capture wrapper rd_data into SRAM
                        // xfer_cnt advances only when data is valid (1 cycle
                        // after wrapper starts its XFER, due to NBA pipeline)
                        if (wrapper_xfer_active) begin
                            sram[r_sram_addr + xfer_cnt  ] <= wrapper_rd_data[7:0];
                            sram[r_sram_addr + xfer_cnt+1] <= wrapper_rd_data[15:8];
                            sram[r_sram_addr + xfer_cnt+2] <= wrapper_rd_data[23:16];
                            sram[r_sram_addr + xfer_cnt+3] <= wrapper_rd_data[31:24];
                            sram[r_sram_addr + xfer_cnt+4] <= wrapper_rd_data[39:32];
                            sram[r_sram_addr + xfer_cnt+5] <= wrapper_rd_data[47:40];
                            sram[r_sram_addr + xfer_cnt+6] <= wrapper_rd_data[55:48];
                            sram[r_sram_addr + xfer_cnt+7] <= wrapper_rd_data[63:56];
                            xfer_cnt <= xfer_cnt + 8;
                        end
                    end else if (r_mode == 2'b10) begin
                        // STORE: drive SRAM data into wrapper wr_data
                        // xfer_cnt advances every cycle; wrapper consumes
                        // the presented data one cycle later
                        wrapper_wr_data[7:0]   <= sram[r_sram_addr + xfer_cnt  ];
                        wrapper_wr_data[15:8]  <= sram[r_sram_addr + xfer_cnt+1];
                        wrapper_wr_data[23:16] <= sram[r_sram_addr + xfer_cnt+2];
                        wrapper_wr_data[31:24] <= sram[r_sram_addr + xfer_cnt+3];
                        wrapper_wr_data[39:32] <= sram[r_sram_addr + xfer_cnt+4];
                        wrapper_wr_data[47:40] <= sram[r_sram_addr + xfer_cnt+5];
                        wrapper_wr_data[55:48] <= sram[r_sram_addr + xfer_cnt+6];
                        wrapper_wr_data[63:56] <= sram[r_sram_addr + xfer_cnt+7];
                        xfer_cnt <= xfer_cnt + 8;
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
