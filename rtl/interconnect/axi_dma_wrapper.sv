// ============================================================
// axi_dma_wrapper.sv — Behavioral AXI4-lite DMA Wrapper
//
// For simulation: counter-based FSM with internal ext_mem model.
// No real AXI bus needed at this stage.
//
// Reference: alexforencich/verilog-axi (MIT licensed)
// ============================================================
// Ports:
//   clk, rst_n              — clock / async active-low reset
//   start                   — pulse to begin a transfer
//   mode[1:0]               — 01=LOAD (ext→sram), 10=STORE (sram→ext)
//   ext_addr[31:0]          — byte address in external memory
//   sram_addr[15:0]         — byte address in SRAM (pass-through)
//   length[15:0]            — transfer length in bytes (must be 8-aligned)
//   done                    — asserted when transfer completes
//   rd_data[63:0]           — data read from ext_mem (LOAD path)
//   wr_data[63:0]           — data to write into ext_mem (STORE path)
//   xfer_active             — high during the XFER FSM state
//   sim_{en,we,addr,wdata}  — debug access to ext_mem
//   sim_rdata               — debug read data from ext_mem
// ============================================================

`include "npu_defines.svh"

module axi_dma_wrapper (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [1:0]  mode,
    input  logic [31:0] ext_addr,
    input  logic [15:0] sram_addr,
    input  logic [15:0] length,
    output logic        done,

    // Data conduit to/from NPU DMA
    output logic [63:0] rd_data,
    input  logic [63:0] wr_data,
    output logic        xfer_active,

    // Simulation: debug access to external memory
    input  logic        sim_en,
    input  logic        sim_we,
    input  logic [31:0] sim_addr,
    input  logic [63:0] sim_wdata,
    output logic [63:0] sim_rdata,

    // External memory bypass — connects to shared ext_mem_model
    // so CPU firmware writes and DMA reads target the same array.
    input  logic [63:0] ext_mem_bypass_rdata,
    output logic [31:0] ext_mem_bypass_addr,
    output logic        ext_mem_bypass_re,
    output logic        ext_mem_bypass_we,
    output logic [63:0] ext_mem_bypass_wdata
);

    // --------------------------------------------------------
    // External memory model — 64 KB, byte-addressable
    // --------------------------------------------------------
    reg [7:0] ext_mem [0:65535];

    // --------------------------------------------------------
    // Simulation debug write (sequential — clocked)
    // --------------------------------------------------------
    always_ff @(posedge clk) begin
        if (sim_en && sim_we) begin
            ext_mem[sim_addr  ] <= sim_wdata[7:0];
            ext_mem[sim_addr+1] <= sim_wdata[15:8];
            ext_mem[sim_addr+2] <= sim_wdata[23:16];
            ext_mem[sim_addr+3] <= sim_wdata[31:24];
            ext_mem[sim_addr+4] <= sim_wdata[39:32];
            ext_mem[sim_addr+5] <= sim_wdata[47:40];
            ext_mem[sim_addr+6] <= sim_wdata[55:48];
            ext_mem[sim_addr+7] <= sim_wdata[63:56];
        end
    end

    // --------------------------------------------------------
    // Simulation debug read (registered — updated after NBA settles)
    reg [63:0] sim_rdata_reg;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sim_rdata_reg <= 64'd0;
        end else if (sim_en && !sim_we) begin
            sim_rdata_reg[7:0]   <= ext_mem[sim_addr  ];
            sim_rdata_reg[15:8]  <= ext_mem[sim_addr+1];
            sim_rdata_reg[23:16] <= ext_mem[sim_addr+2];
            sim_rdata_reg[31:24] <= ext_mem[sim_addr+3];
            sim_rdata_reg[39:32] <= ext_mem[sim_addr+4];
            sim_rdata_reg[47:40] <= ext_mem[sim_addr+5];
            sim_rdata_reg[55:48] <= ext_mem[sim_addr+6];
            sim_rdata_reg[63:56] <= ext_mem[sim_addr+7];
        end
    end
    assign sim_rdata = sim_rdata_reg;

    // --------------------------------------------------------
    // FSM — IDLE → XFER → DONE
    // --------------------------------------------------------
    typedef enum logic [1:0] {
        S_IDLE = 2'b00,
        S_XFER = 2'b01,
        S_DONE = 2'b10
    } state_t;

    state_t state, next;

    reg [15:0] xfer_cnt;        // total bytes transferred so far
    reg [31:0] cur_ext_addr;    // current ext_mem address (increments by 8)

    // --------------------------------------------------------
    // Combinational LOAD read from internal ext_mem
    //
    // ext_mem is indexed by flat byte address (0..65535).
    // When the hierarchical bypass is driven (top_soc), the
    // bypass rdata takes precedence via the registered assignment
    // below.  When bypass is undriven (standalone npu_dma
    // tests), internal ext_mem provides the data.
    // --------------------------------------------------------
    wire [63:0] internal_rd_data;
    assign internal_rd_data[7:0]   = ext_mem[cur_ext_addr[15:0]  ];
    assign internal_rd_data[15:8]  = ext_mem[cur_ext_addr[15:0]+1];
    assign internal_rd_data[23:16] = ext_mem[cur_ext_addr[15:0]+2];
    assign internal_rd_data[31:24] = ext_mem[cur_ext_addr[15:0]+3];
    assign internal_rd_data[39:32] = ext_mem[cur_ext_addr[15:0]+4];
    assign internal_rd_data[47:40] = ext_mem[cur_ext_addr[15:0]+5];
    assign internal_rd_data[55:48] = ext_mem[cur_ext_addr[15:0]+6];
    assign internal_rd_data[63:56] = ext_mem[cur_ext_addr[15:0]+7];

    // --------------------------------------------------------
    // Sequential logic
    // --------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            xfer_active <= 1'b0;
            rd_data     <= 64'd0;
            xfer_cnt    <= 16'd0;
            cur_ext_addr <= 32'd0;
        end else begin
            state <= next;

            case (state)
                S_IDLE: begin
                    done        <= 1'b0;
                    xfer_active <= 1'b0;
                    xfer_cnt    <= 16'd0;
                    if (start) begin
                        cur_ext_addr <= ext_addr;
                    end
                end

                S_XFER: begin
                    xfer_active <= 1'b1;

                    if (mode == 2'b01) begin
                        // LOAD: use bypass rdata if driven, else internal ext_mem
                        if (!$isunknown(ext_mem_bypass_rdata))
                            rd_data <= ext_mem_bypass_rdata;
                        else
                            rd_data <= internal_rd_data;
                    end else if (mode == 2'b10) begin
                        // STORE: write to internal ext_mem so standalone tests
                        // can read data back via sim debug port.  The
                        // combinational bypass_we / bypass_wdata path also
                        // writes to u_dram.mem when connected through top_soc.
                        ext_mem[cur_ext_addr[15:0]  ] <= wr_data[7:0];
                        ext_mem[cur_ext_addr[15:0]+1] <= wr_data[15:8];
                        ext_mem[cur_ext_addr[15:0]+2] <= wr_data[23:16];
                        ext_mem[cur_ext_addr[15:0]+3] <= wr_data[31:24];
                        ext_mem[cur_ext_addr[15:0]+4] <= wr_data[39:32];
                        ext_mem[cur_ext_addr[15:0]+5] <= wr_data[47:40];
                        ext_mem[cur_ext_addr[15:0]+6] <= wr_data[55:48];
                        ext_mem[cur_ext_addr[15:0]+7] <= wr_data[63:56];
                    end

                    cur_ext_addr <= cur_ext_addr + 8;
                    xfer_cnt     <= xfer_cnt + 8;
                end

                S_DONE: begin
                    done        <= 1'b1;
                    xfer_active <= 1'b0;
                end
            endcase
        end
    end

    // --------------------------------------------------------
    // Combinational bypass outputs — driven from registered state
    // so the outside sees stable values at the correct cycle.
    // --------------------------------------------------------
    assign ext_mem_bypass_addr  = cur_ext_addr;
    assign ext_mem_bypass_re    = (state == S_XFER) && (mode == 2'b01);
    assign ext_mem_bypass_we    = (state == S_XFER) && (mode == 2'b10);
    assign ext_mem_bypass_wdata = wr_data;

    // --------------------------------------------------------
    // Next-state logic
    // --------------------------------------------------------
    always_comb begin
        next = state;
        case (state)
            S_IDLE: if (start)              next = S_XFER;
            S_XFER: if (xfer_cnt >= length) next = S_DONE;
            S_DONE: if (~start)             next = S_IDLE;
        endcase
    end

endmodule
