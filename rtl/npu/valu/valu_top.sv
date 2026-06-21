// valu_top: 64-lane SIMD Vector ALU with regfile and 4-state FSM
// FSM: IDLE -> READ -> EXECUTE -> WRITEBACK
// Test harness gets direct regfile access when idle.

`include "npu_defines.svh"

module valu_top (
    input  logic            clk,
    input  logic            rst_n,

    // Command interface
    input  logic            cmd_valid,
    input  logic [7:0]      opcode,
    input  logic [7:0]      opt,
    input  logic [4:0]      rs1,
    input  logic [4:0]      rs2,
    input  logic [4:0]      rd,
    output logic            busy,
    output logic            done,

    // Test harness direct regfile access (valid when not busy)
    input  logic            test_wen,
    input  logic [4:0]      test_waddr,
    input  logic [63:0][7:0] test_wdata,
    input  logic [4:0]      test_raddr,
    output logic [63:0][7:0] test_rdata
);

    // ------------------------------------------------------------------
    // FSM state encoding (4 states)
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE = 2'd0,
        READ = 2'd1,
        EXEC = 2'd2,
        WB   = 2'd3
    } state_t;

    state_t state, next;

    // Captured command operands
    logic [4:0]       rs1_q, rs2_q, rd_q;
    logic [7:0]       opt_q;

    // Latched regfile read data (captured at end of READ cycle)
    logic [63:0][7:0] rs1_data_q, rs2_data_q;

    // Lane computation results
    logic [63:0][7:0] lane_results;

    // Regfile internal connections (muxed between FSM and test)
    logic [4:0]       rf_rs1_addr, rf_rs2_addr, rf_rd_addr;
    logic [63:0][7:0] rf_rs1_data, rf_rs2_data, rf_rd_data;
    logic             rf_wen;

    // ------------------------------------------------------------------
    // State register
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else        state <= next;
    end

    // ------------------------------------------------------------------
    // Next-state logic
    // ------------------------------------------------------------------
    always_comb begin
        next = state;
        case (state)
            IDLE: if (cmd_valid) next = READ;
            READ: next = EXEC;
            EXEC: next = WB;
            WB:   next = IDLE;
        endcase
    end

    // ------------------------------------------------------------------
    // Capture command operands on IDLE -> READ transition
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs1_q <= '0;
            rs2_q <= '0;
            rd_q  <= '0;
            opt_q <= '0;
        end else if (state == IDLE && cmd_valid) begin
            rs1_q <= rs1;
            rs2_q <= rs2;
            rd_q  <= rd;
            opt_q <= opt;
        end
    end

    // ------------------------------------------------------------------
    // Latch regfile read data in READ state
    // ------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs1_data_q <= '0;
            rs2_data_q <= '0;
        end else if (state == READ) begin
            rs1_data_q <= rf_rs1_data;
            rs2_data_q <= rf_rs2_data;
        end
    end

    // ------------------------------------------------------------------
    // Regfile mux: FSM control when busy, test control when idle
    // ------------------------------------------------------------------
    assign rf_rs1_addr = (state != IDLE) ? rs1_q   : test_raddr;
    assign rf_rs2_addr = (state != IDLE) ? rs2_q   : '0;
    assign rf_wen      = (state != IDLE) ? (state == WB) : test_wen;
    assign rf_rd_addr  = (state != IDLE) ? rd_q    : test_waddr;
    assign rf_rd_data  = (state != IDLE) ? lane_results : test_wdata;

    // ------------------------------------------------------------------
    // Register file instance
    // ------------------------------------------------------------------
    vregfile u_regfile (
        .clk      (clk),
        .rs1_addr (rf_rs1_addr),
        .rs1_data (rf_rs1_data),
        .rs2_addr (rf_rs2_addr),
        .rs2_data (rf_rs2_data),
        .rd_addr  (rf_rd_addr),
        .rd_data  (rf_rd_data),
        .rd_wen   (rf_wen)
    );

    // ------------------------------------------------------------------
    // 64-lane ALU instantiation via generate
    // ------------------------------------------------------------------
    generate
        for (genvar i = 0; i < `VALU_LANES; i++) begin : lanes
            wire [7:0] lane_b = opt_q[7] ? rs2_data_q[0] : rs2_data_q[i];
            valu_lane u_lane (
                .op    (opt_q),
                .a     (rs1_data_q[i]),
                .b     (lane_b),
                .result(lane_results[i])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------------
    assign busy       = (state != IDLE);
    assign done       = (state == WB);
    assign test_rdata = rf_rs1_data;

endmodule
