// Dispatch: top-level IF/ID/Dispatch pipeline.
// Instantiates ifetch + idecode internally.
// Routes decoded instructions to unit command interfaces.
// Implements 1-entry cmd queue per unit with backpressure (stall_if).
// NOTE: all combinational logic uses continuous assignments (assign),
// not always_comb, to avoid Icarus sensitivity-ordering issues.

`include "isa_defines.svh"
`include "npu_defines.svh"

module dispatch (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        mem_we,
    input  logic [7:0]  mem_addr,
    input  logic [31:0] mem_wdata,

    input  logic        gemm_busy,
    input  logic        valu_busy,
    input  logic        sfu_busy,
    input  logic        dma_busy,

    output logic        gemm_cmd_valid,
    output logic [31:0] gemm_cmd,
    output logic        valu_cmd_valid,
    output logic [31:0] valu_cmd,
    output logic        sfu_cmd_valid,
    output logic [31:0] sfu_cmd,
    output logic        dma_cmd_valid,
    output logic [31:0] dma_cmd,

    output logic [7:0]  debug_pc,
    output logic [31:0] debug_instr,
    output logic        stall_if
);

    // ---- internal wires from ifetch ----
    logic [7:0]  if_pc;
    logic [31:0] if_instr;
    logic        if_instr_valid;

    // ---- IF/ID pipeline register ----
    logic        id_valid;
    logic [31:0] id_instr;
    logic [7:0]  id_pc;

    // ---- decode wires, continuous assigns ----
    wire [7:0]  opcode;
    wire [7:0]  dst;
    wire [7:0]  src_a;
    wire [7:0]  src_b;
    wire [19:0] imm;
    wire        is_itype;

    assign opcode   = id_instr[31:24];
    assign dst      = id_instr[23:16];
    assign src_a    = id_instr[15:8];
    assign src_b    = id_instr[7:0];
    assign imm      = id_instr[19:0];
    assign is_itype = (id_instr[31:28] == 4'h4) || (id_instr[31:28] == 4'hF);

    // ---- target unit detection ----
    wire target_gemm;
    wire target_valu;
    wire target_sfu;
    wire target_dma;
    wire is_sync;
    wire is_wfi;
    wire is_nop;

    assign target_gemm = (opcode == 8'h01) || (opcode == 8'h02);
    assign target_valu = (opcode == 8'h10) || (opcode == 8'h11);
    assign target_sfu  = (opcode >= 8'h20 && opcode <= 8'h25) ||
                         (opcode >= 8'h30 && opcode <= 8'h31);
    assign target_dma  = (opcode >= 8'h40 && opcode <= 8'h42);
    assign is_sync     = (opcode == 8'hF0);
    assign is_wfi      = (opcode == 8'hF1);
    assign is_nop      = (opcode == 8'hFF);

    // ---- stall logic (combinational) ----
    wire stall_internal;
    assign stall_internal = id_valid && (
        (target_gemm && gemm_busy) ||
        (target_valu && valu_busy) ||
        (target_sfu  && sfu_busy)  ||
        (target_dma  && dma_busy)  ||
        (is_sync && (gemm_busy || valu_busy || sfu_busy || dma_busy))
    );

    // ---- single always_ff for all state ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            id_valid       <= 1'b0;
            id_instr       <= 32'd0;
            id_pc          <= 8'd0;
            stall_if       <= 1'b0;
            gemm_cmd_valid <= 1'b0; gemm_cmd <= 32'd0;
            valu_cmd_valid <= 1'b0; valu_cmd <= 32'd0;
            sfu_cmd_valid  <= 1'b0; sfu_cmd  <= 32'd0;
            dma_cmd_valid  <= 1'b0; dma_cmd  <= 32'd0;
        end else begin
            // ---- pipeline register (capture fetch output) ----
            if (!stall_internal) begin
                id_valid <= 1'b1;
                id_instr <= if_instr;
                id_pc    <= if_pc;
            end

            // ---- stall_if registered from stall_internal ----
            stall_if <= stall_internal;

            // ---- dispatch: pulse cmd_valid for dispatched instructions ----
            gemm_cmd_valid <= 1'b0;
            valu_cmd_valid <= 1'b0;
            sfu_cmd_valid  <= 1'b0;
            dma_cmd_valid  <= 1'b0;

            if (id_valid && !stall_internal) begin
                if (target_gemm) begin
                    gemm_cmd_valid <= 1'b1;
                    gemm_cmd       <= id_instr;
                end else if (target_valu) begin
                    valu_cmd_valid <= 1'b1;
                    valu_cmd       <= id_instr;
                end else if (target_sfu) begin
                    sfu_cmd_valid  <= 1'b1;
                    sfu_cmd        <= id_instr;
                end else if (target_dma) begin
                    dma_cmd_valid  <= 1'b1;
                    dma_cmd        <= id_instr;
                end
                // SYNC/WFI/NOP: advance without command
            end
        end
    end

    // ---- debug outputs ----
    assign debug_pc    = id_pc;
    assign debug_instr = id_instr;

    // ---- instantiate ifetch ----
    ifetch u_ifetch (
        .clk        (clk),
        .rst_n      (rst_n),
        .stall      (stall_internal),
        .pc         (if_pc),
        .instr      (if_instr),
        .instr_valid(if_instr_valid),
        .load_en    (mem_we),
        .load_addr  (mem_addr),
        .load_data  (mem_wdata)
    );

    // ---- instantiate idecode ----
    idecode u_idecode (
        .instruction(if_instr),
        .opcode     (),
        .dst        (),
        .src_a      (),
        .src_b      (),
        .imm        (),
        .is_itype   ()
    );

endmodule
