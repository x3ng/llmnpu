// IF/ID/Dispatch pipeline — synchronous reset, imem writes before reset check.
module if_id_top (
    input  logic        clk, rst_n,
    input  logic        mem_we,
    input  logic [7:0]  mem_addr,
    input  logic [31:0] mem_wdata,
    input  logic        gemm_busy, valu_busy, sfu_busy, dma_busy,
    output logic        gemm_cmd_valid, valu_cmd_valid, sfu_cmd_valid, dma_cmd_valid,
    output logic [31:0] gemm_cmd, valu_cmd, sfu_cmd, dma_cmd,
    output logic [7:0]  debug_pc,
    output logic [31:0] debug_instr,
    output logic        stall_if,
    output logic [31:0] debug_imem0
);

    reg [31:0] imem0, imem1, imem2, imem3;
    assign debug_imem0 = imem0;

    reg [7:0] pc;
    reg [7:0] id_pc;
    reg [31:0] id_instr;
    reg id_valid;
    reg halted;

    // Combinational fetch: pure mux via continuous assignment.
    wire [31:0] fetch_instr = (pc == 8'd0) ? imem0 :
                               (pc == 8'd1) ? imem1 :
                               (pc == 8'd2) ? imem2 :
                               (pc == 8'd3) ? imem3 :
                               32'd0;

    wire [7:0]  opcode   = id_instr[31:24];
    wire target_gemm = (opcode == 8'h01) || (opcode == 8'h02);
    wire target_valu = (opcode == 8'h10) || (opcode == 8'h11);
    wire target_sfu  = (opcode >= 8'h20 && opcode <= 8'h23) ||
                       (opcode >= 8'h30 && opcode <= 8'h31);
    wire target_dma  = (opcode >= 8'h40 && opcode <= 8'h42);
    wire is_sync     = (opcode == 8'hF0);
    wire is_wfi      = (opcode == 8'hF1);

    wire busy_stall_w = id_valid && (
        (target_gemm && gemm_busy) ||
        (target_valu && valu_busy) ||
        (target_sfu  && sfu_busy)  ||
        (target_dma  && dma_busy)  ||
        (is_sync && (gemm_busy || valu_busy || sfu_busy || dma_busy))
    );
    wire dispatch_wfi = id_valid && is_wfi && !halted;
    wire stall_w = halted || busy_stall_w;

    // Single synchronous always block: imem write first, then reset/pipeline.
    always @(posedge clk) begin
        // ---- Instruction memory write: always, even during reset ----
        if (mem_we) begin
            case (mem_addr)
                8'd0: imem0 <= mem_wdata;
                8'd1: imem1 <= mem_wdata;
                8'd2: imem2 <= mem_wdata;
                8'd3: imem3 <= mem_wdata;
                default: ;
            endcase
        end

        // ---- Synchronous reset / pipeline advance ----
        if (!rst_n) begin
            pc       <= 8'd0;
            id_valid <= 1'b0;
            id_instr <= 32'd0;
            id_pc    <= 8'd0;
            halted   <= 1'b0;
            stall_if <= 1'b0;
            gemm_cmd_valid <= 1'b0; gemm_cmd <= 32'd0;
            valu_cmd_valid <= 1'b0; valu_cmd <= 32'd0;
            sfu_cmd_valid  <= 1'b0; sfu_cmd  <= 32'd0;
            dma_cmd_valid  <= 1'b0; dma_cmd  <= 32'd0;
        end else begin
            // ---- stall_if register ----
            stall_if <= stall_w;

            // ---- Advance pipeline when not stalled ----
            if (!stall_w && !dispatch_wfi) begin
                id_instr <= fetch_instr;
                id_pc    <= pc;
                id_valid <= 1'b1;
                pc       <= pc + 8'd1;
            end

            // ---- Dispatch: pulse cmd valid ----
            gemm_cmd_valid <= 1'b0; valu_cmd_valid <= 1'b0;
            sfu_cmd_valid  <= 1'b0; dma_cmd_valid  <= 1'b0;

            if (dispatch_wfi) begin
                halted <= 1'b1;
            end else if (id_valid && !stall_w) begin
                if (target_gemm) begin
                    gemm_cmd_valid <= 1'b1; gemm_cmd <= id_instr;
                end else if (target_valu) begin
                    valu_cmd_valid <= 1'b1; valu_cmd <= id_instr;
                end else if (target_sfu) begin
                    sfu_cmd_valid  <= 1'b1; sfu_cmd  <= id_instr;
                end else if (target_dma) begin
                    dma_cmd_valid  <= 1'b1; dma_cmd  <= id_instr;
                end
            end
        end
    end

    assign debug_pc    = id_pc;
    assign debug_instr = id_instr;

endmodule
