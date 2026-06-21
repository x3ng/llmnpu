// IF/ID/Dispatch pipeline — synchronous reset, imem writes before reset check.
module if_id_top (
    input  logic        clk, rst_n,
    input  logic        mem_we,
    input  logic [7:0]  mem_addr,
    input  logic [31:0] mem_wdata,
    input  logic [31:0] instr_base_addr,
    input  logic        pc_we,
    input  logic [7:0]  pc_wdata,
    input  logic        halt,
    input  logic        gemm_busy, valu_busy, sfu_busy, dma_busy,
    output logic        refill_req,
    output logic [31:0] refill_ext_addr,
    input  logic        refill_valid,
    input  logic [31:0] refill_data,
    output logic        refill_busy,
    output logic        gemm_cmd_valid, valu_cmd_valid, sfu_cmd_valid, dma_cmd_valid,
    output logic        illegal_cmd_valid,
    output logic [31:0] gemm_cmd, valu_cmd, sfu_cmd, dma_cmd,
    output logic [7:0]  debug_pc,
    output logic [7:0]  current_pc,
    output logic [31:0] debug_instr,
    output logic        stall_if,
    output logic [31:0] debug_imem0
);

    localparam int IMEM_WORDS = 256;
    localparam int REFILL_WORDS = 32;
    localparam int REFILL_BLOCKS = IMEM_WORDS / REFILL_WORDS;

    reg [31:0] imem [0:IMEM_WORDS-1];
    wire [31:0] imem0 = imem[0];
    assign debug_imem0 = imem0;

    reg [7:0] pc;
    reg [7:0] id_pc;
    reg [31:0] id_instr;
    reg id_valid;
    reg halted;
    reg [REFILL_BLOCKS-1:0] block_valid;
    reg refill_active;
    reg [2:0] refill_block;
    reg [4:0] refill_idx;

    initial begin
        block_valid = '0;
    end

    wire [31:0] fetch_instr = imem[pc];
    wire [2:0] pc_block = pc[7:5];
    wire pc_block_valid = block_valid[pc_block];
    wire need_refill = !halted && !halt && !pc_block_valid && !refill_active;
    wire [2:0] refill_req_block = refill_active ? refill_block : pc_block;
    wire fetch_ready = pc_block_valid && !refill_active;

    wire [7:0]  opcode   = id_instr[31:24];
    wire target_gemm = (opcode == 8'h01) || (opcode == 8'h02);
    wire target_valu = (opcode == 8'h10) || (opcode == 8'h11);
    wire target_sfu  = (opcode >= 8'h20 && opcode <= 8'h23) ||
                       (opcode >= 8'h30 && opcode <= 8'h31);
    wire target_dma  = (opcode >= 8'h40 && opcode <= 8'h42);
    wire is_sync     = (opcode == 8'hF0);
    wire is_wfi      = (opcode == 8'hF1);
    wire is_nop      = (opcode == 8'hFF);
    wire target_known = target_gemm || target_valu || target_sfu || target_dma ||
                        is_sync || is_wfi || is_nop;

    wire busy_stall_w = id_valid && (
        (target_gemm && gemm_busy) ||
        (target_valu && valu_busy) ||
        (target_sfu  && sfu_busy)  ||
        (target_dma  && dma_busy)  ||
        (is_sync && (gemm_busy || valu_busy || sfu_busy || dma_busy))
    );
    wire dispatch_wfi = id_valid && is_wfi && !halted;
    wire stall_w = halted || halt || busy_stall_w || !fetch_ready;

    assign refill_req      = refill_active || need_refill;
    assign refill_busy     = refill_active;
    assign refill_ext_addr = instr_base_addr + {22'd0, refill_req_block, 7'd0};

    // Single synchronous always block. I-SRAM writes remain active during
    // datapath reset so software can preload instructions before START.
    always @(posedge clk) begin
        // ---- Synchronous reset / pipeline advance ----
        if (!rst_n) begin
            pc       <= 8'd0;
            id_valid <= 1'b0;
            id_instr <= 32'd0;
            id_pc    <= 8'd0;
            halted   <= 1'b0;
            refill_active <= 1'b0;
            refill_block  <= 3'd0;
            refill_idx    <= 5'd0;
            stall_if <= 1'b0;
            gemm_cmd_valid <= 1'b0; gemm_cmd <= 32'd0;
            valu_cmd_valid <= 1'b0; valu_cmd <= 32'd0;
            sfu_cmd_valid  <= 1'b0; sfu_cmd  <= 32'd0;
            dma_cmd_valid  <= 1'b0; dma_cmd  <= 32'd0;
            illegal_cmd_valid <= 1'b0;
        end else begin
            if (need_refill) begin
                refill_active <= 1'b1;
                refill_block  <= pc_block;
                refill_idx    <= 5'd0;
            end else if (refill_active && refill_valid) begin
                imem[{refill_block, refill_idx}] <= refill_data;
                if (refill_idx == 5'd31) begin
                    block_valid[refill_block] <= 1'b1;
                    refill_active <= 1'b0;
                    refill_idx    <= 5'd0;
                end else begin
                    refill_idx <= refill_idx + 5'd1;
                end
            end

            if (pc_we) begin
                pc       <= pc_wdata;
                id_pc    <= 8'd0;
                id_instr <= 32'd0;
                id_valid <= 1'b0;
                stall_if <= 1'b1;
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
            end

            // ---- Dispatch: pulse cmd valid ----
            gemm_cmd_valid <= 1'b0; valu_cmd_valid <= 1'b0;
            sfu_cmd_valid  <= 1'b0; dma_cmd_valid  <= 1'b0;
            illegal_cmd_valid <= 1'b0;

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
                end else if (!target_known) begin
                    illegal_cmd_valid <= 1'b1;
                end
            end
        end

        // ---- Instruction memory write: always, even during reset ----
        // Debug/CPU writes are treated as authoritative I-SRAM contents and
        // mark the containing 32-instruction block valid.
        if (mem_we) begin
            imem[mem_addr] <= mem_wdata;
            block_valid[mem_addr[7:5]] <= 1'b1;
        end
    end

    assign debug_pc    = id_pc;
    assign current_pc  = pc;
    assign debug_instr = id_instr;

endmodule
