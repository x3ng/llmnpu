// IDecode: Instruction Decode — pure combinational.
// R-type: [31:24]=OP, [23:16]=DST, [15:8]=SRC_A, [7:0]=SRC_B
// I-type: [31:28]=OP[3:0], [27:20]=OPT, [19:0]=IMM
// is_itype set for opcodes starting with 0x4 (DMA) or 0xF (SYNC/WFI/NOP).

`include "isa_defines.svh"
`include "npu_defines.svh"

module idecode (
    input  logic [31:0] instruction,
    output logic [7:0]  opcode,
    output logic [7:0]  dst,
    output logic [7:0]  src_a,
    output logic [7:0]  src_b,
    output logic [19:0] imm,
    output logic        is_itype
);

    assign opcode   = instruction[31:24];
    assign dst      = instruction[23:16];
    assign src_a    = instruction[15:8];
    assign src_b    = instruction[7:0];
    assign imm      = instruction[19:0];
    assign is_itype = (instruction[31:28] == 4'h4) || (instruction[31:28] == 4'hF);

endmodule
