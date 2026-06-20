// crossbar.sv — 3-master x 4-slave crossbar with fixed priority arbitration
//
// Masters (priority: 0 > 1 > 2):
//   M0 — DMA  (highest)
//   M1 — GEMM
//   M2 — VALU/SFU (lowest)
//
// Slaves:
//   S0 — A-SRAM   (addr[15:12] == 4'h0)
//   S1 — W-SRAM   (addr[15:12] == 4'h1)
//   S2 — O-SRAM   (addr[15:12] == 4'h2)
//   S3 — D-SRAM   (addr[15:12] == 4'h3)
//
// Address is byte-granular; word address = addr[11:2].
// Each slave bank has a 4 KB window (12 address bits).
`timescale 1ns/1ps

`include "npu_defines.svh"

module crossbar (
    input  logic        clk,
    input  logic        rst_n,

    // ---- Master 0 (DMA) - highest priority ----
    input  logic        m0_req,
    input  logic [15:0] m0_addr,
    input  logic [31:0] m0_wdata,
    input  logic        m0_wen,
    output logic [31:0] m0_rdata,
    output logic        m0_grant,

    // ---- Master 1 (GEMM) ----
    input  logic        m1_req,
    input  logic [15:0] m1_addr,
    input  logic [31:0] m1_wdata,
    input  logic        m1_wen,
    output logic [31:0] m1_rdata,
    output logic        m1_grant,

    // ---- Master 2 (VALU/SFU) - lowest priority ----
    input  logic        m2_req,
    input  logic [15:0] m2_addr,
    input  logic [31:0] m2_wdata,
    input  logic        m2_wen,
    output logic [31:0] m2_rdata,
    output logic        m2_grant
);

    // ====== Slave select decode ======
    wire [3:0] m0_slv_id = m0_addr[15:12];
    wire [3:0] m1_slv_id = m1_addr[15:12];
    wire [3:0] m2_slv_id = m2_addr[15:12];

    // ====== Arbitration: per-slave priority encoder ======
    // Priority: M0 > M1 > M2.

    // -- Slave 0 (A-SRAM) --
    wire       slv0_gv;
    wire [1:0] slv0_gm;
    assign slv0_gv = m0_req && (m0_slv_id == 0) ? 1'b1 :
                     m1_req && (m1_slv_id == 0) ? 1'b1 :
                     m2_req && (m2_slv_id == 0) ? 1'b1 : 1'b0;
    assign slv0_gm = m0_req && (m0_slv_id == 0) ? 2'd0 :
                     m1_req && (m1_slv_id == 0) ? 2'd1 :
                     m2_req && (m2_slv_id == 0) ? 2'd2 : 2'd0;

    // -- Slave 1 (W-SRAM) --
    wire       slv1_gv;
    wire [1:0] slv1_gm;
    assign slv1_gv = m0_req && (m0_slv_id == 1) ? 1'b1 :
                     m1_req && (m1_slv_id == 1) ? 1'b1 :
                     m2_req && (m2_slv_id == 1) ? 1'b1 : 1'b0;
    assign slv1_gm = m0_req && (m0_slv_id == 1) ? 2'd0 :
                     m1_req && (m1_slv_id == 1) ? 2'd1 :
                     m2_req && (m2_slv_id == 1) ? 2'd2 : 2'd0;

    // -- Slave 2 (O-SRAM) --
    wire       slv2_gv;
    wire [1:0] slv2_gm;
    assign slv2_gv = m0_req && (m0_slv_id == 2) ? 1'b1 :
                     m1_req && (m1_slv_id == 2) ? 1'b1 :
                     m2_req && (m2_slv_id == 2) ? 1'b1 : 1'b0;
    assign slv2_gm = m0_req && (m0_slv_id == 2) ? 2'd0 :
                     m1_req && (m1_slv_id == 2) ? 2'd1 :
                     m2_req && (m2_slv_id == 2) ? 2'd2 : 2'd0;

    // -- Slave 3 (D-SRAM) --
    wire       slv3_gv;
    wire [1:0] slv3_gm;
    assign slv3_gv = m0_req && (m0_slv_id == 3) ? 1'b1 :
                     m1_req && (m1_slv_id == 3) ? 1'b1 :
                     m2_req && (m2_slv_id == 3) ? 1'b1 : 1'b0;
    assign slv3_gm = m0_req && (m0_slv_id == 3) ? 2'd0 :
                     m1_req && (m1_slv_id == 3) ? 2'd1 :
                     m2_req && (m2_slv_id == 3) ? 2'd2 : 2'd0;

    // ====== Per-master grant signals ======
    // Combinational — a master is granted when the slave it targets
    // grants access to that master.
    assign m0_grant = m0_req &&
        (m0_slv_id == 0 ? slv0_gv :
         m0_slv_id == 1 ? slv1_gv :
         m0_slv_id == 2 ? slv2_gv :
         m0_slv_id == 3 ? slv3_gv : 1'b0) &&
        (m0_slv_id == 0 ? slv0_gm == 0 :
         m0_slv_id == 1 ? slv1_gm == 0 :
         m0_slv_id == 2 ? slv2_gm == 0 :
         m0_slv_id == 3 ? slv3_gm == 0 : 1'b0);

    assign m1_grant = m1_req &&
        (m1_slv_id == 0 ? slv0_gv :
         m1_slv_id == 1 ? slv1_gv :
         m1_slv_id == 2 ? slv2_gv :
         m1_slv_id == 3 ? slv3_gv : 1'b0) &&
        (m1_slv_id == 0 ? slv0_gm == 1 :
         m1_slv_id == 1 ? slv1_gm == 1 :
         m1_slv_id == 2 ? slv2_gm == 1 :
         m1_slv_id == 3 ? slv3_gm == 1 : 1'b0);

    assign m2_grant = m2_req &&
        (m2_slv_id == 0 ? slv0_gv :
         m2_slv_id == 1 ? slv1_gv :
         m2_slv_id == 2 ? slv2_gv :
         m2_slv_id == 3 ? slv3_gv : 1'b0) &&
        (m2_slv_id == 0 ? slv0_gm == 2 :
         m2_slv_id == 1 ? slv1_gm == 2 :
         m2_slv_id == 2 ? slv2_gm == 2 :
         m2_slv_id == 3 ? slv3_gm == 2 : 1'b0);

    // ====== Route granted master signals to each slave ======
    wire [15:0] slv0_addr, slv1_addr, slv2_addr, slv3_addr;
    wire [31:0] slv0_wdata, slv1_wdata, slv2_wdata, slv3_wdata;
    wire        slv0_wen,   slv1_wen,   slv2_wen,   slv3_wen;
    wire        slv0_ren,   slv1_ren,   slv2_ren,   slv3_ren;
    wire [31:0] slv0_rdata, slv1_rdata, slv2_rdata, slv3_rdata;

    // ── Slave 0 mux ─────────────────────────────────────────
    assign slv0_addr  = slv0_gv && slv0_gm == 0 ? m0_addr  :
                        slv0_gv && slv0_gm == 1 ? m1_addr  :
                        slv0_gv && slv0_gm == 2 ? m2_addr  : 16'h0;
    assign slv0_wdata = slv0_gv && slv0_gm == 0 ? m0_wdata :
                        slv0_gv && slv0_gm == 1 ? m1_wdata :
                        slv0_gv && slv0_gm == 2 ? m2_wdata : 32'h0;
    assign slv0_wen   = slv0_gv && slv0_gm == 0 ? m0_wen   :
                        slv0_gv && slv0_gm == 1 ? m1_wen   :
                        slv0_gv && slv0_gm == 2 ? m2_wen   : 1'b0;
    assign slv0_ren   = slv0_gv && slv0_gm == 0 ? ~m0_wen  :
                        slv0_gv && slv0_gm == 1 ? ~m1_wen  :
                        slv0_gv && slv0_gm == 2 ? ~m2_wen  : 1'b0;

    // ── Slave 1 mux ─────────────────────────────────────────
    assign slv1_addr  = slv1_gv && slv1_gm == 0 ? m0_addr  :
                        slv1_gv && slv1_gm == 1 ? m1_addr  :
                        slv1_gv && slv1_gm == 2 ? m2_addr  : 16'h0;
    assign slv1_wdata = slv1_gv && slv1_gm == 0 ? m0_wdata :
                        slv1_gv && slv1_gm == 1 ? m1_wdata :
                        slv1_gv && slv1_gm == 2 ? m2_wdata : 32'h0;
    assign slv1_wen   = slv1_gv && slv1_gm == 0 ? m0_wen   :
                        slv1_gv && slv1_gm == 1 ? m1_wen   :
                        slv1_gv && slv1_gm == 2 ? m2_wen   : 1'b0;
    assign slv1_ren   = slv1_gv && slv1_gm == 0 ? ~m0_wen  :
                        slv1_gv && slv1_gm == 1 ? ~m1_wen  :
                        slv1_gv && slv1_gm == 2 ? ~m2_wen  : 1'b0;

    // ── Slave 2 mux ─────────────────────────────────────────
    assign slv2_addr  = slv2_gv && slv2_gm == 0 ? m0_addr  :
                        slv2_gv && slv2_gm == 1 ? m1_addr  :
                        slv2_gv && slv2_gm == 2 ? m2_addr  : 16'h0;
    assign slv2_wdata = slv2_gv && slv2_gm == 0 ? m0_wdata :
                        slv2_gv && slv2_gm == 1 ? m1_wdata :
                        slv2_gv && slv2_gm == 2 ? m2_wdata : 32'h0;
    assign slv2_wen   = slv2_gv && slv2_gm == 0 ? m0_wen   :
                        slv2_gv && slv2_gm == 1 ? m1_wen   :
                        slv2_gv && slv2_gm == 2 ? m2_wen   : 1'b0;
    assign slv2_ren   = slv2_gv && slv2_gm == 0 ? ~m0_wen  :
                        slv2_gv && slv2_gm == 1 ? ~m1_wen  :
                        slv2_gv && slv2_gm == 2 ? ~m2_wen  : 1'b0;

    // ── Slave 3 mux ─────────────────────────────────────────
    assign slv3_addr  = slv3_gv && slv3_gm == 0 ? m0_addr  :
                        slv3_gv && slv3_gm == 1 ? m1_addr  :
                        slv3_gv && slv3_gm == 2 ? m2_addr  : 16'h0;
    assign slv3_wdata = slv3_gv && slv3_gm == 0 ? m0_wdata :
                        slv3_gv && slv3_gm == 1 ? m1_wdata :
                        slv3_gv && slv3_gm == 2 ? m2_wdata : 32'h0;
    assign slv3_wen   = slv3_gv && slv3_gm == 0 ? m0_wen   :
                        slv3_gv && slv3_gm == 1 ? m1_wen   :
                        slv3_gv && slv3_gm == 2 ? m2_wen   : 1'b0;
    assign slv3_ren   = slv3_gv && slv3_gm == 0 ? ~m0_wen  :
                        slv3_gv && slv3_gm == 1 ? ~m1_wen  :
                        slv3_gv && slv3_gm == 2 ? ~m2_wen  : 1'b0;

    // ====== Combinational read-data mux ======
    // m_rdata is driven directly by the SRAM's registered output.
    // The SRAM's rdata signal is initialized to 0 at time 0 (see sram_bank.sv)
    // so there is no X propagation.
    assign m0_rdata = m0_grant ?
        (m0_slv_id == 0 ? slv0_rdata :
         m0_slv_id == 1 ? slv1_rdata :
         m0_slv_id == 2 ? slv2_rdata :
         m0_slv_id == 3 ? slv3_rdata : 32'h0) : 32'h0;

    assign m1_rdata = m1_grant ?
        (m1_slv_id == 0 ? slv0_rdata :
         m1_slv_id == 1 ? slv1_rdata :
         m1_slv_id == 2 ? slv2_rdata :
         m1_slv_id == 3 ? slv3_rdata : 32'h0) : 32'h0;

    assign m2_rdata = m2_grant ?
        (m2_slv_id == 0 ? slv0_rdata :
         m2_slv_id == 1 ? slv1_rdata :
         m2_slv_id == 2 ? slv2_rdata :
         m2_slv_id == 3 ? slv3_rdata : 32'h0) : 32'h0;

    // ====== Instantiate 4 SRAM banks (4 KB window each, 32-bit words) ======
    localparam int SRAM_DEPTH = 1024;

    sram_bank #(.DEPTH(SRAM_DEPTH), .WIDTH(32), .READ_PORTS(1))
    u_asram (
        .clk  (clk),
        .addr (slv0_addr[11:2]),
        .wdata(slv0_wdata),
        .rdata(slv0_rdata),
        .wen  (slv0_wen),
        .ren  (slv0_ren)
    );

    sram_bank #(.DEPTH(SRAM_DEPTH), .WIDTH(32), .READ_PORTS(1))
    u_wsram (
        .clk  (clk),
        .addr (slv1_addr[11:2]),
        .wdata(slv1_wdata),
        .rdata(slv1_rdata),
        .wen  (slv1_wen),
        .ren  (slv1_ren)
    );

    sram_bank #(.DEPTH(SRAM_DEPTH), .WIDTH(32), .READ_PORTS(1))
    u_osram (
        .clk  (clk),
        .addr (slv2_addr[11:2]),
        .wdata(slv2_wdata),
        .rdata(slv2_rdata),
        .wen  (slv2_wen),
        .ren  (slv2_ren)
    );

    sram_bank #(.DEPTH(SRAM_DEPTH), .WIDTH(32), .READ_PORTS(1))
    u_dsram (
        .clk  (clk),
        .addr (slv3_addr[11:2]),
        .wdata(slv3_wdata),
        .rdata(slv3_rdata),
        .wen  (slv3_wen),
        .ren  (slv3_ren)
    );

endmodule
