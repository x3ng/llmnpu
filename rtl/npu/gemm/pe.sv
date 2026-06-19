// PE: Processing Element — weight-stationary systolic cell
// Each cycle: acc += A * B_reg. A flows East, B is stationary.
// Reference: tiny-tpu PE design (MIT licensed)

module pe #(
    parameter int K_MAX = 256
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [7:0]  b_in,
    output logic [7:0]  b_out,
    input  logic [7:0]  a_in,
    output logic [7:0]  a_out,
    input  logic        load_b,
    input  logic        clear_acc,
    input  logic        valid_in,
    output logic [31:0] psum_out       // registered = acc (one cycle latency)
);
    logic signed [7:0]  a_reg, b_reg;
    logic signed [31:0] acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)      b_reg <= 8'sd0;
        else if (load_b) b_reg <= b_in;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) a_reg <= 8'sd0;
        else        a_reg <= a_in;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)         acc <= 32'sd0;
        else if (clear_acc) acc <= 32'sd0;
        else if (valid_in)  acc <= acc + a_reg * b_reg;
    end

    always_ff @(posedge clk) begin
        a_out    <= a_reg;
        b_out    <= b_reg;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) psum_out <= 32'sd0;
        else        psum_out <= acc;
    end
endmodule
