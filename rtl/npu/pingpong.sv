// pingpong.sv — Parameterized double-buffer (ping-pong) controller
//
// Two banks (A=0, B=1). Producer writes fill_bank. Consumer reads active_bank.
// ready=1 when active_bank has been filled and is available to consume.
//
// fill_done:  pulse — producer finished writing to fill_bank
// consume_done: pulse — consumer finished reading from active_bank
//
// On fill_done:  fill_bank is marked valid, then moves to an invalid bank
// On consume_done: active_bank is marked invalid, then moves to a valid bank

module pingpong #(
    parameter int BUF_SIZE = 256
) (
    input  logic clk,
    input  logic rst_n,

    input  logic fill_done,
    input  logic consume_done,

    output logic fill_bank,      // which bank producer should fill next
    output logic active_bank,    // which bank consumer should read from
    output logic ready           // non-active bank has data available
);

    logic bank_a_valid, bank_b_valid;
    logic next_fill_bank, next_active_bank;
    logic next_bank_a_valid, next_bank_b_valid;

    always_comb begin
        next_fill_bank    = fill_bank;
        next_active_bank  = active_bank;
        next_bank_a_valid = bank_a_valid;
        next_bank_b_valid = bank_b_valid;

        // Consumer retires the currently active bank.
        if (consume_done) begin
            if (active_bank == 1'b0)
                next_bank_a_valid = 1'b0;
            else
                next_bank_b_valid = 1'b0;
        end

        // Producer publishes the bank it has just filled.
        if (fill_done) begin
            if (fill_bank == 1'b0)
                next_bank_a_valid = 1'b1;
            else
                next_bank_b_valid = 1'b1;
        end

        // Keep active_bank on valid data whenever any bank is available.
        if ((next_active_bank == 1'b0) && !next_bank_a_valid && next_bank_b_valid)
            next_active_bank = 1'b1;
        else if ((next_active_bank == 1'b1) && !next_bank_b_valid && next_bank_a_valid)
            next_active_bank = 1'b0;

        // Point the producer at an invalid bank when one exists.
        if (!next_bank_a_valid)
            next_fill_bank = 1'b0;
        else if (!next_bank_b_valid)
            next_fill_bank = 1'b1;
        else
            next_fill_bank = ~fill_bank;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_bank    <= 1'b0;
            active_bank  <= 1'b0;
            bank_a_valid <= 1'b0;
            bank_b_valid <= 1'b0;
        end else begin
            fill_bank    <= next_fill_bank;
            active_bank  <= next_active_bank;
            bank_a_valid <= next_bank_a_valid;
            bank_b_valid <= next_bank_b_valid;
        end
    end

    assign ready = (active_bank == 1'b0) ? bank_a_valid : bank_b_valid;

endmodule
