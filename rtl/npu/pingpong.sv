// pingpong.sv — Parameterized double-buffer (ping-pong) controller
//
// Two banks (A=0, B=1). Producer writes fill_bank. Consumer reads active_bank.
// ready=1 when the non-active bank has been filled and is available.
//
// fill_done:  pulse — producer finished writing to fill_bank
// consume_done: pulse — consumer finished reading from active_bank
//
// On fill_done:  fill_bank toggles, bank is marked valid
// On consume_done: active_bank toggles, consumed bank is marked invalid

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

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fill_bank    <= 1'b0;
            active_bank  <= 1'b0;
            bank_a_valid <= 1'b0;
            bank_b_valid <= 1'b0;
        end else begin
            // Producer finished filling: mark bank valid, toggle fill_bank
            if (fill_done) begin
                if (fill_bank == 1'b0)
                    bank_a_valid <= 1'b1;
                else
                    bank_b_valid <= 1'b1;
                fill_bank <= ~fill_bank;
            end

            // Consumer finished reading: mark bank invalid, toggle active_bank
            if (consume_done) begin
                if (active_bank == 1'b0)
                    bank_a_valid <= 1'b0;
                else
                    bank_b_valid <= 1'b0;
                active_bank <= ~active_bank;
            end
        end
    end

    // ready: the non-active bank is valid (consumer can switch to it)
    assign ready = (active_bank == 1'b0) ? bank_b_valid : bank_a_valid;

endmodule
