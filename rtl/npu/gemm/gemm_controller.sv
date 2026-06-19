// GEMM Controller: 5-stage pipeline FSM
// IDLE → LOAD_B(1cy) → PREFETCH(1cy) → COMPUTE(k_count cy) → REDUCE(1cy) → WRITEBACK(1cy)
// B is broadcast to all PE rows in LOAD_B (1 cycle, not systolic shift).
// state_out exported so systolic_array can derive reduce/writeback strobes.

module gemm_controller (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [7:0]  k_count,
    output logic        load_b,
    output logic        clear_acc,
    output logic        done,
    output logic        busy,
    output logic [2:0]  state_out
);
    typedef enum logic [2:0] {
        IDLE       = 3'd0,
        LOAD_B     = 3'd1,
        PREFETCH   = 3'd2,
        COMPUTE    = 3'd3,
        REDUCE     = 3'd4,
        WRITEBACK  = 3'd5
    } state_t;

    state_t state, next_state;
    logic [7:0] k_cnt;

    // State register + k_cnt (k_cnt only advances during COMPUTE)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            k_cnt <= 8'd0;
        end else begin
            state <= next_state;
            if (state == COMPUTE)
                k_cnt <= k_cnt + 8'd1;
            else
                k_cnt <= 8'd0;
        end
    end

    // Next-state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE:      if (start)                   next_state = LOAD_B;
            LOAD_B:                                 next_state = PREFETCH;
            PREFETCH:                               next_state = COMPUTE;
            COMPUTE:   if (k_cnt == k_count - 1)    next_state = REDUCE;
            REDUCE:                                 next_state = WRITEBACK;
            WRITEBACK:                              next_state = IDLE;
            default:                                next_state = IDLE;
        endcase
    end

    // Combinational control outputs
    assign load_b    = (state == LOAD_B);
    assign clear_acc = (state == PREFETCH);
    assign done      = (state == WRITEBACK);
    assign busy      = (state != IDLE);
    assign state_out = state;

endmodule
