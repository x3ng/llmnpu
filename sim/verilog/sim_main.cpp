// ============================================================
// sim_main.cpp — Verilator C++ harness for top_soc
//
// Boots the SoC, toggles reset, runs simulation cycles,
// and checks uart_tx for 'P' (pass) or 'F' (fail).
// ============================================================

#include "Vtop_soc.h"
#include <verilated.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    Vtop_soc* top = new Vtop_soc;

    // Reset sequence
    top->clk   = 0;
    top->rst_n = 0;
    top->eval();

    top->clk = 1;
    top->eval();
    top->clk = 0;
    top->eval();

    top->rst_n = 1;   // release reset
    top->eval();

    // Run simulation
    const int MAX_CYCLES = 1000;
    int cycle;
    bool seen_P = false;

    for (cycle = 0; cycle < MAX_CYCLES; cycle++) {
        top->clk = 1;
        top->eval();

        // Check UART output on high phase
        if (top->uart_tx == 'P') {
            seen_P = true;
            printf("[%4d] UART_TX = 'P' (0x%02X) — TEST PASSED\n",
                   cycle, top->uart_tx);
            break;
        }
        if (top->uart_tx == 'F') {
            printf("[%4d] UART_TX = 'F' (0x%02X) — TEST FAILED\n",
                   cycle, top->uart_tx);
            break;
        }

        top->clk = 0;
        top->eval();

        // Trap check (PicoRV32 trap output — caught illegal instruction)
        // Not connected at top level, so we skip.

        // Print progress every 100 cycles
        if ((cycle > 0) && (cycle % 200 == 0)) {
            printf("[%4d] Running... uart_tx = 0x%02X\n",
                   cycle, top->uart_tx);
        }
    }

    // Final check
    if (!seen_P && top->uart_tx != 'F') {
        printf("[%4d] Timeout after %d cycles — uart_tx = 0x%02X\n",
               cycle, MAX_CYCLES, top->uart_tx);
    }

    top->final();
    delete top;

    return seen_P ? 0 : 1;
}
