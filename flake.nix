{
  description = "NPU Subsystem Design — RV32IM RISC-V + Custom NPU Accelerator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
        };

        # RISC-V 32-bit bare-metal cross-compiler
        riscv32-pkgs = import nixpkgs {
          inherit system;
          crossSystem = {
            config = "riscv32-none-elf";
            libc = "newlib";
          };
        };

        # Common RTL tools
        rtlTools = with pkgs; [
          # Simulation
          iverilog
          verilator
          gtkwave

          # Synthesis
          yosys

          # Python verification
          python3
          python3Packages.cocotb
          python3Packages.numpy
          python3Packages.torch
          python3Packages.onnx

          # Code formatting
          verible

          # Build
          gnumake
          cmake
          ninja

          # Misc
          graphviz
        ];

        # Software build tools
        swTools = with pkgs; [
          gcc
          binutils
        ];

        # RISC-V cross-compilation
        riscvTools = [
          riscv32-pkgs.buildPackages.gcc
          riscv32-pkgs.buildPackages.binutils
        ];

      in
      {
        devShells = {
          # Full development shell — everything
          default = pkgs.mkShell {
            name = "npu-dev";
            buildInputs = rtlTools ++ swTools ++ riscvTools;

            shellHook = ''
              echo "🔧 NPU Development Environment"
              echo "   RTL sim:  iverilog | verilator | cocotb"
              echo "   Synthesis: yosys $(yosys --version 2>/dev/null || echo '')"
              echo "   RISC-V:    riscv32-none-elf-gcc"
              echo "   Python:    $(python3 --version)"
            '';

            # Fix verilator --trace-fst zlib issue
            NIX_CFLAGS_COMPILE = "-I${pkgs.zlib.dev}/include";
            NIX_LDFLAGS = "-L${pkgs.zlib}/lib";
          };

          # Minimal shell — just RTL simulation
          rtl = pkgs.mkShell {
            name = "npu-rtl";
            buildInputs = with pkgs; [
              iverilog
              verilator
              gtkwave
              yosys
              python3
              python3Packages.cocotb
              verible
            ];
          };

          # Software-only shell
          sw = pkgs.mkShell {
            name = "npu-sw";
            buildInputs = with pkgs; [
              gcc
              binutils
              gnumake
              cmake
            ] ++ riscvTools;
          };
        };

        # Formatting check
        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
