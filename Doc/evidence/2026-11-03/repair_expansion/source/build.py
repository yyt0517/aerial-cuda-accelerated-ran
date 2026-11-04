"""Build once, outside all measured decision intervals."""
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(["g++", "-O3", "-std=c++17", "-DNDEBUG", "-shared", "-fPIC",
                    "-I/usr/include/eigen3", str(root/"physics_kernel.cpp"),
                    "-o", str(root/"_physics.so")], check=True)


if __name__ == "__main__":
    main()
