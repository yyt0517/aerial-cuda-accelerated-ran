"""Build the experimental cheap decoder, independently of the PHY library."""
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parent
    subprocess.run(['g++', '-O3', '-std=c++17', '-DNDEBUG', '-shared', '-fPIC',
                    str(root/'masked_decode.cpp'), '-o', str(root/'_masked_decode.so')], check=True)


if __name__ == '__main__':
    main()
