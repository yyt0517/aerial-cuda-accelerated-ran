"""Build a separate library; existing full/cache/PHY binaries stay frozen."""
from pathlib import Path
import subprocess

def main():
    p=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-fno-fast-math',
        '-shared','-fPIC','-Wl,-Bsymbolic','-I/usr/include/eigen3',str(p/'bounded_kernel.cpp'),
        '-o',str(p/'_bounded.so')],check=True)

if __name__=='__main__':main()
