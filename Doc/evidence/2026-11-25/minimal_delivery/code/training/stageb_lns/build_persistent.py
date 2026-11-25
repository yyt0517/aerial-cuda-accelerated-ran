"""Build the new versioned library without modifying the frozen PHY binary."""
from pathlib import Path
import subprocess


def main():
    root=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-fno-fast-math',
        '-shared','-fPIC','-Wl,-Bsymbolic','-I/usr/include/eigen3',str(root/'persistent_kernel.cpp'),
        '-o',str(root/'_persistent.so')],check=True)


if __name__=='__main__':main()
