"""Compile SIMD TU separately so original Eigen TU keeps its original ISA."""
from pathlib import Path
import subprocess

def main():
    p=Path(__file__).parent
    common=['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-fno-fast-math','-fPIC']
    subprocess.run(common+['-mavx512f','-mavx512dq','-c',str(p/'gamma_soa.cpp'),'-o',str(p/'gamma_soa.o')],check=True)
    subprocess.run(common+['-I/usr/include/eigen3','-c',str(p/'gamma_bridge.cpp'),'-o',str(p/'gamma_bridge.o')],check=True)
    subprocess.run(['g++','-shared','-Wl,-Bsymbolic',str(p/'gamma_soa.o'),str(p/'gamma_bridge.o'),'-o',str(p/'_gamma_soa.so')],check=True)

if __name__=='__main__':main()
