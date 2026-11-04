from pathlib import Path
import subprocess

def main():
    p=Path(__file__).parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-fno-fast-math',
        '-shared','-fPIC','-Wl,-Bsymbolic','-I/usr/include/eigen3',str(p/'tournament_kernel.cpp'),
        '-o',str(p/'_tournament.so')],check=True)

if __name__=='__main__':main()
