"""Build the approximate sparse PHY cache outside timed search intervals."""
from pathlib import Path
import subprocess

def main():
    root=Path(__file__).resolve().parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-shared','-fPIC',
                    str(root/'draft_kernel.cpp'),'-o',str(root/'_draft.so')],check=True)

if __name__=='__main__':main()
