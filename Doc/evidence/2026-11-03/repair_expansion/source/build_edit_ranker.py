"""Build the cheap candidate-feature kernel outside all timed decisions."""
from pathlib import Path
import subprocess


def main():
    directory=Path(__file__).resolve().parent
    subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-ffp-contract=off','-shared','-fPIC',
                    str(directory/'edit_features.cpp'),'-o',str(directory/'_edit_features.so')],check=True)


if __name__=='__main__':
    main()
