# THUNDER
A particle-filter framework for robust cryoEM 3D reconstruction

## Release Note

* Version 1.4.3: This release increases the stability of 2D classification.
* Version 1.4.2: This release enables THUNDER to run on single workstation with one or multiple GPUs.
* Version 1.4.1: This release fixes minor bugs.
* Version 1.4.0: This release enables THUNDER to run on GPU clusters.

## Quick Start

### Installation from Source

Installation from source requires CMake and a C/C++ compiler with MPI wrapper.

```bash
git clone git@github.com:thuem/THUNDER.git .
cd THUNDER
mkdir build
cd build
cmake ..
make
make install
```

Installation of specified version of THUNDER, such as double-precision and designed SIMD version, is described in the manual in `manual` folder.

### Running THUNDER

Please view the manual in `manual` folder.

## Authors

See [AUTHORS.txt](AUTHORS.txt) file.

## License

See [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgements

This work was supported by funds from The National Key Research and Development Program, National Natural Science Foundation of China, Advanced Innovation Center for Structural Biology, Tsinghua-Peking Joint Center for Life Sciences and One-Thousand Talent Program by the State Council of China. We acknowledge the National Supercomputing Center in Wuxi and Tsinghua University Branch of China National Center for Protein Sciences Beijing for providing facility supports in computation.
