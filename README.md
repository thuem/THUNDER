# THUNDER
A particle-filter framework for robust cryoEM 3D reconstruction

## Literature

[bioRvix](https://www.biorxiv.org/content/early/2018/05/23/329169), A particle-filter framework for robust cryoEM 3D reconstruction

### Citation

[Bai R, Wan R, Yan C, et al. Structures of the fully assembled Saccharomyces cerevisiae spliceosome before activation\[J\]. Science, 2018:eaau0325.](http://science.sciencemag.org/content/360/6396/1423)

## Release Note

* Version 1.4.5: This release fixes a bug which appears when 8 or more GPUs are used in a single workstation. This release also fixes some minor bugs.
* Version 1.4.4: This release elevates the support of NVIDIA V100.
* Version 1.4.3: This release increases the stability of 2D classification.
* Version 1.4.2: This release enables THUNDER to run on single workstation with one or multiple GPUs.
* Version 1.4.1: This release fixes minor bugs.
* Version 1.4.0: This release enables THUNDER to run on GPU clusters.

## Quick Start

### Installation from Source

Installation from source requires CMake and a C/C++ compiler with MPI wrapper.

```bash
git clone git@github.com:thuem/THUNDER.git THUNDER
cd THUNDER
mkdir build
cd build
cmake ..
make
make install
```

Installation of specified version of THUNDER, such as double-precision, designed SIMD version or GPU version, is described in the manual in `manual` folder.

### Dependency of GPU Version

CUDA 8.0 or above and NCCL2 are required. You may download CUDA from https://developer.nvidia.com/cuda-toolkit and NCCL2 from https://developer.nvidia.com/nccl.

Please make sure that the proper version of NCCL2 is installed, as it depends on the version of CUDA, operating system and computer architecture.

### Running THUNDER

Please view the manual in `manual` folder.

## Authors

See [AUTHORS.txt](AUTHORS.txt) file.

## License

See [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgements

This work was supported by funds from The National Key Research and Development Program, National Natural Science Foundation of China, Advanced Innovation Center for Structural Biology, Tsinghua-Peking Joint Center for Life Sciences and One-Thousand Talent Program by the State Council of China. We acknowledge the National Supercomputing Center in Wuxi and Tsinghua University Branch of China National Center for Protein Sciences Beijing for providing facility supports in computation.
