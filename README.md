# THUNDER

A particle-filter framework for robust cryoEM 3D reconstruction

## Publication

[Nature Methods](https://www.nature.com/articles/s41592-018-0223-8), A particle-filter framework for robust cryo-EM 3D reconstruction

## Download

Click [here](https://github.com/thuem/THUNDER/archive/master.zip) to download THUNDER v1.4.14 (stable release).

## Citation

* [Single particle cryo-EM reconstruction of 52 kDa streptavidin at 3.2 Angstrom resolution\[J\]. Nature Communications.](https://doi.org/10.1038/s41467-019-10368-w)
* [Molecular Basis for Ligand Modulation of a Mammalian Voltage-Gated Ca2+ Channel\[J\]. Cell.](https://doi.org/10.1016/j.cell.2019.04.043)
* [Adeno-associated virus 2 bound to its cellular receptor AAVR\[J\]. Nature Microbiology.](https://doi.org/10.1038/s41564-018-0356-7)
* [Mechanism of DNA translocation underlying chromatin remodelling by Snf2\[J\]. Nature.](https://doi.org/10.1038/s41586-019-1029-2)
* [Structures of the human spliceosomes before and after release of the ligated exon\[J\]. Cell Research.](https://doi.org/10.1038/s41422-019-0143-x)
* [Seneca Valley virus attachment and uncoating mediated by its receptor anthrax toxin receptor 1\[J\]. PNAS.](https://www.pnas.org/content/115/51/13087)
* [Structures of the fully assembled Saccharomyces cerevisiae spliceosome before activation\[J\]. Science.](http://science.sciencemag.org/content/360/6396/1423)
* [Modulation of cardiac ryanodine receptor 2 by calmodulin\[J\]. Nature.](https://www.nature.com/articles/s41586-019-1377-y)

## Quick Start

Please find the [manual](https://thuem.github.io/THUNDER/).

### GUI Installation

Installation of GUI requires Qt5. You may download an open source version Qt5 from https://www1.qt.io/download-open-source-access/. You may also install Qt5 with `yum` or `apt-get`.

```bash
cd THUNDER/gui/thunder_stackview
mkdir build
cd build
qmake ..
make
make install
```

### Viewing Result of 2D Classification and Selecting Desired Particles

You may type `thunder_stackview` to get help.

If you want simply view result of 2D classification, you may type `thunder_stackview Reference_Round_XXX.mrcs` or `thunder_stackview Reference_Final.mrcs`. Moreover, if you want to select desired particles, please type `thunder_stackview Reference_Round_XXX.mrcs -thu Meta_Round_XXX.thu` or `thunder_stackview Reference_Final.mrcs -thu Meta_Final.thu`. After selecting the classes you desired, please use the `Save thu` button to save the selection.

## Authors

See [AUTHORS.txt](AUTHORS.txt) file.

## License

See [LICENSE.txt](LICENSE.txt) file for details.

## Release Note

* Version 1.4.14: This release fixes a bug in 2D, 3D classification and a bug of FFT trasnformation occuring when box size is very large.
* Version 1.4.13: This release fixes the segment fault which occurs when boxsize is very large, e.g., over 1000.
* Version 1.4.12: This release fixes a plenty of bugs and adds several new features.
* Version 1.4.11: This release fixes a bug which occurs during the initialisation step of a few datasets.
* Version 1.4.10: This release fixes a compilation bug which occurs when CUDA version is below 9.0.
* Version 1.4.9: Graphic User Inference (GUI) `thunder_stackview` is released. It is used for viewing the result of 2D classification and selecting desired particles.
* Version 1.4.8: This release elevates the support of NVIDIA GeForce GTX 1080/1080Ti.
* Version 1.4.7: This release fixes minor bugs.
* Version 1.4.6: This release fixes a bug occurring during applying mask on the reference.
* Version 1.4.5: This release fixes a bug which appears when 8 or more GPUs are used in a single workstation. This release also fixes some minor bugs.
* Version 1.4.4: This release elevates the support of NVIDIA V100.
* Version 1.4.3: This release increases the stability of 2D classification.
* Version 1.4.2: This release enables THUNDER to run on single workstation with one or multiple GPUs.
* Version 1.4.1: This release fixes minor bugs.
* Version 1.4.0: This release enables THUNDER to run on GPU clusters.

## Acknowledgements

This work was supported by funds from The National Key Research and Development Program, National Natural Science Foundation of China, Advanced Innovation Center for Structural Biology, Tsinghua-Peking Joint Center for Life Sciences and One-Thousand Talent Program by the State Council of China. We acknowledge the National Supercomputing Center in Wuxi and Tsinghua University Branch of China National Center for Protein Sciences Beijing for providing facility supports in computation.
