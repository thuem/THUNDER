# Installation

## Requirement of Installation

### Basic Requirement

1. C/C++ compiler supporting C++98 standard along with MPI wrapper
2. **CMake**

We recommend **gcc** and **Intel C/C++ compiler** as C/C++ compiler. Moreover, **gcc42** has been tested as the oldest supporting version of **gcc**. **OpenMPI** and **MPICH** both can be used as MPI standard. In Tsinghua, we use **openmpi-gcc43** as the C/C++ compiler for compiling THUNDER.

**CMake** is a tool for configuring source code for installation.

**openmpi-gcc43** is open-source software, which can easily installed using **yum** on CentOS and **apt-get** on Ubuntu. **CMake** has been already installed in most Linux operating systems. If not, it can also be conveniently installed by **yum** on CentOS and **apt-get** on Ubuntu.

### Additional Requirement of Installing GPU Version

CUDA 8.0 or above and NCCL2 are required. You may download CUDA from <https://developer.nvidia.com/cuda-toolkit> and NCCL2 from <https://developer.nvidia.com/nccl>.

Please make sure that the proper version of NCCL2 is installed, as it depends on the version of CUDA, operating system and computer architecture. CUDA 8 or higher version of CUDA is required for installing GPU version of THUNDER.

## Installing from Source Code

### Get Source Code

THUNDER is a open-source software package, source code of which is held on Github. You may download the source code at <https://github.com/thuem/THUNDER> or execute `https://github.com/thuem/THUNDER.git` in CLI.

### Configure Using **CMake**

In THUNDER source code directory, please type in the following commands for configuring source code.

```bash
mkdir build
cd build
cmake ..
```

### Configuration Variables of **CMake** (Advanced)

<details>

<summary>installation directory</summary>

<p>
You may assign installation directory using <code>-DCMAKE_INSTALL_PREFIX="install_dir"</code> during configuration, where <code>install_dir</code> stands for where you want THUNDER to be installed.
</p>

</details>

<details>

<summary>single and double precision</summary>

<p>

THUNDER can be compiled into single-float precision version or double-float precision version, by <code>SINGLE_PRECISION</code> variable. The default version is single-float precision. However, you may force it compiling into double-float precision version, by adding parameter <code>-DSINGLE_PRECISION="off"</code> during configuring using <b>cmake</b>.

</p>

</details>

<details>

<summary>CPU and GPU version</summary>

<p>

By default, THUNDER will try to compile a CPU version and a GPU version, into <b>thunder_cpu</b> and <b>thunder_gpu</b>, respectively. If it can not find essentail GPU components in the environment, it will omit the CPU version compilation. You may force it not compiling GPU version, by adding parameter <code>-DGPU_VERSION='off'</code>.

</p>

</details>

<details>

<summary>SIMD</summary>

<p>

THUNDER uses SIMD instructions for accelerating. When you compile THUNDER, SIMD acceleration can be turned on or off by <code>ENABLE_SIMD</code> variable. The default version is with SIMD acceleration on. However, you may force it compiling into a non-SIMD version, by adding parameter <code>-DENABLE_SIMD="off"</code> during configuring using <b>cmake</b>.

AVX256 and AVX512 SIMD instructions are currently supported by THUNDER. By default, AVX256 is enabled and AVX512 is disabled. You can manually enable or disable them by the variable <code>ENABLE_AVX256</code> and <code>ENABLE_AVX512</code>, respectively, by the same method as described above.

It is worth mentioned that you may check whether the CPUs and C/C++ compiler support AVX512 or not, before compiling THUNDER using AVX512. For example, CPUs should be KNL or Xeon newer than Skylake. Meanwhile, if you compile using <b>GCC</b>, please make sure it is newer than version 4.9.3. If you compile with <b>Intel C/C++ compiler</b>, please check up its support on AVX512.

</p>

</details>

### Compile and Stage Binaries into Environment

Please type in the following command to compile source code using 20
threads and stage binaries into installation diectory. You may change the number after `-j` to be number of threads you desire for compiling.

```bash
make -j20
make install
```

After compiling and installation, several folders will appear under the installation directory. **include** containing header files, **bin** containing executable binaries, **lib** containing several libraries, **script** containing scripts needed. The compiled binaries are listed as

<select id = "thunder_binarys" onchange="thunder_binary_explanation();">
    <option value = "">--Please choose an option--</option>
    <option value = "thunder_cpu: the main executable for 2D/3D classification and refinement, CPU version">thunder_cpu</option>
    <option value = "thunder_gpu: the main executable for 2D/3D classification and refinement, GPU version">thunder_gpu</option>
    <option value = "thunder_average: average two volumes">thunder_average</option>
    <option value = "thunder_genmask: generate mask based on a volume, the value of the threshold and the value of edgewidth">thunder_genmask</option>
    <option value = "thunder_postprocess: post-process based on two half maps">thunder_postprocess</option>
    <option value = "thunder_lowpass: perform low-pass filtering on a volume">thunder_lowpass</option>
    <option value = "thunder_resize: resize a volume">thunder_resize</option>
</select>
<p>
<textarea style = "width: 100%;" id = "thunder_binary" value = ""></textarea>
</p>

<script>
    function thunder_binary_explanation()
    {
        var _sel = document.getElementById("thunder_binarys");
        var index = _sel.selectedIndex;
        var _val = _sel.options[index].value;
        var input = document.getElementById("thunder_binary");
        input.value = _val;
    }
</script>

For the purpose of convenience, you may stage binaries into environment. For example, you may add the following command into shell configuration file
```bash
setenv PATH=install_dir:$PATH
```
when **csh** or **tcsh** is used as shell. Meanwhile, you may add the following command into shell configuration file when **bash**, **zsh** or **ksh** is used as shell.
```bash
export PATH=install_dir:$PATH
```

After staging binaries into environment, you may directly access these binaries by typing their filenames in shell.

# Submit Your Job

**thunder_cpu** and **thunder_gpu** is the core program of THUNDER. It executes 2D or 3D classification and refinement. It reads in a JSON parameter file. After parsing the JSON parameter, it reads in initial model, a **.thu** file and particle images. It also reads in mask if necessary.

Three steps are needed, before submiiting your job. Step one, set up **.thu** file. Step two, configure a JSON parameter file. Step three, check whether all files mentioned in the JSON parameter existed and in the right directory.

## Set Up **.thu** File

THUNDER uses **.thu** file for storing information of each particle image. **.thu** file is a space-separate tabular file as each column stands for a specific variable, as listed below.

<select id = "thu_defs" onchange="thu_def_explanation();">
    <option value = "">--Please choose an option--</option>
    <option value = "Acceleration voltage of electron in the microscope. The measuring unit is Voltage.">[1] Voltage</option>
    <option value = "Defocus on the first axle of the defocus ellipsoid. Please refer to CTFFIND3 for more detail. The measuring unit is Angstrom.">[2] DefocusU</option>
    <option value = "Defocus on the second axle of the defocus ellipsoid. Please refer to CTFIND3 for more detail. The measuring unit is Angstrom.">[3] DefocusV</option>
    <option value = "Rotation angle of the defocus ellipsoid. Please refer to CTFFIND3 for more detail. The measuring unit is radius.">[4] DefocusTheta</option>
    <option value = "Spherical aberration. The measuring unit is Angstrom.">[5] Cs</option>
    <option value = "Amplitude. Please refer to CTFFIND3 fore more detail.">[6] Amplitude</option>
    <option value = "Phase shift. The measuring unit is radius.">[7] Phase Shift</option>
    <option value = "Directory path of each single particle image. If it is in a stack, the index of this image in the stack is before @ symbol.">[8] Particle of Particle</option>
    <option value = "Path of Micrograph">[9] Particle of Micrograph</option>
    <option value = "Coordinate X in Micrograph">[10] Coordinate X in Micrograph</option>
    <option value = "Coordinate Y in Micrograph">[11] Coordinate Y in Micrograph</option>
    <option value = "Group ID">[12] Group ID</option>
    <option value = "Class ID">[13] Class ID</option>
    <option value = "the 1st Element of the Unit Quaternion">[14] the 1st Element of the Unit Quaternion</option>
    <option value = "the 2nd Element of the Unit Quaternion">[15] the 2nd Element of the Unit Quaternion</option>
    <option value = "the 3rd Element of the Unit Quaternion">[16] the 3rd Element of the Unit Quaternion</option>
    <option value = "the 4th Element of the Unit Quaternion">[17] the 4th Element of the Unit Quaternion</option>
    <option value = "the 1st Standard Deviation of Rotation">[18] the 1st Standard Deviation of Rotation</option>
    <option value = "the 2nd Standard Deviation of Rotation">[19] the 2nd Standard Deviation of Rotation</option>
    <option value = "the 3rd Standard Deviation of Rotation">[20] the 3rd Standard Deviation of Rotation</option>
    <option value = "Translation X">[21] Translation X</option>
    <option value = "Translation Y">[22] Translation Y</option>
    <option value = "Standard Deviation of Translation X">[23] Standard Deviationof Translation X</option>
    <option value = "Standard Deviation of Translation Y">[24] Standard Deviationof Translation Y</option>
    <option value = "Defocus Factor">[25] Defocus Factor</option>
    <option value = "Standard Deviation of Defocus Factor">[26] Standard Deviation of Defocus Factor</option>
    <option value = "Score">[27] Score</option>
</select>

<p>
<textarea style = "width: 100%;" id = "thu_def" value = ""></textarea>
</p>

<script>
    function thu_def_explanation()
    {
        var _sel = document.getElementById("thu_defs");
        var index = _sel.selectedIndex;
        var _val = _sel.options[index].value;
        var input = document.getElementById("thu_def");
        input.value = _val;
    }
</script>

**.thu** file is generated by **thunder_cpu** or **thunder_gpu** at the end of each iteration to save the information of each particle image.

### Generate **.thu** from Relion

**.thu** file can be converted from and to STAR file of Relion by script **STAR_2_THU.py** and **THU_2_STAR.py** respectively, by the following commands.

```bash
python STAR_2_THU.py -i filename.star -o filename.thu
python THU_2_STAR.py -i filename.thu -o filename.star
```

You can find these two scripts under directory <code>install_dir/script</code>.

It is worth noticed that both of two scripting only convert CTF information but not rotation and translation information. Thus, **.thu** files converted from STAR files can be only used for global search stage of **thunder**. Meanwhile, **.thu** files generated by **thunder** can be used for global search, local search and CTF search. The precise meaning of global search, local search and CTF search will be further discussed in detail in later section.

### Generate **.thu** from Frealign (TBD)

### Generate **.thu** from SPIDER (TBD)

## Configure a JSON Parameter File {#sec:JSON}

**thunder_cpu** and **thunder_gpu** reads in a JSON file as parameter. You may change the values of the keys to fit your purpose. The definition of keys in this JSON parameter file is listed in Table.

**thunder_cpu** and **thunder_gpu** divides 3D refinement into three stages: global search, local search and CTF search. During global search, the rotation and translation result of the previous iteration will **not** inherited into the next iteration. Meanwhile, during local search, the rotation and translation of each particle image will be adjust based on the result of the previous iteration. During CTF search, the CTF parameters will be adjusted for achieving better resolution.

Meanwhile, 2D and 3D classification of **thunder_cpu** and **thunder_gpu** typically only involve global search.

You may find a demo version of this JSON parameter file named **demo.json** under directory **install\_dir/script**.

### Processes and Threads

**Processes and Threads When Using CPU Version THUNDER**

**thunder** needs at least 3 processes. It has perfect linear speed-up when number of nodes increases. Thus, please use as many nodes as possible. We high recommend assigning a node with only one process and using multiple cores in each node by threads. For example, if you have 100 nodes and each node has 20 cores, you may use 100 processes for running **thunder**, and each process should generate 20 threads to achieve maximum usage of computing resource. By changing the value of the key **Number of Threads Per Process** in the JSON parameter file, you may set the number of threads of each process to which you desire. In this example, this value should be set to 20.

**Processes and Threads When Using GPU Version THUNDER**

There is two ways of running GPU version THUNDER.

The most common way is to run THUNDER on a single workstation with one GPU or multiple GPUs. In this method, please set the number of MPI processes to 3, and **Number of Threads Per Process** to the number of CPU cores of this workstation.

The second way of running THUNDER is to run it on a GPU cluster. Similar to running on a CPU clusters, please use one process per node and using multiple cores in each node by threads.

**Master and Slave Processes, and How to Fully Use Computing Resources**

**thunder** divides MPI processes into three parts, a master process (rank 0), slave process(es) in hemisphere A (rank 1, rank 3, rank 5...), slave process(es) in hemisphere B (rank 2, rank 4, rank 6...). The slave processes carries out most of the computing workload, meanwhile the master process is simply a manager. Thus, when running THUNDER on clusters (either CPU or GPU), the master process should be assigned along with a slave process for fully using computing resources. For example, when **thunder** should be run on 4 nodes, 5 process should be initialised, where rank 0 and 1 should be assigned to node 0, rank 2 should be assigned to node 1, rank 3 should be assigned to node 2 and rank 4 should be assigned to node 3.

## Submit

Please examine whether you have generated the correct **.thu** file and configured the JSON parameter file properly, and make sure that the initial model and mask (if necessary) are placed in the right directory. Now, you can submit you job. You may leave it to the cluster job managing software, or you may assign nodes manually by **mpirun**.

# Get Your Result

A log file named **thunder.log** will appear in your submitting directory, recording the state of your job.

In the destination directory, the density maps are outputted as **Reference\_xxx\_A\_Round\_xxx.mrc** and **Reference\_xxx\_B\_Round\_xxx.mrc**, during 3D refinement or classification. For example, the density map of the 5th reference of round 15 from hemisphere A has the filename **Reference\_005\_A\_Round\_015.mrc**. On contrast, the 5th reference of round 15 from hemisphere B has the filename **Reference\_005\_B\_Round\_015.mrc**.

Meanwhile, during 2D classification, the density maps of each round are stored in a MRC stack. For example, the density maps of round 15 has the filename **Reference\_Round\_015.mrcs** which contains $N$ slices of images. $N$ stands for the number of classes.

FSC/FRCs are outputted as **FSC\_Round\_xxx.txt**. The first column of this file is signal frequency in pixel. The seconds column is signal frequency in Angstrom. From the third column to the rest of columns, the FSC of each reference is listed in order.

During classification, the resolution and ratio of images of each class is listed in a file named **Class\_Info\_Round\_xxx.txt**. Each row of this file stands for a class in order. The first column is the index of each class, the second column is the resolution in Angstrom of each class and the third column is the ratio of image of each class.

The rotation and translation information of each particle at each iteration is outputted as **Meta\_Round\_xxx.thu**, which follows the **.thu** file format. For example, rotation and translation of round 15 has the filename **Meta\_Round\_015.thu**.

# Typically Workflow

The typically workflow of cryo-EM single particle analysis includes 3 steps: 2D classification, 3D classification and 3D refinement.

## 2D Classification

The first step of cryo-EM single particle analysis is 2D classification for removing ice and bad particles.

You can find a demo version of this JSON parameter file for 2D classification named **demo_2D.json** under directory **install_dir/script**. There are some options worth noticed in this JSON parameter file. They are listed below.

1. Local Search. Performing local search or not will **NOT** affect the result of 2D classification. However, it gives you a higher resolution density map for examining the detail of the 2D density map. You may turn it off when the computing resource is limited.
2. Number of Classes. It stands for the number of classes you want the images to be classified into.
3. Initial Resolution (Angstrom). It is recommended to start from lower resolution for achieving ideal result of classification.
4. Symmetry. Symmetry has **NO** effect on 2D classification.
5. Initial Model. It is recommended to use a blank initial model in 2D classification. **Please leave it empty**.
6. Calculate FSC Using Core Region. It is not supported in 2D classification. Please turn it off, otherwise a warning will be raised and **thunder** will turn it off forcefully.
7. Calculate FSC Using Masked Region. It is not supported in 2D classification. Please turn it off, otherwise a warning will be raised and **thunder** will turn it off forcefully.
8. Particle Grading. It is not recommended to use particle grading in 2D classification, because the importance of bad particles may be overlooked when particle grading is turned on.
9. Performing Reference Mask. It is **NOT** supported to use provided mask in 2D classification. If so, a fatal error will occur. Please turn it off.

## 3D Classification

The next step of cryo-EM single particle analysis is 3D classification for removing particles belong to “wrong” conformation.

You can find a demo version of this JSON parameter file for 3D classification named **demo\_3D.json** under directory **install\_dir/script**. There are some options worth noticed in this JSON parameter file. They are listed below.

1. Local Search. Performing local search or not will **NOT** affect the result of 3D classification. However, it gives you a higher resolution density map for examining the detail of the 2D density map. You may turn it off when the computing resource is limited.
2. Number of Classes. It stands for the number of classes you want the images to be classified into.
3. Initial Resolution (Angstrom). It is recommended to start from lower resolution for achieving ideal result of classification.
4. Particle Grading. It is **NOT** recommended to use particle grading in 3D classification, because the importance of “noisy” particles may be overlooked when particle grading is turned on.

## 3D Refinement

The final step of cryo-EM single particle analysis is 3D refinement for achieving high resolution density map. You may turn on **particle grading** and **CTF search** for obtaining more information in density map.

You can find a demo version of this JSON parameter file for 3D classification named **demo.json** in this package. There are some options worth noticed in this JSON parameter file. They are listed below.

1. CTF Search. You can refine CTF parameters using CTF search. It may cost some computing resource.
2. Particle Grading. It is recommend to turn on particle grading in refinement.
