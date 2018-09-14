/** @file
 *  @author ???
 *  @author Xiao Long 
 *  @version 1.4.11.080914
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  ???       | ???        | ???           | new file
 *  Xiao Long | 2018/09/14 | 1.4.11.080914 | add documentation
 *
 *  @brief MRCHeader.h contains the MRC main header constitution according to the MRC2014 format.
 *
 *  MRC is a file format that has become industry standard in cryo-electron microscopy (cryoEM) and electron tomography (ET), where the result of the technique is a three-dimensional grid of voxels each with a value corresponding to electron density or electric potential. It was developed by the MRC (Medical Research Council, UK) Laboratory of Molecular Biology in 1982 and standarlised in 2014.
 *
 *  The MRC2014 format describes a binary file consisting of three parts. The first part, the main header, contains fixed format values for metadata about the images/volumes. The second part is a variable length extended header, originally designed to include symmetry operators for crystallographic applications, but now used by different software to hold various additional metadata instead. Finally, the third part contains the actual image/volume data, with grid values represented as one of a range of possible data types, according to the "mode" of the map.
 *
 *  Reference: Cheng A., Henderson R., et al. MRC2014: extensions to the MRC format header for electron cryo-microscopy and tomography. Journal of Structural Biology. 2015; 192: 146-150. doi:10.1016/j.jsb.2015.04.002
 
 */ 

#ifndef MRC_HEADER_H
#define MRC_HEADER_H

/**
 * @brief The detailed specification of MRC main header according to the MRC2014 format.
 *
 * The length of main header is 1024 bytes, organized as 56 4-byte words followed by space for 10 80-byte text labels, including unassigned space in anticipation of future extensions.
 */
struct MRCHeader:
{
    int nx;              /**< number of columns in 3D data array (fast axis) */
    int ny;              /**< number of rows in 3D data array (medium axis) */
    int nz;              /**< number of sections in 3D data array (slow axis) */
    int mode;            /**< data type: 0 -> 8-bit signed integer (range -128 to 127); 1 -> 16-bit signed integer; 2 -> 32-bit signed real; 3 -> transform : complex 16-bit integers; 4 -> transform : complex 32-bit reals; 6 -> 16-bit unsigned integer */   
    int nxstart;         /**< location of first column in unit cell */
    int nystart;         /**< location of first row in unit cell */
    int nzstart;         /**< location of first section in unit cell */
    int mx;              /**< sampling along X axis of unit cell */
    int my;              /**< sampling along Y axis of unit cell */
    int mz;              /**< sampling along Z axis of unit cell */
    float cella[3];      /**< cell dimensions in angstroms */
    float cellb[3];      /**< cell angles in degrees */
    int mapc;            /**< axis corresp to cols (1,2,3 for X,Y,Z) */
    int mapr;            /**< axis corresp to rows (1,2,3 for X,Y,Z) */
    int maps;            /**< axis corresp to sections (1,2,3 for X,Y,Z) */
    float dmin;          /**< minimum density value */
    float dmax;          /**< maximum density value */
    float dmean;         /**< mean density value */
    int ispg;            /**< space group number */
    int nsymbt;          /**< size of extended header (which follows main header) in bytes */
    char extra[100];     /**< extra space used for anything - 0 by default */
    float origin[3];     /**< phase origin (pixels) or origin of subvolume (A) */  
    char map[4];         /**< character string 'MAP ' to identify file type */
    int machst;          /**< machine stamp encoding byte ordering of data */
    float rms;           /**< rms deviation of map from mean density */
    int nlabels;         /**< number of labels being used */
    char label[10][80];  /**< ten 80-character text labels */
};

#endif // MRC_HEADER_H
