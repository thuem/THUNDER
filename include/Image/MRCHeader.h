/** @file
 *  @author Mingxu Hu 
 *  @version 1.4.11.080913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR    | TIME       | VERSION       | DESCRIPTION
 *  ------    | ----       | -------       | -----------
 *  Mingxu Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Xiao Long | 2018/09/13 | 1.4.11.080913 | add documentation
 *
 *  @brief MRCHeader.h contains the main header of the MRC format, including fixed format values for metadata about the images/volumes.
 *
 *  MRC is a file format that has become industry standard in cryo-electron microscopy (cryoEM) and electron tomography (ET), where the result of the technique is a three-dimensional grid of voxels each with a value corresponding to electron density or electric potential.
 *
 *  Reference: Cheng, Anchi; Henderson, Richard; Mastronarde, David; Ludtke, Steven J.; Schoenmakers, Remco H.M.; Short, Judith; Marabini, Roberto; Dallakyan, Sargis; Agard, David; Winn, Martyn (November 2015). "MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography". Journal of Structural Biology. 192 (2): 146–150. doi:10.1016/j.jsb.2015.04.002.
 */ 

#ifndef MRC_HEADER_H
#define MRC_HEADER_H

/**
 * @brief The constitution of MRC main header.
 *
 * The main header is limited to 1024 bytes, but includes unassigned space in anticipation of future extensions.
 */
struct MRCHeader:
{
    int nx;            /**< number of columns (fastest changing in map) */
    int ny;            /**< number of rows */
    int nz;            /**< number of sections (slowest changing in map) */
    int mode;          /**< data type: 
                        *   0 - image->signed 8-bit bytes range -128 to 127; 
                        *   1 - image->16-bit halfwords; 
                        *   2 - image->32-bit reals; 
                        *   3 - transform->complex 16-bit integers; 
                        *   6 - image->unsigned 16-bit range 0 to 65535
                        **/   
    int nxstart;       /**< number of first column in map (Default = 0) */
    int nystart;       /**< number of first row in map */
    int nzstart;       /**< number of first section in map */
    int mx;            /**< number of intervals along X */
    int my;            /**< number of intervals along Y */
    int mz;            /**< number of intervals along Z */
    float cella[3];    /**< cell dimensions in angstroms */
    float cellb[3];    /**< cell angles in degrees */
    int mapc;          /**< axis corresp to cols (1,2,3 for X,Y,Z) */
    int mapr;          /**< axis corresp to rows (1,2,3 for X,Y,Z) */
    int maps;          /**< axis corresp to sections (1,2,3 for X,Y,Z) */
    float dmin;        /**< minimum density value */
    float dmax;        /**< maximum density value */
    float dmean;       /**< mean density value */
    int ispg;          /**< space group number 0 or 1 (default=0) */
    int nsymbt;        /**< number of bytes used for symmetry data (0 or 80) */
    char extra[100];   /**< extra space used for anything   - 0 by default */
    float origin[3];   /**< origin in X,Y,Z used for transforms */
    char map[4];       /**< character string 'MAP ' to identify file type */
    int machst;        /**< machine stamp */
    float rms;         /**< rms deviation of map from mean density */
    int nlabels;       /**< number of labels being used */
    char label[10][80];/**< ten 80-character text labels 
                        *   Symmetry records follow - if any - stored as text 
                        *   as in International Tables, operators separated 
                        *   by * and grouped into 'lines' of 80 characters 
                        *   (ie. symmetry operators do not cross the ends of 
                        *   the 80-character 'lines' and the 'lines' do not 
                        *   terminate in a *). 
                        *   Data records follow. 
                        **/
};

#endif // MRC_HEADER_H
