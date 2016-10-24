/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description: some macros
 *
 * Manual:
 * ****************************************************************************/

#ifndef MACRO_H
#define MACRO_H

#define KILOBYTE 1024
#define MEGABYTE (1024 * 1024)

#define BLOCKSIZE 1024
#define BLOCKSIZE_1D 1024
#define BLOCKSIZE_2D 32

#define FILE_NAME_LENGTH 256
#define SQL_COMMAND_LENGTH 256

#define FILE_LINE_LENGTH 1024

#define EDGE_WIDTH_FT 2 // edge width in filtering
#define EDGE_WIDTH_RL 6 // edge width in mask

#define EQUAL_ACCURACY 0.0001

#define SAVE_DELETE(p) \
    if (p != NULL) { delete[] p; p = NULL; }

#define IMAGE_FOR_EACH_PIXEL_IN_GRID(a) \
    for (int y = -a; y < a; y++) \
        for (int x = -a; x < a; x++)

/**
 * This macro loops over all voxels in a grid of certain side length.
 *
 * @param a side length
 */
#define VOLUME_FOR_EACH_PIXEL_IN_GRID(a) \
    for (int z = -a; z < a; z++) \
        for (int y = -a; y < a; y++) \
            for (int x = -a; x < a; x++)

#define B_FACTOR_EST_LOWER_THRES 10 // Angstrom

#endif // MACRO_H
