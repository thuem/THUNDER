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

#define VERBOSE_LEVEL_0

#define VERBOSE_LEVEL_1

//#define VERBOSE_LEVEL_2

//#define VERBOSE_LEVEL_3

//#define VERBOSE_LEVEL_4

#define IMG_VOL_BOUNDARY_NO_CHECK

#define MATRIX_BOUNDARY_NO_CHECK

/**
 * 2D mode
 */
#define MODE_2D 0

/**
 * 3D mode
 */
#define MODE_3D 1

#define IF_MODE_2D if (_mode == MODE_2D)

#define IF_MODE_3D if (_mode == MODE_3D)

#define NT_MODE_2D if (_mode != MODE_2D)

#define NT_MODE_3D if (_mode != MODE_3D)

/**
 * 1 KB
 */
#define KILOBYTE 1024

/**
 * 1 MB
 */
#define MEGABYTE (1024 * 1024)

#define BLOCKSIZE 1024

#define BLOCKSIZE_1D 1024

#define BLOCKSIZE_2D 32

/**
 * maximum length of filename
 */
#define FILE_NAME_LENGTH 1024

/**
 * maximum length of SQL command
 */
#define SQL_COMMAND_LENGTH 1024

/**
 * maximum length of a line in a file
 */
#define FILE_LINE_LENGTH (1024 * 1024)

#define FILE_WORD_LENGTH 1024

/**
 * edge width in Fourier space
 */
#define EDGE_WIDTH_FT 4

/**
 * edge width in real space
 */
#define EDGE_WIDTH_RL 6
//#define EDGE_WIDTH_RL 12
//#define EDGE_WIDTH_RL 20

/**
 * threshold for determining that one is equal to another
 */
#define EQUAL_ACCURACY 1e-4

/**
 * This macro loops over all pixels in a grid of certain side length.
 *
 * @param a side length
 */
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


#endif // MACRO_H
