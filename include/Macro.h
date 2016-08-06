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
#define EDGE_WIDTH_RL 2 // edge width in mask

#define EQUAL_ACCURACY 0.0001

#define SAVE_DELETE(p) \
    if (p != NULL) { delete[] p; p = NULL; }

#endif // MACRO_H
