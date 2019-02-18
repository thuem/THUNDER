/** @file
 *  @author Mingxu Hu
 *  @author Shouqing Li
 *  @version 1.4.11.080928
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu   Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Shouqing Li | 2018/09/28 | 1.4.11.080928 | add options
 *  Shouqing Li | 2018/01/07 | 1.4.11.090107 | output error information of missing options
 *
 *  @brief thunder_alignZ.cpp realizes the function of aligning the direction vector of a input file to Z-axis and outputting the results. The parameters provided by users are the directory of input and output file, the coordinate of X,Y,Z, the number of threads and the pixelsize.
 *
 */

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>

#include <json/json.h>

#include "FFT.h"
#include "ImageFile.h"
#include "Volume.h"
#include "Euler.h"
#include "Transformation.h"
#include "Utils.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_alignZ"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
        PROGRAM_NAME); \
    } \
while (0)

void usage (int status)
{
    if (status != EXIT_SUCCESS)
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("\nAlign the directory vector of a input file to Z-axis.\n\n",stdout);

        fputs("-i  --input    set the filename of input file.\n", stdout);
        fputs("-o  --output   set the filename of output file.\n", stdout);
        fputs("-x             set the coordinate of X.\n", stdout);
        fputs("-y             set the coordinate of Y.\n", stdout);
        fputs("-z             set the coordinate of Z.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);
        fputs("-j             set the number of threads to carry out work.\n", stdout);

        fputs("\n--help         display this help\n", stdout);
        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

const struct option long_options[] =
{
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* output;
    char* input;
    double x, y, z, pixelsize;
    int nThread;

    char option[7] = {'o', 'i', 'x', 'y', 'z', 'p', 'j'};

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_SUCCESS);
    }

    while((opt = getopt_long(argc, argv, "i:o:x:y:z:j:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('o'):
                output = optarg;
                option[0] = '\0';
                break;
            case('i'):
                input = optarg;
                option[1] = '\0';
                break;
            case('x'):
                x = atof(optarg);
                option[2] = '\0';
                break;
            case('y'):
                y = atof(optarg);
                option[3] = '\0';
                break;
            case('z'):
                z = atof(optarg);
                option[4] = '\0';
                break;
            case('p'):
                pixelsize = atof(optarg);
                option[5] = '\0';
                break;
            case('j'):
                nThread = atoi(optarg);
                option[6] = '\0';
                break;
            case('h'):
                usage(EXIT_SUCCESS);
                break;
            default:
                usage(EXIT_FAILURE);
        }

    }

    optionCheck(option, sizeof(option) / sizeof(*option), long_options);

    loggerInit(argc, argv);

    omp_set_nested(false);

    ImageFile imfSrc(input, "rb");

    Volume src;

    imfSrc.readMetaData();
    imfSrc.readVolume(src);

    dvec3 v(x, y, z);
    v << x, y, z;

    dmat33 rot;
    alignZ(rot, v);

    Volume dst(src.nColRL(), src.nRowRL(), src.nSlcRL(), RL_SPACE);

    VOL_TRANSFORM_MAT_RL(dst, src, rot, src.nColRL() / 2 - 1, LINEAR_INTERP, nThread);

    ImageFile imfDst;

    imfDst.readMetaData(dst);
    imfDst.writeVolume(output, dst, pixelsize);

    return 0;
}