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
 *  @brief thunder_resize.cpp resizes the input image-file according to the value of boxsize. The parameters provided by users are directory of the input and output files, target boxsize to resize, number of threads to work and pixelsize.
 *
 */

#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <iostream>

#include <json/json.h>

#include "FFT.h"
#include "ImageFile.h"
#include "Volume.h"
#include "Utils.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_resize"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while(0)

void usage (int status)
{
    if (status != EXIT_SUCCESS)
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("\nResize the input image-file according to the value of boxsize.\n\n", stdout);

        fputs("-i  --input    set the filename of input file\n", stdout);
        fputs("-o  --output   set the filename of output file\n", stdout);
        fputs("--boxsize      set the target boxsize to resize\n", stdout);
        fputs("--pixelsize    set the pixelsize\n", stdout);
        fputs("-j             set the number of threads to carry out work\n", stdout);

        fputs("\n--help         display this help\n", stdout);
        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] =
{
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"boxsize", required_argument, NULL, 'b'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* output;
    char* input;
    double pixelsize;
    int boxsize, nThread;

    char option[5] = {'o', 'i', 'b', 'p', 'j'};

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_SUCCESS);
    }

    while((opt = getopt_long(argc, argv, "i:o:j:", long_options, &option_index)) != -1)
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
            case('b'):
                boxsize = atoi(optarg);
                option[2] = '\0';
                break;
            case('p'):
                pixelsize = atof(optarg);
                option[3] = '\0';
                break;
            case('j'):
                nThread = atoi(optarg);
                option[4] = '\0';
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

    FFT fft;
    fft.fw(src, nThread);

    int size = boxsize;

    Volume dst(size, size, size, FT_SPACE);

    if (src.nColRL() >= dst.nColRL())
    {
        #pragma omp parallel for num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_FT(dst)
            dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
    }
    else
    {
        #pragma omp parallel for num_threads(nThread)
        SET_0_FT(dst);

        #pragma omp parallel for num_threads(nThread)
        VOLUME_FOR_EACH_PIXEL_FT(src)
            dst.setFTHalf(src.getFTHalf(i, j, k), i, j, k);
    }

    fft.bw(dst, nThread);

    ImageFile imfDst;

    imfDst.readMetaData(dst);
    imfDst.writeVolume(output, dst, pixelsize);

    return 0;
}