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
 *  @brief thunder_bfactor.cpp applies bfactor-filter to the input image-file and outputs. The parameters provided by users are the b-factor, the directory of input and output files, the number of threads to work and the pixelsize.
 *
 */

#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <iostream>

#include "ImageFile.h"
#include "Volume.h"
#include "Filter.h"
#include "FFT.h"
#include "Utils.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_bfactor"

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
        emit_try_help ();
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("\nApply bfactor-filter to the input image-file.\n\n", stdout);

        fputs("-i  --input    set the filename of input file.\n", stdout);
        fputs("-o  --output   set the filename of output file.\n", stdout);
        fputs("--bFactor      set the bFactor.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);
        fputs("-j             set the number of threads to carry out work.\n", stdout);

        fputs("\n--help         display this help\n", stdout);
        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] =
{
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"bfactor", required_argument, NULL, 'b'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* output;
    char* input;
    double bfactor, pixelsize;
    int nThread;

    char option[5] = {'o', 'i', 'b', 'p', 'j'};

    int option_index = 0;

    if (optind == argc)
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
                bfactor = atof(optarg);
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

    TSFFTW_init_threads();

    omp_set_nested(false);

    ImageFile imf(input, "rb");
    imf.readMetaData();

    Volume ref;
    imf.readVolume(ref);

    FFT fft;
    fft.fw(ref, nThread);

    bFactorFilter(ref,
                  ref,
                  bfactor,
                  nThread);

    fft.bw(ref, nThread);

    imf.readMetaData(ref);

    imf.writeVolume(output, ref, pixelsize);

    TSFFTW_cleanup_threads();

    return 0;
}