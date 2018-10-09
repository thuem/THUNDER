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

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_bfactor"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while (0)

#define HELP_OPTION_DESCRIPTION "--help     display this help\n"

void usage (int status)
{
    if (status != EXIT_SUCCESS)
        emit_try_help ();
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("Apply bfactor-filter to the input image-file.\n", stdout);

        fputs("-o    set the directory of output file.\n", stdout);
        fputs("-j    set the thread-number to carry out work.\n", stdout);
        fputs("--input    set the directory of input file.\n", stdout);
        fputs("--bFactor    set the bFactor.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);

    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"input", required_argument, NULL, 'i'},
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

    int option_index = 0;

    if (optind == argc)
    {
        usage(EXIT_FAILURE);
    }

    while((opt = getopt_long(argc, argv, "o:j:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('o'):
                output = optarg;
                break;
            case('i'):
                input = optarg;
                break;
            case('b'):
                bfactor = atof(optarg);
                break;
            case('p'):
                pixelsize = atof(optarg);
                break;
            case('j'):
                nThread = atoi(optarg);
                break;
            case('h'):
                usage(EXIT_SUCCESS);
                break;
            default:
                usage(EXIT_FAILURE);
        }

    }

    loggerInit(argc, argv);

    TSFFTW_init_threads();

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
