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
 *  @brief thunder_genmask.cpp generates a mask on input volume and outputs. The parameters provided by users are the directory of input and output files, thread-numbers, length of extending in pixel, threshold value, edgewidth of sphere and pixelsize.
 *
 */

#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <iostream>

#include "ImageFile.h"
#include "Volume.h"
#include "Mask.h"

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_genmask"

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
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("Generate a mask on input volume.\n", stdout);

        fputs("-o    set the directory of output file.\n", stdout);
        fputs("-j    set the thread-number to carry out work.\n", stdout);
        fputs("-ext    set the length of extending in pixel.\n", stdout);
        fputs("--input    set the directory of input file.\n", stdout);
        fputs("--threshold    set the threshold value.\n", stdout);
        fputs("--edgewidth    set the edge width of the sphere.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"input", required_argument, NULL, 'i'},
    {"threshold", required_argument, NULL, 't'},
    {"ext", required_argument, NULL, 'x'},
    {"edgewidth", required_argument, NULL, 'e'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* output;
    char* input;
    double threshold, ext, edgewidth, pixelsize;
    int nThread;

    int option_index = 0;

    if(optind == argc)
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
            case('t'):
                threshold = atof(optarg);
                break;
            case('x'):
                ext = atof(optarg);
                break;
            case('e'):
                edgewidth = atof(optarg);
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

    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    ImageFile imf(input, "rb");
    imf.readMetaData();

    Volume ref;
    imf.readVolume(ref);

    CLOG(INFO, "LOGGER_SYS") << "Removing Corners of the Map";

    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_RL(ref)
        if (QUAD_3(i, j, k) >= TSGSL_pow_2(ref.nColRL() / 2))
            ref.setRL(0, i, j, k);

    CLOG(INFO, "LOGGER_SYS") << "Generating Mask";

    Volume mask(ref.nColRL(),
                ref.nRowRL(),
                ref.nSlcRL(),
                RL_SPACE);

    genMask(mask,
            ref,
            threshold,
            ext,
            edgewidth,
            nThread);

    CLOG(INFO, "LOGGER_SYS") << "Writing Mask";

    imf.readMetaData(mask);

    imf.writeVolume(output, mask, pixelsize);
}
