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
 *  @brief thunder_average.cpp reads two input image-files and outputs the average file of their pixels' value. The parameters provided by users are the directory of output file and two parts input files and the pixelsize.
 *
 */

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>

#include "ImageFile.h"
#include "Volume.h"
#include "Utils.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_average"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while (0)

void usage(int status)
{
    if (status != EXIT_SUCCESS)
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("\nRead two input image-files and output the average file of their pixels' value.\n\n", stdout);

        fputs("--inputA       set the filename of input file A.\n", stdout);
        fputs("--inputB       set the filename of input file B.\n", stdout);
        fputs("-o  --output   set the filename of output file.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);

        fputs("\n--help         display this help\n", stdout);
        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] =
{
    {"inputA", required_argument, NULL, 'a'},
    {"inputB", required_argument, NULL, 'b'},
    {"output", required_argument, NULL, 'o'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* output;
    char* inputA;
    char* inputB;
    double pixelsize;

    char option[4] = {'o', 'a', 'b', 'p'};

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_SUCCESS);
    }

    while((opt = getopt_long(argc, argv, "o:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('o'):
                output = optarg;
                option[0] = '\0';
                break;
            case('a'):
                inputA = optarg;
                option[1] = '\0';
                break;
            case('b'):
                inputB = optarg;
                option[2] = '\0';
                break;
            case('p'):
                pixelsize = atof(optarg);
                option[3] = '\0';
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

    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    ImageFile imfA(inputA, "rb");
    imfA.readMetaData();

    Volume refA;
    imfA.readVolume(refA);

    ImageFile imfB(inputB, "rb");
    imfB.readMetaData();

    Volume refB;
    imfB.readVolume(refB);

    FOR_EACH_PIXEL_RL(refA)
    {
        refA(i) += refB(i);
        refA(i) /= 2;
    }

    ImageFile imf;
    imf.readMetaData(refA);
    imf.writeVolume(output, refA, pixelsize);

    return 0;
}