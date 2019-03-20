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

#define NUMBER_VIEW_PER_DIMENSION 5

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_view"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while (0)

#define HELP_OPTION_DESCRIPTION "--help     display this help\n"

void usage(int status)
{
    if (status != EXIT_SUCCESS)
    {
        emit_try_help ();
    }
    else
    {
        printf("Usage: %s [OPTION]...\n", PROGRAM_NAME);

        fputs("Read two input image-files and output the average file of their pixels' value.\n", stdout);

        fputs("-o  --output   set the directory of output file.\n", stdout);
        fputs("-i  --input    set the directory of input file B.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);

    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{
    
    int opt;
    char* output;
    char* input;

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_FAILURE);
    }

    while((opt = getopt_long(argc, argv, "o:i:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('o'):
                output = optarg;
                break;
            case('i'):
                input = optarg;
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

    int boxsize = ref.nColRL();

    Image view(boxsize * NUMBER_VIEW_PER_DIMENSION, boxsize * 3, RL_SPACE);

    Image panel(boxsize, boxsize, RL_SPACE);

    for (int vj = 0; vj < 3; vj++)
        for (int vi = 0; vi < NUMBER_VIEW_PER_DIMENSION; vi++)
        {
            int oi = vi * boxsize - view.nColRL() / 2 + boxsize / 2;
            int oj = vj * boxsize - view.nRowRL() / 2 + boxsize / 2;

            IMAGE_FOR_EACH_PIXEL_RL(panel)
            {
                if (vj == 0)
                {
                    panel.setRL(ref.getRL(i, j, boxsize / (NUMBER_VIEW_PER_DIMENSION + 1) * (vi + 1) - boxsize / 2), i, j);
                }
                
                if (vj == 1)
                {
                    panel.setRL(ref.getRL(i, boxsize / (NUMBER_VIEW_PER_DIMENSION + 1) * (vi + 1) - boxsize / 2, j), i, j);
                }

                if (vj == 2)
                {
                    panel.setRL(ref.getRL(boxsize / (NUMBER_VIEW_PER_DIMENSION + 1) * (vi + 1) - boxsize / 2, i, j), i, j);
                }
            }

            IMAGE_FOR_EACH_PIXEL_RL(panel)
            {
                view.setRL(panel.getRL(i, j), oi + i, oj + j);
            }
        }

    view.saveRLToBMP(output);

    return 0;
}
