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
 *  @brief thunder_genmask_shell.cpp generates a shell-shape mask on the input volume and outputs. The parameters provided by users are directory of output file, the inner-radius and outer-radius, edgewidth of sphere, boxsize and pixelsize.
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

#define PROGRAM_NAME "thunder_genmask_shell"

#define emit_try_help() \
do \
    { \
        fprintf(stderr, "Try '%s --help' for more information.\n", \
                PROGRAM_NAME); \
    } \
while(0)

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

        fputs("Generate a shell-shape mask on the input volume.\n", stdout);

        fputs("-j                set the thread-number to carry out work.\n", stdout);
        fputs("-o  --output      set the directory of output file.\n", stdout);
        fputs("--boxsize         set the boxsize value.\n", stdout);
        fputs("--edgewidth       set the edge width of the sphere.\n", stdout);
        fputs("--pixelsize       set the pixelsize.\n", stdout);
        fputs("--inner_radius    set the length of inner_radius.\n", stdout);
        fputs("--outer_radius    set the length of outer_radius.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);

    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"boxsize", required_argument, NULL, 'b'},
    {"output", required_argument, NULL, 'o'},
    {"inner_radius", required_argument, NULL, 'n'},
    {"outer_radius", required_argument, NULL, 't'},
    {"edgewidth", required_argument, NULL, 'e'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* output;
    double inner_radius, outer_radius, edgewidth, pixelsize;
    int boxsize, nThread;

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
            case('n'):
                inner_radius = atof(optarg);
                break;
            case('t'):
                outer_radius = atof(optarg);
                break;
            case('b'):
                boxsize = atoi(optarg);
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

    ImageFile imf;

    //ImageFile imf(argv[2], "rb");
    //imf.readMetaData();

    /***
    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    Volume ref;
    imf.readVolume(ref);

    CLOG(INFO, "LOGGER_SYS") << "Removing Corners of the Map";

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(ref)
        if (QUAD_3(i, j, k) >= TSGSL_pow_2(ref.nColRL() / 2))
            ref.setRL(0, i, j, k);
    ***/

    CLOG(INFO, "LOGGER_SYS") << "Generating Mask";

    Volume mask(boxsize,
                boxsize,
                boxsize,
                RL_SPACE);

    RFLOAT ew = edgewidth;

    #pragma omp parallel for num_threads(nThread)
    VOLUME_FOR_EACH_PIXEL_RL(mask)
    {
        RFLOAT d = NORM_3(i, j, k);

        RFLOAT inner = inner_radius / pixelsize;
        RFLOAT outer = outer_radius / pixelsize;

        if (d < inner - ew)
        {
            mask.setRL(0, i, j, k);
        }
        else if (d < inner)
        {
            mask.setRL(cos((d - inner) / ew) + 0.5, i, j, k);
        }
        else if (d < outer)
        {
            mask.setRL(1, i, j, k);
        }
        else if (d < outer + ew)
        {
            mask.setRL(cos((d - outer) / ew) + 0.5, i, j, k);
        }
        else
        {
            mask.setRL(0, i, j, k);
        }
    }

    //softEdge(mask, atof(argv[5]));

    /***
    genMask(mask,
            ref,
            atof(argv[3]),
            atof(argv[4]),
            atof(argv[5]));
    ***/

    CLOG(INFO, "LOGGER_SYS") << "Writing Mask";

    imf.readMetaData(mask);

    imf.writeVolume(output, mask, pixelsize);
}
