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
 *  @brief thunder_postprocess.cpp helps users to post-process the input image-file. The parameters provided by users are the directory of two parts input files, radius of mask, number of threads and pixelsize.
 *
 */

#include <iostream>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <iostream>

#include "Postprocess.h"

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_postprocess"

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

        fputs("Post-process the input image-file.\n", stdout);

        fputs("-j             set the thread-number to carry out work.\n", stdout);
        fputs("--mask         set the directory of mask file.\n", stdout);
        fputs("--inputA       set the directory of input file A.\n", stdout);
        fputs("--inputB       set the directory of input file B.\n", stdout);
        fputs("--pixelsize    set the pixelsize.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);

    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"inputA", required_argument, NULL, 'a'},
    {"inputB", required_argument, NULL, 'b'},
    {"mask", required_argument, NULL, 'm'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{

    int opt;
    char* inputA;
    char* inputB;
    char* mask;
    double pixelsize;
    int nThread;

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_FAILURE);
    }

    while((opt = getopt_long(argc, argv, "j:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('m'):
                mask = optarg;
                break;
            case('a'):
                inputA = optarg;
                break;
            case('b'):
                inputB = optarg;
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

    Postprocess pp(inputA,
                   inputB,
                   mask,
                   pixelsize);

    pp.run(nThread);

    TSFFTW_cleanup_threads();

    return 0;
}
