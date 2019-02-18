/** @file
 *  @author Mingxu Hu
 *  @version 1.4.11.081226
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2018/12/26 | 1.4.11.081226 | new file
 *  Shouqing Li | 2018/01/07 | 1.4.11.090107 | output error information of missing options
 *
 */

#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>

#include <json/json.h>

#include "Config.h"
#include "Logging.h"
#include "Macro.h"
#include "Projector.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "Optimiser.h"
#include "Utils.h"

using namespace std;

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_project"

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

        fputs("\nCalculate a model's projections according to the parameters number.\n\n", stdout);

        fputs("-i  --input      set the filename of input reference.\n", stdout);
        fputs("-o  --output     set the filename of output stack.\n", stdout);
        fputs("--metadata       set the filename of outputting metadata.\n", stdout);
        fputs("-n               set the number of projections.\n", stdout);
        fputs("--pixelsize      set the pixel size.\n", stdout);
        fputs("-j               set the number of threads per process to carry out work.\n", stdout);

        fputs("\n--help           display this help\n", stdout);
        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] =
{
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"metadata", required_argument, NULL, 'm'},
    {"pixelsize", required_argument, NULL, 'p'},
    {"help", no_argument, NULL, 'h'},
    {NULL, 0, NULL, 0}
};

int main(int argc, char* argv[])
{
    int opt;
    char* output;
    char* input;
    char* metadata;
    int n;
    double pixelsize;
    int nThread;

    char option[6] = {'o', 'i', 'n', 'm', 'p', 'j'};

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_SUCCESS);
    }

    while((opt = getopt_long(argc, argv, "i:o:j:n:", long_options, &option_index)) != -1)
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
            case('n'):
                n = atoi(optarg);
                option[2] = '\0';
                break;
            case('p'):
                pixelsize = atof(optarg);
                option[3] = '\0';
                break;
            case('m'):
                metadata = optarg;
                option[4] = '\0';
                break;
            case('j'):
                nThread = atoi(optarg);
                option[5] = '\0';
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

    // rotations
    dmat4 quat(n, 4);
    sampleACG(quat, 1, 1, 1, n);

    CLOG(INFO, "LOGGER_SYS") << "Reading Map";

    ImageFile imf(input, "rb");
    imf.readMetaData();

    Volume ref;
    imf.readVolume(ref);

    FFT fft;
    fft.fw(ref, nThread);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = n / size * size;
    CLOG(INFO, "LOGGER_SYS") << "Total of " << n << " Projections Will Be Output.";

    // boxsize
    int N = ref.nColRL();

    CLOG(INFO, "LOGGER_SYS") << "Setting Projector";

    Projector proj;
    proj.setProjectee(ref.copyVolume(), nThread);

    FFT ffwImg;

    CLOG(INFO, "LOGGER_SYS") << "Openning Stack";

    char filename[FILE_NAME_LENGTH];
    sprintf(filename, "%s_Rank_%06d.mrcs", output, rank);

    imf.openStack(filename, N, n / size, pixelsize);

    for (int l = 0; l < n / size; l++)
    {
        int m = l + rank * (n / size);

        dmat33 mat;

        rotate3D(mat, quat.row(m).transpose());

        // randRotate3D(mat);

        Image img(N, N, FT_SPACE);

        SET_0_FT(img);

        proj.project(img, mat, nThread);

        fft.bw(img, nThread);

        imf.writeStack(img, l);
    }

    CLOG(INFO, "LOGGER_SYS") << "Closing Stack";

    imf.closeStack();

    bool flag;
    MPI_Status status;

    if (rank != 0)
        MPI_Recv(&flag, 1, MPI_C_BOOL, rank - 1, 0, MPI_COMM_WORLD, &status);

    FILE* file = (rank == 0) ? fopen(metadata, "w") : fopen(metadata, "a");

    CLOG(INFO, "LOGGER_SYS") << "Recording Metadata";

    for (int l = 0; l < n / size; l++)
    {
        int m = l + rank * (n / size);

        fprintf(file,
                "%012ld@%s_Rank_%06d.mrcs %18.9lf %18.9lf %18.9lf %18.9lf\n",
                 l,
                 output,
                 rank,
                 quat(m, 0),
                 quat(m, 1),
                 quat(m, 2),
                 quat(m, 3));
    }

    fclose(file);

    if (rank != size - 1)
        MPI_Send(&flag, 1, MPI_C_BOOL, rank + 1, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
