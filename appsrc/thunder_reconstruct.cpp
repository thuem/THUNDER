#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>

#include "Model.h"
#include "FFT.h"
#include "Optimiser.h"
#include "Reconstructor.h"
#include "Logging.h"
#include "Config.h"
#include "Macro.h"

INITIALIZE_EASYLOGGINGPP

#define PROGRAM_NAME "thunder_reconstruct"

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

        fputs("Generate .\n", stdout);

        fputs("-j               set the thread-number per process to carry out work.\n", stdout);
        fputs("-i  --input      set the directory of input metadata file.\n", stdout);
        fputs("-o  --output     set the directory of output mrc.\n", stdout);
        fputs("--symmetry       set the symmetry.\n", stdout);
        fputs("--boxsize        set the boxsize of input image.\n", stdout);
        fputs("--pixelsize      set the pixel size.\n", stdout);

        fputs(HELP_OPTION_DESCRIPTION, stdout);

        fputs("Note: all parameters are indispensable.\n", stdout);
    }
    exit(status);
}

static const struct option long_options[] = 
{
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"symmetry", required_argument, NULL, 's'},
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
    char* symmetry;
    double pixelsize;
    int boxsize, nThread;

    int option_index = 0;

    if(optind == argc)
    {
        usage(EXIT_FAILURE);
    }

    while((opt = getopt_long(argc, argv, "i:o:j:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case('o'):
                output = optarg;
                break;
            case('i'):
                input = optarg;
                break;
            case('s'):
                symmetry = optarg;
                break;
            case('b'):
                boxsize = atoi(optarg);
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

    //get particle number and offset

    CLOG(INFO, "LOGGER_SYS") << "Determining Number of Particles and Offset of Each Particle in Metadata File";

    char line[FILE_LINE_LENGTH];
    vector<long> offset;
    int n = 0;

    FILE* file = fopen(input, "r");

    while (fgets(line, FILE_LINE_LENGTH - 1, file)) n++;

    offset.resize(n);

    rewind(file);

    for (int i = 0; i < (int)offset.size(); i++)
    {
        offset[i] = ftell(file);

        FGETS_ERROR_HANDLER(fgets(line, FILE_LINE_LENGTH - 1, file));
    }    

    CLOG(INFO, "LOGGER_SYS") << "Initialising MPI Environment";

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nImgPerProcess = n / (size - 1);

    n = nImgPerProcess * (size - 1);

    CLOG(INFO, "LOGGER_SYS") << n << " Images Will be Used in Reconstruction";

    CLOG(INFO, "LOGGER_SYS") << "Reading Metadata";

    FILE* dataBase = fopen(input, "r");

    char* word;
    string path;
    dvec4 quat;
    const int maxRadius = boxsize / 2;

    Image ctf(boxsize, boxsize, FT_SPACE);
    SET_1_FT(ctf);

    Symmetry sym(symmetry);

    Reconstructor recon(MODE_3D, boxsize, boxsize, 2, &sym, 1.9, 15);

    recon.setMPIEnv();
    recon.allocSpace(nThread);
    recon.setMaxRadius(maxRadius);

    Volume tarVol;
    
    CLOG(INFO, "LOGGER_SYS") << "Reading Particle Information";

    if (rank != 0)
    {
        FFT fft;
        Image img;
        dmat33 rot;

        for (int l = 0; l < nImgPerProcess; l++)
        {
            fseek(dataBase, offset[(rank - 1) * nImgPerProcess + l], SEEK_SET);
        
            FGETS_ERROR_HANDLER(fgets(line, FILE_LINE_LENGTH - 1, dataBase));

            word = strtok(line, " ");
            path = string(word);
            
            word = strtok(NULL, " ");
            quat(0) = atof(word);

            word = strtok(NULL, " ");
            quat(1) = atof(word);

            word = strtok(NULL, " ");
            quat(2) = atof(word);

            word = strtok(NULL, " ");    
            quat(3) = atof(word);

            int nslc = atoi(path.substr(0, path.find('@')).c_str());
            ImageFile imf(path.substr(path.find('@') + 1).c_str(), "rb");
            imf.readMetaData();
            imf.readImage(img, nslc);

            fft.fw(img, nThread);

            rotate3D(rot, quat);

            // CLOG(INFO, "LOGGER_SYS") << "inserting"  <<" THIS IS RANK"<< rank;
            recon.insert(img, ctf, rot, 1);
            
            fft.bw(img, nThread);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    CLOG(INFO, "LOGGER_SYS") << "Allreducing Reconstructor";

    recon.prepareTF(nThread);

    MPI_Barrier(MPI_COMM_WORLD);

    if ((rank == HEMI_A_LEAD) || (rank == HEMI_B_LEAD))
    {
        recon.setMAP(false);

        CLOG(INFO, "LOGGER_SYS") << "Reconstructing Reference";

        recon.reconstruct(tarVol, nThread);

        CLOG(INFO, "LOGGER_SYS") << "Saving Reference";
        ImageFile imf;
        imf.readMetaData(tarVol);

        char filename[FILE_NAME_LENGTH];

        if (rank == HEMI_A_LEAD)
        {
            snprintf(filename,
                     sizeof(filename),
                     "%s_A.mrc",
                     output);
        }
        else if (rank == HEMI_B_LEAD)
        {
            snprintf(filename,
                     sizeof(filename),
                     "%s_B.mrc",
                     output);
        }

       imf.writeVolume(filename, tarVol, pixelsize);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    recon.freeSpace();

    MPI_Finalize();

    return 0;

}
