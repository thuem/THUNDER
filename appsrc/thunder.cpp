//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <fstream>

#include <json/json.h>

#include "Config.h"
#include "Logging.h"
#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "Optimiser.h"

using namespace std;

inline Json::Value JSONCPP_READ_ERROR_HANDLER(const Json::Value src)
{
    if (src == Json::nullValue)
    {
        REPORT_ERROR("INVALID JSON KEY");
        abort();
    }
    else
    {
        return src;
    }
}

/**
 *  This function is added by huabin
 *  This function is used to covert seconds to day:hour:min:sec format
 */

void fmt_time(int timeInSeconds, char *outputBuffer)
{
    int day = 0;
    int hour = 0;
    int min = 0;
    int sec = 0;
    int inputSeconds = timeInSeconds;

    day = timeInSeconds / (24 * 3600);
    timeInSeconds = timeInSeconds % (24 * 3600);
    hour = timeInSeconds/3600;
    timeInSeconds = timeInSeconds%3600;
    min = timeInSeconds/60;
    timeInSeconds = timeInSeconds%60;
    sec = timeInSeconds;
    snprintf(outputBuffer, 512, "%ds (%d days:%d hours:%d mins:%d seconds)\n", inputSeconds, day, hour, min, sec);

}

template <size_t N>
static inline void copy_string(char (&array)[N], const std::string& source)
{
    if (source.size() + 1 >= N)
    {
        CLOG(FATAL, "LOGGER_SYS") << "String too large to fit in parameter. "
                                  << "Destination length is "
                                  << N
                                  << ", while source length is "
                                  << source.size() + 1;
                                  
        return;
    }
    memcpy(array, source.c_str(), source.size() + 1);
}

void readPara(OptimiserPara& dst,
              const Json::Value src)
{
    dst.nThreadsPerProcess = JSONCPP_READ_ERROR_HANDLER(src["Number of Threads Per Process"]).asInt();

    if (JSONCPP_READ_ERROR_HANDLER(src["2D or 3D Mode"]).asString() == "2D")
    {
        dst.mode = MODE_2D;
    }
    else if (JSONCPP_READ_ERROR_HANDLER(src["2D or 3D Mode"]).asString() == "3D")
    {
        dst.mode = MODE_3D;
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    dst.gSearch = JSONCPP_READ_ERROR_HANDLER(src["Global Search"]).asBool();
    dst.lSearch = JSONCPP_READ_ERROR_HANDLER(src["Local Search"]).asBool();
    dst.cSearch = JSONCPP_READ_ERROR_HANDLER(src["CTF Search"]).asBool();

    dst.k = JSONCPP_READ_ERROR_HANDLER(src["Number of Classes"]).asInt();
    dst.size = JSONCPP_READ_ERROR_HANDLER(src["Size of Image"]).asInt();
    dst.pixelSize = JSONCPP_READ_ERROR_HANDLER(src["Pixel Size (Angstrom)"]).asFloat();
    dst.maskRadius = JSONCPP_READ_ERROR_HANDLER(src["Radius of Mask on Images (Angstrom)"]).asFloat();
    dst.transS = JSONCPP_READ_ERROR_HANDLER(src["Estimated Translation (Pixel)"]).asFloat();
    dst.initRes = JSONCPP_READ_ERROR_HANDLER(src["Initial Resolution (Angstrom)"]).asFloat();
    dst.globalSearchRes = JSONCPP_READ_ERROR_HANDLER(src["Perform Global Search Under (Angstrom)"]).asFloat();
    copy_string(dst.sym, JSONCPP_READ_ERROR_HANDLER(src["Symmetry"]).asString());
    copy_string(dst.initModel, JSONCPP_READ_ERROR_HANDLER(src["Initial Model"]).asString());
    copy_string(dst.db, JSONCPP_READ_ERROR_HANDLER(src[".thu File Storing Paths and CTFs of Images"]).asString());
    copy_string(dst.parPrefix, JSONCPP_READ_ERROR_HANDLER(src["Prefix of Particles"]).asString());
    copy_string(dst.dstPrefix, JSONCPP_READ_ERROR_HANDLER(src["Prefix of Destination"]).asString());
    dst.coreFSC = JSONCPP_READ_ERROR_HANDLER(src["Calculate FSC Using Core Region"]).asBool();
    dst.maskFSC = JSONCPP_READ_ERROR_HANDLER(src["Calculate FSC Using Masked Region"]).asBool();
    dst.parGra = JSONCPP_READ_ERROR_HANDLER(src["Particle Grading"]).asBool();

    dst.performMask = JSONCPP_READ_ERROR_HANDLER(src["Reference Mask"]["Perform Reference Mask"]).asBool();
    dst.globalMask = JSONCPP_READ_ERROR_HANDLER(src["Reference Mask"]["Perform Reference Mask During Global Search"]).asBool();
    copy_string(dst.mask, JSONCPP_READ_ERROR_HANDLER(src["Reference Mask"]["Provided Mask"]).asString());

    dst.iterMax = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Max Number of Iteration"]).asInt();
    dst.goldenStandard = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Using Golden Standard FSC"]).asBool();
    dst.pf = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Padding Factor"]).asInt();
    dst.a = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["MKB Kernel Radius"]).asFloat();
    dst.alpha = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["MKB Kernel Smooth Factor"]).asFloat();

    if (dst.mode == MODE_2D)
    {
        dst.mS = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points for Scanning in Global Search (2D)"]).asInt();

        dst.mLR = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points of Rotation in Local Search (2D)"]).asInt();
    }
    else if (dst.mode == MODE_3D)
    {
        dst.mS = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points for Scanning in Global Search (3D)"]).asInt();

        dst.mLR = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points of Rotation in Local Search (3D)"]).asInt();
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    dst.mLT = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points of Translation in Local Search"]).asInt();
    dst.mLD = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points of Defocus in Local Search"]).asInt();
    dst.mReco = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Number of Sampling Points Used in Reconstruction"]).asInt();
    dst.ignoreRes = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Ignore Signal Under (Angstrom)"]).asFloat();
    dst.sclCorRes = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Correct Intensity Scale Using Signal Under (Angstrom)"]).asFloat();
    dst.thresCutoffFSC = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["FSC Threshold for Cutoff Frequency"]).asFloat();
    dst.thresReportFSC = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["FSC Threshold for Reporting Resolution"]).asFloat();
    dst.thresSclCorFSC = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["FSC Threshold for Scale Correction"]).asFloat();
    dst.groupSig = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Grouping when Calculating Sigma"]).asBool();
    dst.groupScl = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Grouping when Correcting Intensity Scale"]).asBool();
    dst.zeroMask = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["Mask Images with Zero Noise"]).asBool();
    dst.ctfRefineS = JSONCPP_READ_ERROR_HANDLER(src["Advanced"]["CTF Refine Standard Deviation"]).asFloat();

    dst.transSearchFactor = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Translation Search Factor"]).asFloat();
    dst.perturbFactorL = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Perturbation Factor (Large)"]).asFloat();
    dst.perturbFactorSGlobal = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Perturbation Factor (Small, Global)"]).asFloat();
    dst.perturbFactorSLocal = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Perturbation Factor (Small, Local)"]).asFloat();
    dst.perturbFactorSCTF = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Perturbation Factor (Small, CTF)"]).asFloat();
    dst.skipE = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Skip Expectation"]).asBool();
    dst.skipM = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Skip Maximization"]).asBool();
    dst.skipR = JSONCPP_READ_ERROR_HANDLER(src["Professional"]["Skip Reconstruction"]).asBool();
};

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        cout << "Welcome to THUNDER! You may visit the website http://166.111.30.94/THUNDER for more information."
             << endl;

        return 0;
    }
    else if (argc != 2)
    {
        cout << "Wrong Number of Parameters Input!"
             << endl;

        return -1;
    }

    loggerInit(argc, argv);

    CLOG(INFO, "LOGGER_SYS") << "Initialising Processes";

    RFLOAT startTime = 0.0;
    RFLOAT endTime = 0.0;
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0)
    {
        startTime = MPI_Wtime();
    }

    CLOG(INFO, "LOGGER_SYS") << "Process " << rank << " Initialised";

    Json::Reader reader;
    Json::Value root;

    CLOG(INFO, "LOGGER_SYS") << "Initialising Threads Setting in FFTW";

    ifstream in(argv[1], ios::binary);

    CLOG(INFO, "LOGGER_SYS") << "Openning Parameter File";

    if (!in.is_open())
    {
        CLOG(FATAL, "LOGGER_SYS") << "Fail to Open Parameter File";

        abort();
    }

    OptimiserPara para;

    if (reader.parse(in, root))
    {
        readPara(para, root);
    }
    else
    {
        CLOG(FATAL, "LOGGER_SYS") << "Fail to Parse Parameter File";

        abort();
    }

    CLOG(INFO, "LOGGER_SYS") << "Setting Maximum Number of Threads Per Process";

    omp_set_num_threads(para.nThreadsPerProcess);

    CLOG(INFO, "LOGGER_SYS") << "Maximum Number of Threads in a Process is " << omp_get_max_threads();

    CLOG(INFO, "LOGGER_SYS") << "Initialising Threads Setting in FFTW";

    if (TSFFTW_init_threads() == 0)
    {
        REPORT_ERROR("ERROR IN INITIALISING FFTW THREADS");

        abort();
    }

    CLOG(INFO, "LOGGER_SYS") << "Setting Time Limit for Creating FFTW Plan";
    TSFFTW_set_timelimit(60);

    CLOG(INFO, "LOGGER_SYS") << "Setting Parameters";
    
    Optimiser opt;

    opt.setPara(para);

    CLOG(INFO, "LOGGER_SYS") << "Setting MPI Environment";

    opt.setMPIEnv();

    CLOG(INFO, "LOGGER_SYS") << "Running";

    opt.run();

    if(rank == 0)
    {
        endTime = MPI_Wtime();
        int totalSeconds = (int)(endTime - startTime);
        char timeBuffer[512];
        memset(timeBuffer, '\0', sizeof(timeBuffer));
        fmt_time(totalSeconds, timeBuffer);
        fprintf(stderr, "Elapse Time: %s\n", timeBuffer);
        
    }

    MPI_Finalize();

    TSFFTW_cleanup_threads();

    return 0;
}
