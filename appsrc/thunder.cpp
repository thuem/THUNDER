/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <fstream>

#include <json/json.h>

#include "Projector.h"
#include "Reconstructor.h"
#include "FFT.h"
#include "ImageFile.h"
#include "Particle.h"
#include "CTF.h"
#include "MLOptimiser.h"

using namespace std;

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

void readPara(MLOptimiserPara& dst,
              const Json::Value src)
{
    dst.nThreadsPerProcess = src["Number of Threads Per Process"].asInt();

    if (src["2D or 3D Mode"].asString() == "2D")
    {
        dst.mode = MODE_2D;
    }
    else if (src["2D or 3D Mode"].asString() == "3D")
    {
        dst.mode = MODE_3D;
    }
    else
        REPORT_ERROR("INEXISTENT MODE");

    dst.gSearch = src["Global Search"].asBool();
    dst.lSearch = src["Local Search"].asBool();
    dst.cSearch = src["CTF Search"].asBool();

    dst.k = src["Number of Classes"].asInt();
    dst.size = src["Size of Image"].asInt();
    dst.pixelSize = src["Pixel Size (Angstrom)"].asFloat();
    dst.maskRadius = src["Radius of Mask on Images (Angstrom)"].asFloat();
    dst.transS = src["Estimated Translation (Pixel)"].asFloat();
    dst.initRes = src["Initial Resolution (Angstrom)"].asFloat();
    dst.globalSearchRes = src["Perform Global Search Under (Angstrom)"].asFloat();
    copy_string(dst.sym, src["Symmetry"].asString());
    copy_string(dst.initModel, src["Initial Model"].asString());
    copy_string(dst.db, src[".thu File Storing Paths and CTFs of Images"].asString());
    copy_string(dst.parPrefix, src["Prefix of Particles"].asString());
    copy_string(dst.dstPrefix, src["Prefix of Destination"].asString());
    dst.coreFSC = src["Calculate FSC Using Core Region"].asBool();
    dst.maskFSC = src["Calculate FSC Using Masked Region"].asBool();
    dst.parGra = src["Particle Grading"].asBool();

    dst.performMask = src["Reference Mask"]["Perform Reference Mask"].asBool();
    dst.autoMask = src["Reference Mask"]["Automask"].asBool();
    copy_string(dst.mask, src["Reference Mask"]["Provided Mask"].asString());

    dst.iterMax = src["Advanced"]["Max Number of Iteration"].asInt();
    dst.goldenStandard = src["Advanced"]["Use Golden Standard FSC"].asBool();
    dst.pf = src["Advanced"]["Padding Factor"].asInt();
    dst.a = src["Advanced"]["MKB Kernel Radius"].asFloat();
    dst.alpha = src["Advanced"]["MKB Kernel Smooth Factor"].asFloat();
    dst.mS = src["Advanced"]["Number of Sampling Points for Scanning in Global Search"].asInt();
    dst.mGMax = src["Advanced"]["Number of Sampling Points in Global Search (Max)"].asInt();
    dst.mGMin = src["Advanced"]["Number of Sampling Points in Global Search (Min)"].asInt();
    dst.mLR = src["Advanced"]["Number of Sampling Points of Rotation in Local Search"].asInt();
    dst.mLT = src["Advanced"]["Number of Sampling Points of Translation in Local Search"].asInt();
    dst.mLD = src["Advanced"]["Number of Sampling Points of Defocus in Local Search"].asInt();
    dst.mReco = src["Advanced"]["Number of Sampling Points Used in Reconstruction"].asInt();
    dst.ignoreRes = src["Advanced"]["Ignore Signal Under (Angstrom)"].asFloat();
    dst.sclCorRes = src["Advanced"]["Correct Intensity Scale Using Signal Under (Angstrom)"].asFloat();
    dst.thresCutoffFSC = src["Advanced"]["FSC Threshold for Cutoff Frequency"].asFloat();
    dst.thresReportFSC = src["Advanced"]["FSC Threshold for Reporting Resolution"].asFloat();
    dst.thresSclCorFSC = src["Advanced"]["FSC Threshold for Scale Correction"].asFloat();
    dst.groupSig = src["Advanced"]["Grouping when Calculating Sigma"].asBool();
    dst.groupScl = src["Advanced"]["Grouping when Correcting Intensity Scale"].asBool();
    dst.zeroMask = src["Advanced"]["Mask Images with Zero Noise"].asBool();
    dst.ctfRefineS = src["Advanced"]["CTF Refine Standard Deviation"].asFloat();

    dst.transSearchFactor = src["Professional"]["Translation Search Factor"].asFloat();
    dst.perturbFactorL = src["Professional"]["Perturbation Factor (Large)"].asFloat();
    dst.perturbFactorSGlobal = src["Professional"]["Perturbation Factor (Small, Global)"].asFloat();
    dst.perturbFactorSLocal = src["Professional"]["Perturbation Factor (Small, Local)"].asFloat();
    dst.perturbFactorSCTF = src["Professional"]["Perturbation Factor (Small, CTF)"].asFloat();
    dst.skipE = src["Professional"]["Skip Expectation"].asBool();
    dst.skipM = src["Professional"]["Skip Maximization"].asBool();
    dst.skipR = src["Professional"]["Skip Reconstruction"].asBool();
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

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    MLOptimiserPara para;

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

    if (fftw_init_threads() == 0)
    {
        REPORT_ERROR("ERROR IN INITIALISING FFTW THREADS");

        abort();
    }

    CLOG(INFO, "LOGGER_SYS") << "Setting Time Limit for Creating FFTW Plan";
    fftw_set_timelimit(60);

    CLOG(INFO, "LOGGER_SYS") << "Setting Parameters";
    
    MLOptimiser opt;

    opt.setPara(para);

    CLOG(INFO, "LOGGER_SYS") << "Setting MPI Environment";

    opt.setMPIEnv();

    CLOG(INFO, "LOGGER_SYS") << "Running";

    opt.run();

    MPI_Finalize();

    fftw_cleanup_threads();

    return 0;
}
