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
#include "Experiment.h"
#include "MLOptimiser.h"

using namespace std;

template <size_t N>
static inline void copy_string(char (&array)[N], const std::string& source)
{
    if (source.size() + 1 >= N)
    {
        CLOG(FATAL, "LOGGER_SYS") << "String too large to fit in parameter";
        return;
    }
    memcpy(array, source.c_str(), source.size()+1);
}

void readPara(MLOptimiserPara& dst,
              const Json::Value src)
{
    if (src["2D or 3D Mode"].asString() == "2D")
        dst.mode = MODE_2D;
    else if (src["2D or 3D Mode"].asString() == "3D")
        dst.mode = MODE_3D;
    else
        REPORT_ERROR("INEXISTENT MODE");

    dst.k = src["Number of Classes"].asInt();
    dst.size = src["Size of Image"].asInt();
    dst.pixelSize = src["Pixel Size (Angstrom)"].asFloat();
    dst.maskRadius = src["Radius of Mask on Images (Angstrom)"].asFloat();
    dst.transS = src["Estimated Translation (Pixel)"].asFloat();
    dst.initRes = src["Initial Resolution (Angstrom)"].asFloat();
    dst.globalSearchRes = src["Perform Global Search Under (Angstrom)"].asFloat();
    copy_string(dst.sym, src["Symmetry"].asString());
    copy_string(dst.initModel, src["Initial Model"].asString());
    copy_string(dst.db, src["Sqlite3 File Storing Paths and CTFs of Images"].asString());
    dst.autoSelection = src["Auto Selection"].asBool();
    dst.localCTF = src["Local CTF"].asBool();

    dst.performMask = src["Reference Mask"]["Perform Reference Mask"].asBool();
    dst.autoMask = src["Reference Mask"]["Automask"].asBool();
    copy_string(dst.mask, src["Reference Mask"]["Provided Mask"].asString());

    dst.performSharpen = src["Sharpening"]["Perform Sharpening"].asBool();
    dst.estBFactor = src["Sharpening"]["Auto Estimate B-Factor"].asBool();
    dst.bFactor = src["Sharpening"]["B-Factor (Angstrom^2)"].asFloat();

    dst.iterMax = src["Advanced"]["Max Number of Iteration"].asInt();
    dst.pf = src["Advanced"]["Padding Factor"].asInt();
    dst.a = src["Advanced"]["MKB Kernel Radius"].asFloat();
    dst.alpha = src["Advanced"]["MKB Kernel Smooth Factor"].asFloat();
    dst.mS = src["Advanced"]["Number of Sampling Points for Scanning in Global Search"].asInt();
    dst.mG = src["Advanced"]["Number of Sampling Points in Global Search"].asInt();
    dst.mL = src["Advanced"]["Number of Sampling Points in Local Search"].asInt();
    dst.mReco = src["Advanced"]["Number of Sampling Points Used in Reconstruction"].asInt();
    dst.ignoreRes = src["Advanced"]["Ignore Signal Under (Angstrom)"].asFloat();
    dst.sclCorRes = src["Advanced"]["Correct Intensity Scale Using Signal Under (Angstrom)"].asFloat();
    dst.thresCutoffFSC = src["Advanced"]["FSC Threshold for Cutoff Frequency"].asFloat();
    dst.thresReportFSC = src["Advanced"]["FSC Threshold for Reporting Resolution"].asFloat();
    dst.thresSclCorFSC = src["Advanced"]["FSC Threshold for Scale Correction"].asFloat();
    dst.groupSig = src["Advanced"]["Grouping when Calculating Sigma"].asBool();
    dst.groupScl = src["Advanced"]["Grouping when Correcting Intensity Scale"].asBool();
    dst.zeroMask = src["Advanced"]["Mask Images with Zero Noise"].asBool();
};

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    loggerInit(argc, argv);

    fftw_init_threads();

    Json::Reader reader;
    Json::Value root;

    ifstream in(argv[1], ios::binary);

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

    std::cout << "Setting Parameters" << std::endl;
    MLOptimiser opt;
    opt.setPara(para);

    std::cout << "MPISetting" << std::endl;
    opt.setMPIEnv();

    std::cout << "Run" << std::endl;
    opt.run();

    MPI_Finalize();

    fftw_cleanup_threads();

    return 0;
}
