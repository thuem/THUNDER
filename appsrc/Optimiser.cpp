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

void readPara(MLOptimiserPara& dst,
              const Json::Value src)
{
    dst.k = src["Number of Classes"].asInt();
    dst.size = src["Size of Image"].asInt();
    dst.pixelSize = src["Pixel Size (Angstrom)"].asFloat();
    dst.maskRadius = src["Radius of Mask on Images (Angstrom)"].asFloat();
    dst.mS = src["Number of Sampling Points for Scanning in Global Search"].asInt();
    dst.transS = src["Estimated Translation (Pixel)"].asFloat();
    dst.initRes = src["Initial Resolution (Angstrom)"].asFloat();
    dst.globalSearchRes = src["Perform Global Search Under (Angstrom)"].asFloat();
    sprintf(dst.sym, src["Symmetry"].asString().c_str());
    sprintf(dst.initModel, src["Initial Model"].asString().c_str());
    sprintf(dst.db, src["Sqlite3 File Storing Paths and CTFs of Images"].asString().c_str());
    dst.autoSelection = src["Auto Selection"].asBool();

    dst.performMask = src["Reference Mask"]["Perform Reference Mask"].asBool();
    dst.autoMask = src["Reference Mask"]["Automask"].asBool();
    sprintf(dst.mask, src["Reference Mask"]["Provided Mask"].asString().c_str());

    dst.performSharpen = src["Sharpening"]["Perform Sharpening"].asBool();
    dst.estBFactor = src["Sharpening"]["Auto Estimate B-Factor"].asBool();
    dst.bFactor = src["Sharpening"]["B-Factor (Angstrom^2)"].asFloat();

    dst.iterMax = src["Advanced"]["Max Number of Iteration"].asInt();
    dst.pf = src["Advanced"]["Padding Factor"].asInt();
    dst.a = src["Advanced"]["MKB Kernel Radius"].asFloat();
    dst.alpha = src["Advanced"]["MKB Kernel Smooth Factor"].asFloat();
    dst.mG = src["Advanced"]["Number of Sampling Points in Global Search"].asInt();
    dst.mL = src["Advanced"]["Number of Sampling Points in Local Search"].asInt();
    dst.ignoreRes = src["Advanced"]["Ignore Signal Under (Angstrom)"].asFloat();
    dst.sclCorRes = src["Advanced"]["Correct Intensity Scale Using Signal Under (Angstrom)"].asFloat();
    dst.thresCutoffFSC = src["Advanced"]["FSC Threshold for Cutoff Frequency"].asFloat();
    dst.thresReportFSC = src["Advanced"]["FSC Threshold for Reporting Resolution"].asFloat();
    dst.groupSig = src["Advanced"]["Grouping when Calculating Sigma"].asBool();
    dst.groupScl = src["Advanced"]["Grouping when Correcting Intensity Scale"].asBool();
    dst.zeroMask = src["Advanced"]["Mask Images with Zero Noise"].asBool();
};

INITIALIZE_EASYLOGGINGPP

int main(int argc, char* argv[])
{
    loggerInit(argc, argv);

    Json::Reader reader;
    Json::Value root;

    ifstream in(argv[1], ios::binary);

    if (!in.is_open())
    {
        CLOG(FATAL, "LOGGER_SYS") << "Fail to Open Parameter File";

        __builtin_unreachable();
    }

    MLOptimiserPara para;

    if (reader.parse(in, root))
    {
        readPara(para, root);
    }
    else
    {
        CLOG(FATAL, "LOGGER_SYS") << "Fail to Parse Parameter File";

        __builtin_unreachable();
    }

    MPI_Init(&argc, &argv);

    cout << "Setting Parameters" << endl;
    MLOptimiser opt;
    opt.setPara(para);

    cout << "MPISetting" << endl;
    opt.setMPIEnv();

    cout << "Run" << endl;
    opt.run();

    MPI_Finalize();
}
