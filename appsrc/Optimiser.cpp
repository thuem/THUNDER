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

#define PF 2

#define N 380
#define TRANS_S 20

#define PIXEL_SIZE 1.30654

#define MG 1000
#define ML 100

using namespace std;

void readPara(MLOptimiserPara& dst,
              const Json::Value src)
{
    dst.k = src["Number of Classes"].asInt();
    dst.size = src["Size of Image"].asInt();
    dst.pixelSize = src["PixelS Size (Angstrom)"].asFloat();
    dst.transS = src["Estimated Translation (Pixel)"].asFloat();
    dst.initRes = src["Initial Resolution (Angstrom)"].asFloat();
    dst.globalSearchRes = src["Perform Global Search Under (Angstrom)"].asFloat();
    sprintf(dst.sym, src["Symmetry"].asString().c_str());
    sprintf(dst.initModel, src["Initial Model"].asString().c_str());
    sprintf(dst.db, src["Sqlite3 File Storing Paths and CTFs of Images"].asString().c_str());

    dst.iterMax = src["Advanced"]["Max Number of Iteration"].asInt();
    dst.pf = src["Advanced"]["Padding Factor"].asInt();
    dst.a = src["Advanced"]["MKB Kernel Radius"].asFloat();
    dst.alpha = src["Advanced"]["MKB Kernel Smooth Factor"].asFloat();
    dst.mG = src["Advanced"]["Number of Sampling Points in Global Search"].asInt();
    dst.mL = src["Advanced"]["Number of Sampling Points in Local Search"].asInt();
    dst.ignoreRes = src["Advanced"]["Ignore Signal Under (Angstrom)"].asFloat();
    dst.sclCorRes = src["Advanced"]["Correct Intensity Scale Using Signal Under (Angstrom)"].asFloat();
    dst.groupSig = src["Advanced"]["Grouping when Calculating Sigma"].asBool();
    dst.groupScl = src["Advanced"]["Grouping when Correcting Intensity Scale"].asBool();
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

    /***
    cout << "Initialising Parameters" << endl;
    para.iterMax = atoi(argv[1]);
    para.k = 1;
    para.size = N;
    para.pf = PF;
    //para.a = 0.95;
    para.a = 1.9;
    para.alpha = 10;
    para.pixelSize = PIXEL_SIZE;
    para.mG = MG;
    para.mL = ML;
    para.transS = TRANS_S;
    para.initRes = 40;
    //para.ignoreRes = 200;
    para.ignoreRes = 200;
    para.sclCorRes = 80;
    //para.globalSearchRes = 15;
    para.globalSearchRes = 10;
    //para.globalSearchRes = 40;
    //para.globalSearchRes = 12;
    sprintf(para.sym, "C15");
    // sprintf(para.initModel, "padCylinder.mrc");
    sprintf(para.initModel, "padRef.mrc");
    sprintf(para.db, "C15.db");
    para.groupSig = true;
    para.groupScl = false;
    ***/

    MPI_Init(&argc, &argv);

    /***
    cout << "Setting Parameters" << endl;
    MLOptimiser opt;
    opt.setPara(para);

    cout << "MPISetting" << endl;
    opt.setMPIEnv();

    cout << "Run" << endl;
    opt.run();
    ***/

    MPI_Finalize();
}
