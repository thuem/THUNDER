#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <unistd.h>
#include <vector>

#include "Macro.h"

#include "Image.h"
#include "ImageFile.h"
#include "ImageFunctions.h"

#include "Experiment.h"

#include "Parallel.h"

using namespace std;

typedef struct X_Y
{
    int x;
    int y;
} XY;

typedef struct PREPROCESS_PARA
{    
    int nCol;    

    int nRow;

    int xOffBias;

    int yOffBias;

    bool doNormalise;

    bool doInvertConstrast;

    double wDust;

    double bDust;

    double r;

    char db[FILE_NAME_LENGTH];

} PreprocessPara;

class Preprocess : public Parallel
{
    private:

        PreprocessPara _para;

        Experiment _exp;

        vector<int> _micIDs;

    public:        

        Preprocess();

        Preprocess(const PreprocessPara& para);

        PreprocessPara& getPara();

        void setPara(const PreprocessPara& para);

        void run();
        
    private:

        void removeOutOfBoundaryPar(const int micID);

        void getMicIDs(vector<int>& dst);

        void getMicName(char micName[],
                        const int micID);

        void getParXOffYOff(int& xOff,
                            int& yOff,
                            const int parID);

        void extractParticles(const int micID);
};

#endif // PREPROCESS_H
