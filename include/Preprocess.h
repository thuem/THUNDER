/*******************************************************************************
 * Author: Mingxu Hu, Bing Li
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>

#include "Macro.h"
#include "Typedef.h"
#include "Logging.h"

#include "Image.h"
#include "ImageFile.h"
#include "ImageFunctions.h"

#include "Experiment.h"

#include "Parallel.h"



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

        PreprocessPara& para();

        void setPara(const PreprocessPara& para);

        void init();

        void run();
        
    private:

        void getMicIDs(vector<int>& dst);

        std::string getMicName(const int micID);

        void getParXOffYOff(int& xOff,
                            int& yOff,
                            const int parID);

        void extractParticles(const int micID);
};

#endif // PREPROCESS_H
