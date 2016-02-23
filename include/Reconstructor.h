

#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H


#include <armadillo>
#include <utility>
#include <vector>


#include "Volume.h"
#include "Coordinate5D.h"

#include "Euler.h"

#include "Image.h"
#include "ImageFunctions.h"


using namespace arma;

using namespace std;


class Reconstructor
{

    
    private:
        int _nCol;
        int _nRow;
        int _nSlc;

        double _a = 2;
        double _alpha = 3.6;
        
        int _commRank = 0;
        int _commSize = 1;

        Volume _F;
        Volume _W;
        Volume _C;

        vector<pair<Coordinate5D, double>> _coordWeight;

    public:
        Reconstructor();

        Reconstructor(const int nCol,
                      const int nRow,
                      const int nSlc,
                      const double a,
                      const double alpha);

        Reconstructor(const Reconstructor& that);

        ~Reconstructor();

/////////////////////////////////////////////////////////////////////////////////////////


        void initial(const int nCol,
                     const int nRow,
                     const int nSlc,
                     const double a,
                     const double alpha);
        void setCommRank(int commRank);
        void setCommSize(int commSize);


        void insert(const Image& im,
                    const Coordinate5D coord,
                    const double u,
                    const double v);

        void reduceAllWeight_master();

        void reduceAllWeight_slave();
        
        void scatterAllWeight_master();
        
        void scatterAllWeight_slave();
        
        void reduceAllFTImage_master();
        
        void reduceAllFTImage_slave();



};

#endif //RECONSTRUCTOR_H


