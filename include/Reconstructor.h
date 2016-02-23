

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


        void insert(const Image& src,
                    const Coordinate5D coord,
                    const double u,
                    const double v);

        void reduceWM();

        void reduceWS();
        
        void broadcastWM();
        
        void broadcastWS();
        
        void reduceFM();
        
        void reduceFS();


};

#endif //RECONSTRUCTOR_H


