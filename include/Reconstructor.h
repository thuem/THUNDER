/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

//#define RECO_ZERO_MASK

#include <utility>
#include <mpi.h>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "omp_compat.h"

#include "Config.h"
#include "Macro.h"
#include "Typedef.h"
#include "Parallel.h"
#include "Coordinate5D.h"
#include "Functions.h"
#include "Euler.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"
#include "ImageFunctions.h"
#include "Symmetry.h"
#include "Transformation.h"
#include "TabFunction.h"
#include "Spectrum.h"
#include "Mask.h"

#define PRE_CAL_MODE 0

#define POST_CAL_MODE 1

#define PAD_SIZE (_pf * _size)

#define RECO_LOOSE_FACTOR 1

#define MIN_N_ITER_BALANCE 5

#define MAX_N_ITER_BALANCE 50

#define DIFF_C_THRES 0.01

#define DIFF_C_DECREASE_THRES 0.95

#define WIENER_FACTOR_MIN_R 5

/**
 * @ingroup Reconstructor
 * @brief The 3D model reconstruction class.
 * 
 * Provides all APIs that is used for reconstructing a 3D Fourier transform
 * of a 3D model in 3D Fourier space from the pixels data of 2D Fourie 
 * transform of real images and the associated 5D coordinates which are 
 * learned from sampling in last turn.
 * With the projection-slice theorem, the Fourier transform of the projection
 * of a 3D model onto a 2D plane is equal to a 2D slice of the 3D Fourier 
 * transform of that model consisting of a 2D plane through the origin in 
 * the Fourier space which is parallel to the projection plane. So each 
 * Fourier transform of the images should be a slice of 3d Fourier transform
 * of the protein in the 3D Fourier space. 
 * By inserting the Fourier transform of the images into a empty volume with 
 * correct 5D coordinate in 3D Fourie space, the 3D Fourier transform of the 
 * protein can be reconstructed as similar as the origin.
 * Since slices that are inserted into the 3D empty volume are all through 
 * the origin in 3D Fourier space, the value at site near the origin will be 
 * affected by much more slices than that far from the origin. It is necessary
 * to balance the value at different site in 3D Fourier space. By using 
 * weights, we reduce the scale of value at site near origin and add the scale
 * of value at site far from the origin. Simultaneously, normalization is done 
 * to make sure the value at every site is proper in 3D Fourie space.
 */

class Reconstructor : public Parallel
{
    private:

        int _mode;

        int _calMode;

        bool _MAP;

        bool _joinHalf;

        /**
         * The real size of the 3D Fourier reconstructor space that is used to
         * determine the size (PAD_SIZE) of Volume in 3 dimensions(xyz).
         * The result of reconstructor will be a (PAD_SIZE)^3 volume in 3D 
         * Fourier space.
         */
        int _size;

        Image _F2D;

        Image _W2D;

        Image _C2D;

        Image _T2D;

        /**
         * The 3D grid volume used to save the accumulation of the pixels 
         * values of 2D Fourier transforms of inserting images in 3D Fourier
         * space. Since the discretization in computation, the volume is a 
         * 3D grid. Pixels data of 2D Fourier transforms of real images is 
         * inserted into proper grid points of this volume by interpolation. 
         * After inserting in a single node, it will multiply the relative 
         * weights value of each grid point stored in @ref_W to get balanced.
         * Then an allreduce operation will be done to get the final sum of
         * _F volumes of all nodes, which the 3D Fourier transform of the 
         * model is obtained. This volume is initialised to be all zero.
         */
        Volume _F3D;
        
        /**
         * The 3D grid volume used to save the balancing weights factors of 
         * every 3D grid point in a single node. Since the discretization 
         * in computation, the volume is a 3D grid. Every inserting operation
         * will accumulate weights of associated points get by interpolation
         * of this volume into the grid point of accumulated weights volume 
         * @ref_C also by interpolation. After multiple rounds of divion by
         * relative accumulated weights value of each grid points in volume _C
         * that has been allreduced within all nodes, the weights are balanced
         * and normalized, with which can be multiply with volume @ref_F to 
         * get the 3D Fourier transform of the model. This volume initialised
         * to be all one.
         */
        Volume _W3D;
        
        
        /**
         * The 3D grid volume used to save the accumulation of weights of every
         * 3D grid point. Since the discretization in computation, the volume 
         * is a 3D grid. Every inserting operation will accumulate weights of
         * associated points get by interpolation of @ref _W grid into the 
         * grid point of this volume also by interpolation. An allreduce 
         * operation will be done to get the sum of accumulated weights value 
         * of each grid points of all nodes before balancing and normalization
         * by the divion of _W and _C, with which _C can be approximately equal
         * to 1. This volume initialised to be all zero.
         */
        Volume _C3D;

        Volume _T3D;

        /**
         * The vector to save the rotate matrixs of each insertion with image 
         * and associated 5D coordinate. Since each 2D Fourier transformsof 
         * images is a slice in the 3D Fourier transforms in a particular 
         * direction, matrixs to rotate the image 2D coordinate(x,y) associated
         * the 3rd coordinate z(always 0) to its real location in the 3D space 
         * can be obtained by the 5D coordinate of the image. Every inserting 
         * operation will also insert the rotate matrix into this vector. Note
         * that matrix can be the same because of the multiple possibility of
         * 5D coordinates in a single image. 
         */
        /***
        vector<mat33> _rot3D;

        vector<mat33> _rot2D;

        vector<const Image*> _ctf;
        ***/

        //vector<vec> _sig;

        //vector<const double*> _ctfP;

        int _nPxl;

        const int* _iCol;

        const int* _iRow;

        const int* _iPxl;

        const int* _iSig;
        
        /**
         * The vector to save the weight values of each insertion with image, 
         * associated 5D coordinate and weight. Since there are serveral 
         * different 5D coordinates that have the similar possibility for a 
         * single image, multiple 5D coordinates and weights of a single image
         * can be inserted. Every inserting operation will also insert the
         * weights into this vector.
         */
        vector<double> _w;
        
        /**
         * The max radius within the distance of which other point can be 
         * affected when one point is doing interpolation. 
         */
        int _maxRadius;

        /**
         * The padding factor which defined the PAD_SIZE (_pf * _size).
         * See @ref _size. By default, _pf = 2.
         */
        int _pf;

        /**
         * The symmetry mark of the model, which is used to reduce computation.
         */
        const Symmetry* _sym;

        vec _FSC;

        vec _sig;

        vec _tau;

        /**
         * The width of the Kernel. Parameter of modified Kaiser-Bessel Kernel.
         */
        double _a;

        /**
         * The smoothness parameter. Parameter of modified Kaiser-Bessel Kernel.
         */
        double _alpha;
        
        /**
         * the blob kernel stored as a tabular function
         */
        TabFunction _kernelFT;

        TabFunction _kernelRL;

        FFT _fft;

        void defaultInit()
        {
            _mode = MODE_3D;

            _calMode = POST_CAL_MODE;

            _MAP = true;

            _joinHalf = false;

            _pf = 2;
            _sym = NULL;
            _a = 1.9;
            _alpha = 15;

            _iCol = NULL;
            _iRow = NULL;
            _iPxl = NULL;
            _iSig = NULL;

            _FSC = vec::Constant(1, 1);
            _sig = vec::Zero(1);
            _tau = vec::Constant(1, 1);
        }

    public:

        /**
         * Construct a empty Reconstructor object
         */
        Reconstructor();

        /**
         * Construct a specific Reconstructor object with specific parameters.
         *
         * @param size  the size of real reconstructor space
         * @param pf    Padding factor. By default is 2. 
         * @param sym   Symmetry mark. By default is NULL.
         * @param a     The width of the modified Kaiser-Bessel Kernel. The
         *              default value is 1.9.
         * @param alpha The smoothness parameter of modified Kaiser-Bessel 
         *              Kernel. By default is 15.
         */
        Reconstructor(const int mode,
                      const int size,
                      const int pf = 2,
                      const Symmetry* sym = NULL,
                      const double a = 1.9,
                      const double alpha = 15);

        /**
         * default deconstructor
         */
        ~Reconstructor();

        /** 
         * initialize the member data of a reconstruct object. 
         *
         * @param size  The size of real reconstructor space.
         * @param pf    Padding factor. By default is 2. 
         * @param sym   Symmetry mark. By default is NULL.
         * @param a     The width of the modified Kaiser-Bessel Kernel. By 
         *              default is 0.95.
         * @param alpha The smoothness parameter of modified Kaiser-Bessel 
         *              kernel. By default is 15.
         */
        void init(const int mode,
                  const int size,
                  const int pf = 2,
                  const Symmetry* sym = NULL,
                  const double a = 1.9,
                  const double alpha = 15);

        void allocSpace();

        void resizeSpace(const int size);

        void reset();

        int mode() const;

        void setMode(const int mode);

        bool MAP() const;

        void setMAP(const bool MAP);

        bool joinHalf() const;

        void setJoinHalf(const bool joinHalf);

        /** 
         * set the symmetry mark of the model to be reconstructed. 
         *
         * @param sym Symmetry mark. Details see @ref Symmetry.
         */
        void setSymmetry(const Symmetry* sym);

        void setFSC(const vec& FSC);

        void setTau(const vec& tau);

        void setSig(const vec& sig);

        /**
         * get the max radius that points can affect each other 
         * during interpolation         
         */
        int maxRadius() const;

        /**
         * set the value of member data _maxRadius.
         *
         * @param maxRadius The max radius that points can affect each other 
         * during interpolation. 
         */
        void setMaxRadius(const int maxRadius);

        void preCal(int& nPxl,
                    const int* iCol,
                    const int* iRow,
                    const int* iPxl,
                    const int* iSig) const;

        void setPreCal(const int nPxl,
                       const int* iCol,
                       const int* iRow,
                       const int* iPxl,
                       const int* iSig);

        void insert(const Image& src,
                    const Image& ctf,
                    const mat22& rot,
                    const double w);

        /**
         * Insert a 2D Fourier transform of image pixel data with associated
         * 3D rotated matrix, 2D translation vector and weight into member 
         * data. The image data src will be accumulate into the relative grid
         * points of volume @ref_F by translation vecto, rotate matrix and 
         * interpolation. The rotate matrix will be recorded into vector 
         * @ref_rot. The weight values will be saved in vector @ref_w.
         *
         * @param src The image data that need to be inserted.
         * @param rot The rotate matrix to transform the image to its real
         * location based on the projection direction.
         * @param t The translation vector to move the origin of 2D Fourier
         * transform of the image to the image center.
         * @param w The weight values to measure the possibility of the rotate
         * matrix and translation vector.
         */
        void insert(const Image& src,
                    const Image& ctf,
                    const mat33& rot,
                    const double w);

        void insertP(const Image& src,
                     const Image& ctf,
                     const mat22& rot,
                     const double w);

        void insertP(const Image& src,
                     const Image& ctf,
                     const mat33& rot,
                     const double w);

        void reconstruct(Image& dst);

        /**
         * reconstruct a 3D model and save it into a volume.
         *
         * MODE_2D: the volume will be a special one with nSlc = 1
         * MODE_3D: the regular volume
         * 
         * @param dst The destination volume that reconstructor object saves the
         *            result of reconstruction into.
         */
        void reconstruct(Volume& dst);

    private:

        /**
         * The size of the reconstructor area that is used to determine the
         * size of Volume in 3 dimension xyz.
         */
        void allReduceF();

        void allReduceT();

        double checkC() const;

        void convoluteC();

        void symmetrizeF();

        void symmetrizeT();
};

#endif //RECONSTRUCTOR_H
