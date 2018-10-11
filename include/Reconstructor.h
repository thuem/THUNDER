/** @file
 *  @author Hongkun Yu
 *  @author Mingxu Hu
 *  @version 1.4.11.180926
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR     | TIME       | VERSION       | DESCRIPTION
 *  ------     | ----       | -------       | -----------
 *  Hongkun Yu | 2015/03/23 | 0.0.1.050323  | new file
 *  He Zhao    | 2018/09/26 | 1.4.11.180926 | add notes & header
 *
 *  @brief Reconstructor.h contains a 2D and 3D model reconstruction class. For 2D reconstruction, it provides all functions that are used for reconstructing a 2D Fourier transform of a 2D modal in 2D Fourier space from the pixel data of 2D Fourier transform of real images and the associated 5D coordinates which are learned from sampling in the former round. For 3D reconstruction, it provides all functions that are used for reconstructing a 3D Fourier transform of a 3D model in 3D Fourier space from the pixel data of 2D Fourier transform of real images and the associated 5D coordinates which are learned from sampling in the former round. Apart from the ordinary CPU-version functions, it also provides several GPU-version insert functions.
 */


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
#include "Precision.h"
#include "Parallel.h"
#include "Coordinate5D.h"
#include "Functions.h"
#include "Euler.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"
#include "Particle.h"
#include "ImageFunctions.h"
#include "Symmetry.h"
#include "Transformation.h"
#include "TabFunction.h"
#include "Spectrum.h"
#include "Mask.h"

#ifdef GPU_VERSION
#include "Interface.h"
#endif

#define PRE_CAL_MODE 0

#define POST_CAL_MODE 1

#define PAD_SIZE (_pf * _size)

#define RECO_LOOSE_FACTOR 1

#define MIN_N_ITER_BALANCE 10

#define MAX_N_ITER_BALANCE 30

#define DIFF_C_THRES 1e-2

#define DIFF_C_DECREASE_THRES 0.95

#define N_DIFF_C_NO_DECREASE 2

#define WIENER_FACTOR_MIN_R 5

#define FSC_BASE_L 1e-3

#define FSC_BASE_H (1 - 1e-3)

/**
 * @brief The 2D and 3D model reconstruction class. 
 * It provides all APIs that are used for reconstructing a 2D/3D Fourier transform of a 2D/3D model in 2D/3D Fourier space from the pixel data of 2D Fourier transform of real images and the associated 5D coordinates which are learned from sampling in the former round. 
 * According to the central section theorem, the Fourier transform of a specific projection of a 3D model onto a 2D plane is equal to the 2D slice of the 3D Fourier transform of that model consisting of a 2D plane through the origin in the Fourier space which is parallel to the projection plane. So each Fourier transform of the images should be a slice of 3D Fourier transform of the protein in the 3D Fourier space. 
 * By inserting the Fourier transform of the images into an empty volume with correct 5D coordinates in 3D Fourier space, the 3D Fourier transform of the protein can be reconstructed as similar as the original one. 
 * Since slices inserted into the 3D empty volume all go through the origin in 3D Fourier space, the value at site near the origin will be affected by comparatively more slices than that far from the origin. It is necessary to balance the value at different site in 3D Fourier space. Using weights, we reduce the scale of value at site near origin and increase the scale of value at site far from the origin. 
 * Simultaneously, normalization is done to make sure the value at every site is proper in 3D Fourier space.
 */

class Reconstructor : public Parallel
{
    private:

        /**
         * @brief the indicator that the constructor operates under MODE_2D(the volume will be a special one with nSlc = 1) or MODE_3D(the regular volume)
         */
        int _mode;

        /**
         * @brief the indicator that the constructor operates under PRE_CAL_MODE or POST_CAL_MODE
         */
        int _calMode;

        /**
         * @brief the indicator of whether to use MAP method in the reconstruction or not
         */
        bool _MAP;

        /**
         * @brief the indicator of whether to correct grid or not
         */
        bool _gridCorr;

        /**
         * @brief the indicator of whether to join the two halves or not
         */
        bool _joinHalf;

        /**
         * @brief the size (PAD_SIZE) of Volume in 3 dimensions(xyz)
         */
        int _size;

        /**
         * @brief the real size of the 3D Fourier reconstructor space that is used to determine the size (PAD_SIZE) of Volume in 3 dimensions(xyz). The result of reconstructor will be a @f$\left({PAD\_SIZE}\right)^{3}@f$ volume in 3D Fourier space.
         */
        int _N;

        /**
         * @brief the 2D grid image used to save the accumulation of the pixel values of 2D Fourier transforms of inserting images in 2D Fourier space. 
         */
        Image _F2D;

        /**
         * @brief the 2D grid image used to save the balancing weighting factors of every 2D grid point
         */
        Image _W2D;

        /**
         * @brief the 2D grid image helpful to calculate _W in grid correction operation
         */
        Image _C2D;

        /**
         * @brief the 2D grid image used to save the accumulation of weights of every 2D grid point
         */
        Image _T2D;

        /**
         * @brief the 3D grid volume used to save the accumulation of the pixel values of 2D Fourier transforms of inserting images in 3D Fourier space. 
         * Since the discretization in computation, the volume is a 3D grid. Pixel data of 2D Fourier transforms of real images is inserted into proper grid points of this volume by interpolation. After inserting in a single process, it will multiply the relative weights' value of each grid point stored in _W to get balanced. Then an allreduce operation will be done to get the final summation of _F volumes of all processes, which the 3D Fourier transform of the model is obtained. 
         * This volume is initialised to be all zeros.
         */
        Volume _F3D;
        
        /**
         * @brief the 3D grid volume used to save the balancing weighting factors of every 3D grid point. 
         * Since the discretization in computation, the volume is a 3D grid. Every inserting operation will accumulate weights of associated points got by interpolation of this volume into the grid point of accumulated weights volume _C also by interpolation. After multiple rounds of division by relative accumulated weights of each grid points in volume _C that has been allreduced within all processes, the weights are balanced and normalized, with which can be multiplied with volume _F to get the 3D Fourier transform of the model. 
         * This volume is initialised to be all ones.
         */
        Volume _W3D;
        
        
        /**
         * @brief the 3D grid volume helpful to calculate _W in grid correction operation
         */
        Volume _C3D;

        /**
         * @brief the 3D grid volume used to save the accumulation of weights of every 3D grid point. 
         * Since the discretization in computation, the volume is a 3D grid. Every inserting operation will accumulate weights of associated points, which is gotten by interpolation of _W grid into the grid point of this volume also by interpolation. An allreduce operation will be done to get the summation of accumulated weights of each grid points of all processes before balancing and normalizing by the division of _W and _C, with which _C can be approximately equal to 1. 
         * This volume is initialised to be all zeros.
         */
        Volume _T3D;

        /**
         * the vector to save the rotation matrices of each insertion with image and associated 5D coordinates. 
         * Since 2D Fourier transform of each image is a slice extracted from a particular direction in the 3D Fourier transform domain, rotation matrices that project the image's 2D coordinate(x,y), associated the third coordinate z always being 0, onto its real location in the 3D space can be obtained by the 5D coordinates of the image. Every inserting operation will also insert the rotation matrix into this vector. 
         * Note that matrix can be the same because of the multiple possibility of 5D coordinates in a single image. 
         */
        /***
        vector<dmat33> _rot3D;

        vector<dmat33> _rot2D;

        vector<const Image*> _ctf;
        ***/

        //vector<vec> _sig;

        //vector<const RFLOAT*> _ctfP;

        /**
        * @brief the number of the pixels
        */
        int _nPxl;

        /**
         * @brief the pointer that points to the memory blocks that save the column index of the pixels
         */
        const int* _iCol;

        /**
         * @brief the pointer that points to the memory blocks that save the row index of the pixels
         */
        const int* _iRow;

        /**
         * @brief the pointer that points to the memory blocks that save the indices of the pixels
         */
        const int* _iPxl;

        /**
         * @brief the pointer that points to the memory blocks that save the average power spectrum of noise, @f$\sigma^{2}@f$
         */
        const int* _iSig;
        
        /**
         * @brief the vector saving the weights of each insertion with image, associated 5D coordinate and weight.
         * Since there are serveral different 5D coordinates that have the similar possibility for a single image, multiple 5D coordinates and weights of a single image can be inserted. Every inserting operation will also insert the weights into this vector.
         */
        vector<RFLOAT> _w;
        
        /**
         * @brief the max radius within the distance of which other point can be affected when one point is doing interpolation
         */
        int _maxRadius;

        /**
         * @brief the padding factor which defined the PAD_SIZE (_pf * _size). By default, _pf = 2.
         */
        int _pf;

        /**
         * @brief the symmetry mark of the model, which is used to reduce computation
         */
        const Symmetry* _sym;

        /**
         * @brief the vetcor containing FSC values
         */
        vec _FSC;

        /**
         * @brief the average power spectrum of noise, @f$\sigma^{2}@f$
         */
        vec _sig;

        /**
         * @brief the power spectrum of a certain reference, @f$\tau^{2}@f$
         */
        vec _tau;

        /**
         * @brief the x-axis component of offset, defaulted by 0
         */
        double _ox;

        /**
         * @brief the y-axis component of offset, defaulted by 0
         */
        double _oy;

        /**
         * @brief the z-axis component of offset, defaulted by 0
         */
        double _oz;

        /**
         * @brief the counted number of offset
         */
        int _counter;

        /**
         * @brief the width of the kernel of modified Kaiser-Bessel Kernel
         */
        RFLOAT _a;

        /**
         * @brief the smoothness parameter of modified Kaiser-Bessel Kernel
         */
        RFLOAT _alpha;
        
        /**
         * @brief the blob kernel in Fourier space stored as a tabular function
         */
        TabFunction _kernelFT;

        /**
         * @brief the blob kernel in real space stored as a tabular function
         */
        TabFunction _kernelRL;

        /**
         * @brief FFT structure
         */
        FFT _fft;

        /**
         * @brief Initialize the default parameters.
         */
        void defaultInit()
        {
            _mode = MODE_3D;

            _calMode = POST_CAL_MODE;

            _MAP = true;

            _gridCorr = true;

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

            _ox = 0;
            _oy = 0;
            _oz = 0;

            _counter = 0;
        }

    public:

        /**
         * @brief Construct an empty Reconstructor object.
         */
        Reconstructor();

        /**
         * @brief Construct a specific Reconstructor object with specific parameters.
         */
        Reconstructor(const int       mode,        /**< [in] the indicator that the constructor operates in MODE_2D or MODE_3D */
                      const int       size,        /**< [in] the size (PAD_SIZE) of Volume in 3 dimensions(xyz) */
                      const int       N,           /**< [in] the real size of the 3D Fourier reconstructor space */
                      const int       pf = 2,      /**< [in] the padding factor, defaulted by 2 */
                      const Symmetry* sym = NULL,  /**< [in] the symmetry mark, defaulted by NULL */
                      const RFLOAT    a = 1.9,     /**< [in] the width of the modified Kaiser-Bessel Kernel, defaulted by 1.9 */
                      const RFLOAT    alpha = 15   /**< [in] the smoothness parameter of modified Kaiser-Bessel kernel, defaulted by 15 */
);

        /**
         * @brief Default deconstructor.
         */
        ~Reconstructor();

        /** 
         * @brief Initialize the member data of a reconstruct object. 
         */
        void init(const int       mode,        /**< [in] the indicator that the constructor operates in MODE_2D or MODE_3D */
                  const int       size,        /**< [in] the size (PAD_SIZE) of Volume in 3 dimensions(xyz) */
                  const int       N,           /**< [in] the real size of the 3D Fourier reconstructor space */
                  const int       pf = 2,      /**< [in] the padding factor, defaulted by 2 */
                  const Symmetry* sym = NULL,  /**< [in] the symmetry mark, defaulted by NULL */
                  const RFLOAT    a = 1.9,     /**< [in] the width of the modified Kaiser-Bessel Kernel, defaulted by 1.9 */
                  const RFLOAT    alpha = 15   /**< [in] the smoothness parameter of modified Kaiser-Bessel kernel, defaulted by 15 */
                 ); 

        /**
         * @brief Create Fourier transform plan and then allocate space for _F3D(_F2D), _W3D(_W2D), _C3D(_C2D), _T3D(_T2D).
         */
        void allocSpace(const unsigned int nThread     /**< [in] the number of threads */);

        /**
         * @brief Free the space created by allocSpace().
         */
        void freeSpace();

        /**
         * @brief Destroy the Fourier transform plan and change the size of the Fourier space(_size).
         */
        void resizeSpace(const int size /**< [in] the new real size of the 3D Fourier reconstructor space */);

        /**
         * @brief Reset all parameters.
         */
        void reset(const unsigned int nThread     /**< [in] the number of threads */);

        /**
         * @brief Return the operation mode, MODE_2D or MODE_3D.
         *
         * @return the operation mode, MODE_2D or MODE_3D
         */
        int mode() const;

        /**
        * @brief Set the operation mode, MODE_2D or MODE_3D.
        */
        void setMode(const int mode /**< [in] the operation mode, MODE_2D or MODE_3D */);

        /**
         * @brief Return the indicator of whether to use the MAP method(TRUE) or not(FLASE).
         *
         * @return the indicator of whether to use the MAP method(TRUE) or not(FLASE).
         */
        bool MAP() const;

        /**
         * @brief Set the indicator of whether to use the MAP method(TRUE) or not(FLASE).
         */
        void setMAP(const bool MAP /**< [in] the indicator of whether to use the MAP method(TRUE) or not(FLASE) */);

        /**
         * @brief Return the indicator of whether to correct the grid(TRUE) or not(FLASE).
         *
         * @return the indicator of whether to correct the grid(TRUE) or not(FLASE).
         */
        bool gridCorr() const;

        /**
         * @brief Set the indicator of whether to correct the grid(TRUE) or not(FALSE).
         */
        void setGridCorr(const bool gridCorr /**< [in] the indicator of whether to correct the grid(TRUE) or not(FALSE) */);

        /**
         * @brief Return the indicator of whether to join the two hemispheres(TRUE) or not(FLASE), helpful for calculating FSC.
         *
         * @return the indicator of whether to join the two hemispheres(TRUE) or not(FLASE).
         */
        bool joinHalf() const;

        /**
         * @brief Set the indicator of whether to join the two halves(TRUE) or not(FALSE).
         */
        void setJoinHalf(const bool joinHalf /**< [in] the indicator of whether to join the two halves(TRUE) or not(FALSE) */);

        /** 
         * @brief Set the symmetry mark of the model to be reconstructed.
         */
        void setSymmetry(const Symmetry* sym /**< [in] symmetry mark */);

        /**
         * @brief Set the FSC of the model to be reconstructed.
         */
        void setFSC(const vec& FSC /**< [in] the FSC of the model */);

        /**
         * @brief Set the power spectrum of a certain reference, @f$\tau^{2}@f$.
         */
        void setTau(const vec& tau /**< [in] the the power spectrum of a certain reference, @f$\tau^{2}@f$ */);

        /**
         * @brief Set the average power spectrum of noise, @f$\sigma^{2}@f$.
         */
        void setSig(const vec& sig /**< [in] the average power spectrum of noise, @f$\sigma^{2}@f$ */);

        /**
         * @brief Set the estimated x-axis offset of the reference (pixel).
         */
        void setOx(const double ox /**< [in] x-axis offset of the reference (pixel) */);

        /**
         * @brief Set the estimated y-axis offset of the reference (pixel).
         */
        void setOy(const double oy /**< [in] y-axis offset of the reference (pixel) */);

        /**
         * @brief Set the estimated z-axis offset of the reference (pixel).
         */
        void setOz(const double oz /**< [in] z-axis offset of the reference (pixel) */);

        /**
         * @brief Set the counted total number of the offsets.
         */
        void setCounter(const int counter /**< [in] the counter of the offsets */);

        /**
         * @brief Return the estimated x-axis offset of the reference (pixel).
         *
         * @return the estimated x-axis offset of the reference (pixel).
         */
        double ox() const;

        /**
         * @brief Return the estimated y-axis offset of the reference (pixel).
         *
         * @return the estimated y-axis offset of the reference (pixel).
         */
        double oy() const;

        /**
         * @brief Return the estimated z-axis offset of the reference (pixel).
         *
         * @return the estimated z-axis offset of the reference (pixel).
         */
        double oz() const;

        /**
         * @brief Return the counted total number of the offsets.
         *
         * @return the counted total number of the offsets.
         */
        int counter() const;

        /**
         * @brief Return the maximal radius within which adjacent points may be affected when doing interpolation on a point. 
         *
         * @return the maximal radius within which adjacent points may be affected when doing interpolation on a point. 
         */
        int maxRadius() const;

        /**
         * @brief Set the value of the maximal radius within which adjacent points may be affected when doing interpolation on a point. 
         */
        void setMaxRadius(const int maxRadius /**< [in] the maximal radius that points may affect each other during interpolation */);

        /**
         * @brief Get pre-calculation parameters.
         *
         * @return pre-calculation parameters
         */
        void preCal(int&       nPxl,   /**< [out] the number of pixels */
                    const int* iCol,   /**< [out] the pointer that points to the memory blocks that save the column index of the pixels */
                    const int* iRow,   /**< [out] the pointer that points to the memory blocks that save the row index of the pixels */
                    const int* iPxl,   /**< [out] the pointer that points to the memory blocks that save the indices of the pixels */
                    const int* iSig    /**< [out] the pointer that points to the memory blocks that save the average power spectrum of noise, @f$\sigma^{2}@f$ */
                   ) const;

        /**
         * @brief Set pre-calculation parameters.
         */
        void setPreCal(const int  nPxl,  /**< [in] the number of pixels */
                       const int* iCol,  /**< [in] the pointer that points to the memory blocks that save the column index of the pixels */
                       const int* iRow,  /**< [in] the pointer that points to the memory blocks that save the row index of the pixels */
                       const int* iPxl,  /**< [in] the pointer that points to the memory blocks that save the indices of the pixels */
                       const int* iSig   /**< [in] the pointer that points to the memory blocks that save the average power spectrum of noise, @f$\sigma^{2}@f$ */
                      );

        /**
         * @brief Set the offsets in 2D mode.
         */
        void insertDir(const dvec2& dir  /**< [in] 2D offset direction */);

        /**
         * @brief Set the offsets in 3D mode.
         */
        void insertDir(const dvec3& dir  /**< [in] 3D offset direction */);

        /**
         * @brief Set the estimated offsets in 3D mode.
         */
        void insertDir(const double ox,  /**< [in] the estimated x-axis offset of reference */
                       const double oy,  /**< [in] the estimated y-axis offset of reference */
                       const double oz   /**< [in] the estimated z-axis offset of reference */
                      );

        /**
         * @brief Insert the 2D images into reconstructor with associated 2D rotaion matrix, 2D translation vector and weights by CPU. The image data src will be accumulated onto the relative grid points of volume _F by translation vector, rotation matrix and interpolation. The rotation matrix will be recorded into vector _rot. The weights will be saved in vector _w.
         */
        void insert(const Image&  src,   /**< [in] the image data to insert */
                    const Image&  ctf,   /**< [in] the CTF of images */
                    const dmat22& rot,   /**< [in] the rotation matrix to transform the image to its real location according to the projecting direction */
                    const RFLOAT  w      /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                   );

        /**
         * @brief Insert a 2D Fourier transform of image pixel data with associated 3D rotation matrix, 2D translation vector and weights into member data by CPU. The image data src will be accumulated onto the relative grid points of volume _F by translation vector, rotation matrix and interpolation. The rotation matrix will be recorded into vector _rot. The weights will be saved in vector _w.
         */
        void insert(const Image&  src,   /**< [in] the image data to be inserted */
                    const Image&  ctf,   /**< [in] the CTF of images */
                    const dmat33& rot,   /**< [in] the rotation matrix to transform the image to its real location based on the projecting direction */
                    const RFLOAT  w      /**< [in] the weights that measure the possibility of the rotate matrix and translation vector */
                   );

        /**
         * @brief Insert the 2D images into reconstructor with associated 2D rotation matrix, weights and average power spectrum of noise by CPU. 
         * The image data src will be accumulated onto the relative grid points of volume _F by translation vector, rotation matrix and interpolation. The rotation matrix will be recorded into vector _rot. The weights will be saved in vector _w.
         */
        void insertP(const Image&  src,       /**< [in] the image data to be inserted */
                     const Image&  ctf,       /**< [in] the CTF of images */
                     const dmat22& rot,       /**< [in] the rotation matrix that transforms the image into its real location based on the projecting direction */
                     const RFLOAT  w,         /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                     const vec*    sig = NULL /**< [in] the average power spectrum of noise */
                    );

        /**
         * @brief Insert a 2D Fourier transform of image pixel data with associated 3D rotation matrix, weights into member data, 2D translation vector and average power spectrum of noise by CPU. 
         * The image data src will be accumulated onto the relative grid points of volume _F by translation vector, rotation matrix and interpolation. The rotation matrix will be recorded into vector _rot. The weights will be saved in vector _w.
         */
        void insertP(const Image&  src,       /**< [in] the image data to be inserted */
                     const Image&  ctf,       /**< [in] the CTF of images */
                     const dmat33& rot,       /**< [in] the rotation matrix that transforms the image into its real location based on the projecting direction */
                     const RFLOAT  w,         /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                     const vec*    sig = NULL /**< [in] the average power spectrum of noise */
                    );

        /**
        * @brief  Insert a 2D Fourier transform of image pixel data with associated 2D rotaion matrix, weights and average power spectrum of noise by CPU. 
        * The image data src will be accumulated onto the relative grid points of volume _F by translation vector, rotation matrix and interpolation. The rotation matrix will be recorded into vector _rot. The weights will be saved in vector _w.
        */
        void insertP(const Complex* src,       /**< [in] the complex image data to be inserted */
                     const RFLOAT*  ctf,       /**< [in] the CTF of images */
                     const dmat22&  rot,       /**< [in] the rotation matrix that transforms the image into its real location based on the projecting direction */
                     const RFLOAT   w,         /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                     const vec*     sig = NULL /**< [in] the average power spectrum of noise */
                    );

        /**
         * @brief Insert a 2D Fourier transform of image pixel data with associated 3D rotation matrix, weights into member data, 2D translation vector and average power spectrum of noise by CPU. 
         * The image data src will be accumulated onto the relative grid points of volume _F by translation vector, rotation matrix and interpolation. The rotation matrix will be recorded into vector _rot. The weights will be saved in vector _w.
         */
        void insertP(const Complex* src,       /**< [in] the complex image data to be inserted */
                     const RFLOAT*  ctf,       /**< [in] the CTF of images */
                     const dmat33&  rot,       /**< [in] the rotation matrix that transforms the image into its real location based on the projecting direction */
                     const RFLOAT   w,         /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                     const vec*     sig = NULL /**< [in] the average power spectrum of noise */
                    );

#ifdef GPU_INSERT

        /**
         * @brief Insert the complex 2D images into reconstructor by GPU. 
         */
        void insertI(Complex* datP,      /**< [in]  */
                     RFLOAT*  ctfP,      /**< [in]  */
                     RFLOAT*  sigP,      /**< [in]  */
                     RFLOAT*  w,         /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                     double*  offS,      /**< [in]  */
                     double*  nr,        /**< [in]  */
                     double*  nt,        /**< [in]  */
                     double*  nd,        /**< [in]  */
                     int*     nc,        /**< [in]  */
                     CTFAttr* ctfaData,  /**< [in]  */
                     RFLOAT   pixelSize, /**< [in] pixel size */
                     bool     cSearch,   /**< [in] the indicator of whther to perform ctf search or not */
                     int      opf,       /**< [in]  */
                     int      mReco,     /**< [in]  */
                     int      idim,      /**< [in] boxsize of image */
                     int      imgNum     /**< [in] number of images */
                    );

        /**
         * @brief Insert the complex 2D images into reconstructor by GPU. 
         */
        void insertI(Complex* datP,      /**< [in]  */
                     RFLOAT*  ctfP,      /**< [in]  */
                     RFLOAT*  sigP,      /**< [in]  */
                     RFLOAT*  w,         /**< [in] the weights that measure the possibility of the rotation matrix and translation vector */
                     double*  offS,      /**< [in]  */
                     double*  nr,        /**< [in]  */
                     double*  nt,        /**< [in]  */
                     double*  nd,        /**< [in]  */
                     CTFAttr* ctfaData,  /**< [in]  */
                     RFLOAT   pixelSize, /**< [in] pixel size */
                     bool     cSearch,   /**< [in] the indicator of whther to perform ctf search or not */
                     int      opf,       /**< [in]  */
                     int      mReco,     /**< [in]  */
                     int      idim,      /**< [in] boxsize of image */
                     int      imgNum     /**< [in] number of images */
                    );

        /**
         * @brief Get the dimension of _F2D.
         *
         * @return the dimension of _F2D
         */
        int getModelDim();

        /**
         * @brief Get the size of _F2D.
         *
         * @return the size of _F2D
         */
        int getModelSize();

        /**
         * @brief Get the values of _F2D.
         */
        void getF(Complex* modelF    /**< [out] the values of _F2D */);

        /**
         * @brief Get the real part values of _T2D.
         */
        void getT(RFLOAT* modelT     /**< [out] the real part of _T2D */);

        /**
         * @brief Reset the values of _F2D by modelF.
         */
        void resetF(Complex* modelF  /**< [in] reset values */);

        /**
         * @brief Reset the values of _T2D by modelT.
         */
        void resetT(RFLOAT* modelT   /**< [in] reset values */);

        /**
         * @brief Prepare parameters for symmetrizing _T and _F by GPU.
         */
        void prepareTFG(int gpuIdx   /**< [in] gpu index */);
#endif

        /**
         * @brief Prepare parameters for symmetrizing _T and _F by CPU.
         */
        void prepareTF(const unsigned int nThread      /**< [in] the number of threads */);

        /**
         * @brief Estimate X-offset, Y-offset and Z-offset of reference by averaging offsets summation, which is calculated by former allreduction operation.
         */
        void prepareO();

        /**
         * @brief Reconstruct a 2D model by CPU and save it into a volume.
         */
        void reconstruct(Image& dst,                 /**< [in] the destination volume that reconstructor object saves the result of reconstruction into */
                         const unsigned int nThread  /**< [in] the number of threads */
                        );

        /**
         * @brief Reconstruct a 3D model by CPU and save it into a volume.
         */
        void reconstruct(Volume& dst,                /**< [in] the destination volume that reconstructor object saves the result of reconstruction into */
                         const unsigned int nThread  /**< [in] the number of threads */
                        );

#ifdef GPU_RECONSTRUCT
        /**
         * @brief Reconstruct a 3D model and save it into a volume by GPU.
         */
        void reconstructG(Volume& dst,               /**< [in] the destination volume where the reconstructor object saves the result of reconstruction */
                          int gpuIdx,                /**< [in] the gpu index */
                          const unsigned int nThread /**< [in] the number of threads */
                         );
#endif

    private:

        /**
         * @brief The allreduce operation gets the final summation of _F volumes of all processes, by which the 3D Fourier transform of the model is obtained. The size of the reconstructor area that is used to determine the size of Volume in 3 dimension xyz.
         */
        void allReduceF();

        /**
         * @brief The allreduce operation gets the summation of _T volumes. Get the summation of accumulated weights value of each grid points of all processes before balancing and normalization by the divion of _W and _C, with which _C can be approximately equal to 1.
         */
        void allReduceT(const unsigned int nThread   /**< [in] the number of threads */);

        /**
         * @brief The allreduce operation gets the summation of X-offset, Y-offset, Z-offset of reference and the counter.
         */
        void allReduceO();

        /**
         * @brief Calculate the distance to total balanced. The distance is the summation of diffrernces between real weights and ideal total balance weights, calculated by @f$\sum_{i^2+j^2+k^2<\left({maxradius}*{pf}\right)^2}\left(\left|C\left(i,j,k\right)\right|-1\right)@f$ .
         *
         * @return the distance to total balanced
         */
        RFLOAT checkC(const unsigned int nThread      /**< [in] the number of threads */) const;

        /**
         * @brief Convolute _C.
         */
        void convoluteC(const unsigned int nThread    /**< [in] the number of threads */);

        /**
         * @brief Symmetrize _F.
         */
        void symmetrizeF(const unsigned int nThread   /**< [in] the number of threads */);

        /**
         * @brief Symmetrize _T.
         */
        void symmetrizeT(const unsigned int nThread  /**< [in] the number of threads */);

        /**
         * @brief Symmetrize X-offset, Y-offset and Z-offset of reference.
         */
        void symmetrizeO();
};

#endif //RECONSTRUCTOR_H
