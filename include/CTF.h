/** @file
 *  @author Mingxu Hu
 *  @author Shouqing Li
 *  @version 1.4.11.080913
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu   Hu | 2015/03/23 | 0.0.1.050323  | new file
 *  Shouqing Li | 2018/09/13 | 1.4.11.080913 | add doc 
 *  
 *  @brief CTF.h contains several functions to satisfy the need of CTF calculations for various conditions, such as 1D, 2D and certain pixels. All these operations are carried in Fourier space with the variable spatial frequency.
 *
 *  Noted that, @f$CTF=-\sqrt{1-A^2}\sin(\chi)+A\cos(\chi)@f$ is the function used to calculate CTF, in which A is the fraction of total contrast attributed to amplitude contrast and @f$\chi@f$ denotes the function @f$\chi=\pi\lambda\Delta fH^{2}+\frac{\pi}{2}C_{S}\lambda^{3}H^{4}-\Delta\varphi@f$, the effect of defocus @f$\Delta f@f$, spherical aberration @f$C_S@f$ and an additional phase shift @f$\Delta\varphi@f$. @f$H@f$ means the spatial frequency. When it comes to the 2D condition, the astigmatism should also be taken into consideration through @f$\Delta{f}=\frac{1}{2}[\Delta{f_1}+\Delta{f_2}+\Delta\Delta f\cos(2[\alpha{_g}-\alpha{_{\alpha st}}])] @f$, where @f$ \Delta{f_1} @f$ and @f$ \Delta{f_2} @f$ are the len's defocus along normal directions, @f$ \Delta\Delta{f}=\Delta{f_1}-\Delta{f_2} @f$, @f$\alpha{_g}@f$ represents the angle between vector of pixel choosed and X axis and @f$\alpha{_{\alpha st}}@f$ is the angle between X axis and @f$\Delta{f_1}@f$ direction. @f$V@f$ in @f$V@f$. @f$H@f$ in @f${\si{\angstrom}}^{-1}@f$. @f$r@f$ in @f${\si{\angstrom}}^{-1}@f$. @f$\Delta f@f$ in @f${\si{\angstrom}}@f$. @f$\Delta f_1@f$ in @f${\si{\angstrom}}@f$. @f$\Delta f_2@f$ in @f${\si{\angstrom}}@f$. @f$\lambda@f$ in @f${\si{\angstrom}}@f$. @f$a@f$ in @f${\si{\angstrom}}@f$.@f$\Delta\varphi@f$ in @f$rad@f$. @f$\alpha_g@f$ in @f$rad@f$. @f$\alpha_{\alpha st}@f$ in @f$rad@f$.  
 */

#ifndef CTF_H
#define CTF_H

#include "Complex.h"
#include "Functions.h"
#include "Image.h"
#include "Precision.h"

/**
 * @brief This function returns the 1D CTF with paremeters given in Fourier space. 
 *
 * The input of voltage determines the value of wavelenghth @f$\lambda@f$. The other variables are substituded into CTF equation computing 1D CTF.
 */
RFLOAT CTF(const RFLOAT f,                  /**< [in] @f$H@f$ */
           const RFLOAT voltage,            /**< [in] @f$V@f$ */
           const RFLOAT defocus,            /**< [in] @f$\Delta f@f$ */
           const RFLOAT CS,                 /**< [in] @f$C_S@f$ */
           const RFLOAT amplitudeContrast,  /**< [in] @f$A@f$ */
           const RFLOAT phaseShift          /**< [in] @f$\Delta\varphi@f$ */
           );

/**
 * @brief This function is used to calculate the 2D CTF of a given image in Fourier space, output into the image @f$I@f$.
 *
 * Assign the computed 2D CTF values to every image pixel after getting the column and row number of it. 
 */
void CTF(Image& dst,                        /**< [out, in] @f$I@f$ */
         const RFLOAT pixelSize,            /**< [in] @f$a@f$ */
         const RFLOAT voltage,              /**< [in] @f$V@f$ */
         const RFLOAT defocusU,             /**< [in] @f$\Delta f_1@f$ */
         const RFLOAT defocusV,             /**< [in] @f$\Delta f_2@f$ */
         const RFLOAT theta,                /**< [in] @f$\alpha{_{\alpha st}}@f$ */
         const RFLOAT Cs,                   /**< [in] @f$C_S@f$ */
         const RFLOAT amplitudeContrast,    /**< [in] @f$A@f$ */
         const RFLOAT phaseShift            /**< [in] @f$\Delta\varphi@f$ */
         );

/**
 * @brief This function calculates CTF within a cut-off spatial frequency @f$r@f$ in Fourier space, output into the image @f$I@f$. 
 */
void CTF(Image& dst,                        /**< [out, in] @f$I@f$ */
         const RFLOAT pixelSize,            /**< [in] @f$a@f$  */
         const RFLOAT voltage,              /**< [in] @f$V@f$ */
         const RFLOAT defocusU,             /**< [in] @f$\Delta f_1@f$ */
         const RFLOAT defocusV,             /**< [in] @f$\Delta f_2@f$ */
         const RFLOAT theta,                /**< [in] @f$\alpha{_{\alpha st}}@f$ */
         const RFLOAT Cs,                   /**< [in] @f$C_S@f$ */
         const RFLOAT amplitudeContrast,    /**< [in] @f$A@f$ */
         const RFLOAT phaseShift,           /**< [in] @f$\Delta\varphi@f$ */
         const RFLOAT r                     /**< [in] @f$r@f$ */
         );

/**
 * @brief This function provides a convenient way to compute certain pixel-positions' CTF values in Fourier space, output in a float array @f$I@f$.
 *
 * @f$X@f$ and @f$Y@f$ are the number of columns and rows of the given image in real space. @f$x@f$ and @f$y@f$ mean the column and row number of a certain pixel. @f$N@f$ represents the total number of pixels to be calculated. All the results will be assigned to these points respectively. The spatial frequency @f$H@f$ can be got by formula @f$H=\sqrt{{(\frac{x_i}{aX})}^2+{(\frac{y_i}{aY})}^2}@f$, the range of i is from 0 to N-1.
 */
void CTF(RFLOAT* dst,                       /**< [out] @f$I@f$ */          
         const RFLOAT pixelSize,            /**< [in] @f$a@f$ */
         const RFLOAT voltage,              /**< [in] @f$V@f$ */
         const RFLOAT defocusU,             /**< [in] @f$\Delta f_1@f$ */
         const RFLOAT defocusV,             /**< [in] @f$\Delta f_2@f$ */
         const RFLOAT theta,                /**< [in] @f$\alpha{_{\alpha st}}@f$ */
         const RFLOAT Cs,                   /**< [in] @f$C_S@f$ */
         const RFLOAT amplitudeContrast,    /**< [in] @f$A@f$ */
         const RFLOAT phaseShift,           /**< [in] @f$\Delta\varphi@f$ */
         const int nCol,                    /**< [in] @f$X@f$ */
         const int nRow,                    /**< [in] @f$Y@f$ */
         const int* iCol,                   /**< [in] @f$x_i@f$ */
         const int* iRow,                   /**< [in] @f$y_i@f$ */
         const int nPxl                     /**< [in] @f$N@f$ */
         );

#endif // CTF_H
