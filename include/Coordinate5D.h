/*******************************************************************************
 * Authors Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef COORDINATE_5D_H
#define COORDINATE_5D_H

#include <cstdio>

/**
 * @ingroup Coordinate5D
 * @brief The 5-Dimension coordinate parameters.
 * 
 * Provides the relation between real images and the 3D model of the protein.
 * Each real image is a projection of the real model in a particular coordinate
 * in the 5-Dimension space.
 * Consider a fixed coordinates axis system Oxyz in 3D space and a coordinate 
 * axis system Ox1y1z1 fixed on the protein, which is moved with the protein 
 * that is regarded as a rigid body, in 3D space. 
 * The Euler angles phi, theta, psi are used to describe the spatial 
 * orientation of protein rigid body coordinates axis system Ox1y1z1 as a 
 * composition of three elemental rotations starting from the fixed axis 
 * system Oxyz.  
 * As the figure shown below. Fixed axis system Oxyz, moved axis system Ox1y1z1
 * and the intersection of the xy and the x1y1 coordinate planes, the line 
 * of nodes (N axis, which satisfies the right hand rule with the rotation from
 * z axis to z1 axis).
 */
struct Coordinate5D
{

    /**
     * The angle rotates from the fixed x axis to N axis is called intrinsic
     * angle phi.
     * Phi represents a rotation around fixed z axis. 
     * The range of phi is defined modulo 2PI radians, which can be set as
     * [-PI, PI].
     */
    double phi;
    
    /**
     * The angle rotates from fixed z axis to moved z1 axis is called nutation
     * angle theta.
     * Theta represents a rotation around N axis.
     * The range of theta cover PI radians but cannot be saided to be modulo
     * PI, which can be set as [-PI/2, PI/2] or [0, PI].
     */
    double theta;
    
    /**
     * The angle rotates from N axis to the moved x1 axis is called precession
     * angle psi.
     * Psi represents a rotation around moved z1 axis.
     * The range of psi is defined modulo 2PI radians, which can be set as
     * [-PI, PI].
     */
    double psi;
    
    /**
     * The offset between the center of the image and the center of protein 
     * projection in x-direction.
     */
    double x;
    
    /**
     * The offset between the center of the image and the center of protein 
     * projection in y-direction.
     */
    double y;

    /**
     * Construct a empty Coordinate5D object.
     */
    Coordinate5D();

    /**
     * Construct a specific Coordinate5D object with 5D coordinates parameters.
     *
     * @param phi Intrinsic angle phi. Details see @ref phi.
     * @param theta Nutation angle theta. Details see @ref theta. 
     * @param psi Precession angle psi. Details see @ref psi.
     * @param x Offset in X direction. Details see @ref x.
     * @param y Offset in Y direction. Details see @ref y
     */
    Coordinate5D(const double phi,
                 const double theta,
                 const double psi,
                 const double x,
                 const double y);
};

/**
 * Display the specific values of each dimension of the Coordinate5D object. 
 * @param coord the Coordinate5D object
 */
void display(const Coordinate5D& coord);

#endif // COORDINATE_5D_H
