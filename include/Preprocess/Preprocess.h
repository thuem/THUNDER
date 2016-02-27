/*******************************************************************************
 * Author: Ice
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include  <vector>
#include  <string>
#include  <stdlib.h>
#include  <stdio.h>
#define  MAX_FILE_NAME_LENGTH  (1024)

#define  MICROGRAPH_ROOT_PATH   "/home/xxx/mic"
#define  PARTICLE_ROOT_PATH     "/home/xxx/particle"



typedef  _COORDINATE {
	char *pMicroGraphFileName;   // e.g,  "mg1.mic" 
	char *pParticleName;     // e.g,  "mg1".  
	int   x;
	int   y;
	
}  COORDINATE;

typedef  _IMAGE_COORDINATE  {
	
	int         size;
	COORDINATE *pCoordinate;


}   IMAGE_COORDINATE;


typedef struct   _PPREPROCESS_PARA 
{

	
	int verbose;   // Verbosity

    
	////////////////////////////////////// Extract particles from the micrographs
	// Perform particle extraction?
	bool  toExtractParticle;

	// Dimensionality of the micrographs (2 for normal micrographs, 3 for tomograms)
	int dimensionality;

	// Flag to project subtomograms along Z
	bool  toProject3D;

	// Box size to extract the particles in
	int  particleSize;

	// Bias in picked coordinates in X and in Y direction (in pixels)
	DOUBLE  biasXCord,   biasYCord;   // icelee ???

	////////////////////////////////////// Post-extraction image modifications
	// Perform re-scaling of extracted images
	bool  toRescale;
	int   scale;

	// Perform re-windowing of extracted images
	bool  toRewindow;
	int   window;

	// Perform normalization of the extract images
	bool  toNormalise;

	// Subtract ramp instead of a level background in normalization
	bool  toRamp;

	// Perform contrast inversion of the extracted images
	bool  toInvertContrast;

	// Standard deviations to remove black and white dust
	DOUBLE  white_dust_stddev, black_dust_stddev;

	// Radius of a circle in the extracted images outside of which one calculates background mean and stddev
	int   bg_radius;

	char  micrographPath[MAX_FILE_NAME_LENGTH];
	char  particlePath[MAX_FILE_NAME_LENGTH];


}    PPREPROCESS_PARA;



class Preprocess :
{

protected:
		
	PPREPROCESS_PARA  para;
	


public:
        
    Preprocess();
		
    ~Preprocess();
        
    init();
        
    run();
        
        
private:
    
    void  extractParticles(COORDINATE &coordinate);
		
    void  getParticleCoordinate( COORDINATE & pCoordinate );

    void  coordinatesInBoundaryRL(const int iCol,
                                  const int iRow) const;
        // check whether the given coordinates are in the boundary of the image
        // if not, throw out an Error
        

};



#endef   // of  PREPROCESS_H_