
/*******************************************************************************
 * Author: Ice  
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Preprocess.h"



void  Preprocess:init()
{
    char    sysCmd[1024];
	char   *parDir;	
	
    verbose             = 0;
    toExtractParticle   = false;
    dimensionality      = 2 ;
    toProject3D         = false;
    particleSize        = DEFAULT_PARTICLE_SIZE;
     biasXCord          = 0;
     biasYCord          = 0;

     scale              = 1;

     window             = 1;
     
     toRescale          = true;     
     toRewindow         = true;
     toNormalise        = true;
     toRamp             = false;
     toInvertContrast   = true;
     
     white_dust_stddev  = 0;
     black_dust_stddev  = 0;
     bg_radius          = 1;

	// getpid()?
	sprintf(micrographPath,    MICROGRAPH_ROOT_PATH);
	sprintf(particlePath,      PARTICLE_ROOT_PATH  );

	
	// If particle path not exists, make a Particles directory first.
	if ( ! exists(particlePath) )
    {        
        sprintf(sysCmd,"mkdir -p %s ", particlePath );
		system(sysCmd );                      
    }   
		
}




int  Preprocess::extractParticles(COORDINATE &coordinate)
{
    
	char *  fnMic[MAX_FILE_NAME_LENGTH];
	char *  fnParticle[MAX_FILE_NAME_LENGTH];
			
        // Check the micrograph exists      
		
		if (  (  strlen(micrographPath) +  strlen( coordinate.pMicroGraphFileName) +1 > MAX_FILE_NAME_LENGTH -1 ) 
	       || (  strlen(particlePath) +    strlen( coordinate.pParticleName      ) +1 > MAX_FILE_NAME_LENGTH -1 )   ) 
		{
			printf(" Micrograph or particle name is too long. Aborting ...\n");
			return ;
		}

        sprintf( fnMic ,     "%s/%s",micrographPath, coordinate.pMicroGraphFileName );		
        sprintf( fnParticle ,"%s/%s",particlePath  , coordinate.pParticleName       );		
		
        // Return if the micrograph does not exist
        if (fnMic == ""  ||   ! exists(fnMic)  )
        {
            std::cout << "WARNING: cannot find micrograph for coordinate file " << pCoordinate->pMicroGraphFileName <<  std::endl;
            continue;
        }		
		
		ImageFile   imgMicFile(fnMic,"rb");
		Image       imgMic;
        Image       imgParticle;
		
		//  read micrograph image 
		//  ???? what's slc meaning? how to determine its value ?		
		imgMicFile.readImageMRC(imgMic, 1 );	
		
		
		x0= pCoordinate->x + (particleSize)/2;
		x1= x0 + particleSize -1;		
		y0= pCoordinate->y + (particleSize)/2;
		y1= y0 + >particleSize -1;
		
		imgMic.window( imgParticle , x0, y0, x1, y1);
		
        if (toRescale) 
			imgParticle.rescale(Ipart, scale);

        if (toRewindow) 
			imgParticle.rewindow(Ipart, window);

        if (toNormalise) 
			imgParticle.normalise(Ipart, bg_radius, white_dust_stddev, black_dust_stddev, do_ramp);

        if (toInvertContrast) 
			imgParticle.invert_contrast(Ipart);
	
        imgMic.saveRLToBMP( fnParticle );
		
		 
}



//  get one particle coordinate from experiment database

void Preprocess::getParticleCoordinate(COORDINATE & coordinate)
{
    //TODO:
		
		 
}


void Preprocess::run()
{
 
	int     i;
	
	COORDINATE   coordinate;
	
    if (verb > 0)
    {
        std::cout << " Extracting particles from the micrographs ..." << std::endl;            
    }
   
        	
	while (	getParticleCoordinate(coordinate )  )
	{		
		extractParticles(coordinate); 		
		
    }

    if (verb > 0)
    {
        std::cout << " Extracting particles from the micrographs is finished..." << std::endl;            
    }
	
}

