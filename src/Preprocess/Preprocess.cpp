
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
	
    para.verbose             = 0;
    para.toExtractParticle   = false;
    para.dimensionality      = 2 ;
    para.toProject3D         = false;
    para.particleSize        = DEFAULT_PARTICLE_SIZE;
    para.biasXCord          = 0;
    para.biasYCord          = 0;

    para.scale              = 1;

    para.window             = 1;
     
    para.toRescale          = true;     
    para.toRewindow         = true;
    para.toNormalise        = true;
    para.toRamp             = false;
    para.toInvertContrast   = true;
     
    para.white_dust_stddev  = 0;
    para.black_dust_stddev  = 0;
    para.bg_radius          = 1;

	// getpid()?
	sprintf(para.micrographPath,    MICROGRAPH_ROOT_PATH);
	sprintf(para.particlePath,      PARTICLE_ROOT_PATH  );

	
	// If particle path not exists, make a Particles directory first.
	if ( ! exists(para.particlePath) )
    {        
        sprintf(sysCmd,"mkdir -p %s ", para.particlePath );
		system(sysCmd );                      
    }   
		
}


int Preprocess:extractParticles(const int micrographID)
{
	// get micrographPath
	// get vector<particleInfo>

	ImageFile micrographFile;
	Image     micrograph;

	// read in micrograph

	// loop over particle
		// extract
		// [rescale]
		// [rewindow]
		// [normalise]
		// save
}


int  Preprocess::extractParticles(COORDINATE &coordinate)
{
    
	char *  fnMic[MAX_FILE_NAME_LENGTH];
	char *  fnParticle[MAX_FILE_NAME_LENGTH];
			
        // Check the micrograph exists      
		
		if (  (  strlen(para.micrographPath) +  strlen( coordinate.pMicroGraphFileName) +1 > MAX_FILE_NAME_LENGTH -1 ) 
	       || (  strlen(para.particlePath) +    strlen( coordinate.pParticleName      ) +1 > MAX_FILE_NAME_LENGTH -1 )   ) 
		{
			printf(" Micrograph or particle name is too long. Aborting ...\n");
			return ;
		}

        sprintf( fnMic ,     "%s/%s",para.micrographPath, coordinate.pMicroGraphFileName );		
        sprintf( fnParticle ,"%s/%s",para.particlePath  , coordinate.pParticleName       );		
		
        // Return if the micrograph does not exist
        if (fnMic == ""  ||   ! exists(fnMic)  )
        {
            std::cout << "WARNING: cannot find micrograph for coordinate file " << pCoordinate->pMicroGraphFileName <<  std::endl;
            continue;
        }		
		
		ImageFile micrographFile(fnMic, "w");		 
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

		// resacle(dst, src, ??)
		// rewind(dst, src, ???)
		
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
		// vector<int> micrographIDs;
		extractParticles(coordinate); 		
		
    }

    if (verb > 0)
    {
        std::cout << " Extracting particles from the micrographs is finished..." << std::endl;            
    }
	
}

