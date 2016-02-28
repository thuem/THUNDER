#include "Preprocess.h"
#include "Experiment.h"

/* TODO list: 

  1  normalise() : how to fill the parameter?
  2  GET_PARTICLE_INFO:  how to write Lambda function and where to get x, y?
  3  invert is not implemented yet
  4  OMP #pragma is not written yet
*/

Preprocess::Preprocess();

Preprocess::Preprocess(const PREPROCESS_PARA& para,
                                      Experiment* exp)
{
    _para = para;
    _exp = exp;
}


void Preprocess::extractParticles(const int micrographID)
{
	// x, y
	// micrograph
	// _exp
	// save
    
    char    micName[FILE_NAME_LENGTH];
    char    particleName[FILE_NAME_LENGTH];
	
	vector<int> particleIDs;

    _exp->getMicrographName( micName, micrographID   );
	if ( ! exists(micName) )
    {
    	char msg[256];
    	sprintf("[Error] %s doesn't exists .\n",  micName);
        REPORT_ERROR("");
        
        return ;
    };

    // get all particleID;
	_exp->particleIDsMicrograph(particleIDs,  micrographID );
	if  (particleIDs.size() ==0 )
	{
		return ;
	}
 
    ImageFile micrographFile(fnMic, "rb");
	Image micrograph;

	//  read micrograph image 
	//  ???? what's slc meaning? how to determine its value ?		
	micrographFile.readImageMRC(micrograph, 0);    
    
    ImageFile particleFile;
    Image particle(para->nCol, para->nRow, realSpace);

	for (int i = 0; i < particleIDs.size(); i++)
	{
		int xOff, yOff;

        // extractPartcilesInMicrograph(micrographImage, particleIDs[i], particleImage  );
		
		_exp->getParticleInfo(micrographID, xOff, yOff);

		/***
        x0= x-  (_para->nParticleSize /2);
        y0= y-  (_para->nParticleSize /2);
        x1= x0+_para->nParticleSize;
        y1= y0+_para->nParticleSize;
        ***/

        extract(particle, micrograph, xOff, yOff)

        // particleImage=  window(micrographImage, x0, y0, x1, y1);    

        normalise(particle,
                  para->wDust,
                  para->bDust,
                  para->r);  // ???

		// sprintf( particleName, "%s/%d.mrc", micName, i);
		particleFile.writeImageMRC(particleName,  particleImage  );

	}

    
}



void Preprocess::run(int start, 
	                 int end )
{   
	// get all micrographID;
	vector<int> micrographIDs;

	_exp->getMicrographIDs(micrographIDs ,start, end  );
	
	for (int i = 0; i < micrographIDs.size(); i++)
	{		
		extractPartciles(micrographIDs[i]);		
	}

}
