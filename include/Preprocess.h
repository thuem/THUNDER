#ifndef PREPROCESS_H
#define PREPROCESS_H

#include "Experiment.h"

typedef struct _PREPROCESS_PARA
{
	int  nParticleSize;
	

	int nColBias;
	int nRowBias;
	bool doNormalise;
	bool doInvertConstrast;
	double wDust;
	double bDust;
	double r;
} PREPROCESS_PARA;

class Preprocess
{
	protected:

		int _commRank;
		int _commSize;

		PREPROCESS_PARA _para;
		
		Experiment* _exp;

		vector<int> _micrographIDs;

	public:

        Preprocess();

        Preprocess(const PREPROCESS_PARA& para,
                   Experiment* exp); 

		PREPROCESS_PARA& getPara() const;

		void setPara(PREPROCESS_PARA& para);
		
		void extractParticles(int micrographID);
		
		void run();

    private:

    	void getMicrographIDs();

    	void getMicrographName(char micrographName[],
    		                   const int micrographID);

        void getParticleXOffYOff(int& xOff,
        	                     int& yOff,
        	                     const int particleID);

};

#endif // PREPROCESS_H
