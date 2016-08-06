
#include "Particle.h"

INITIALIZE_EASYLOGGINGPP

int main()
{
    loggerInit();
    Particle p;
    load(p, "Particle_0001_Round_003_000_000.par");
    
    return 0;
}
