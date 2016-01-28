#include <cstdio>

#include "Database.h"

#define N 1000000
#define A 1000
#define B 1000

/***
#define N 10000
#define A 10
#define B 10
***/

int main(int argc, char* argv[])
{
    Database db;
    size_t start, end;

    int rank, size;
    int groupID;
    int micrographID;



    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    
 //   printf("rank = %d size = %d \n", rank, size);

    system("mkdir -p " MASTER_RAMPATH);  
    system("mkdir -p " SLAVE_RAMPATH);  

    db.setCommSize(size);
    db.setCommRank(rank);

    if (rank == 0)
    {
    	db.createTableGroups();
    	db.createTableMicrographs();
    	db.createTableParticles();

    	for (int i = 0; i < A; i++)
        	db.appendGroup("");
    	for (int i = 0; i < B; i++)
        	db.appendMicrograph("", 0, 0, 0, 0, 0);


        for (int i = 0; i < N; i++)
        {
            groupID = (int)((float)rand() / RAND_MAX * A);
            micrographID = (int)((float)rand() / RAND_MAX * B);
            db.appendParticle("", groupID, micrographID);
        }
	}	

       
    if (rank==0)
    {
        start = clock();    
        db.prepareTmpFile();
        end = clock();    
    //    printf("Time Consumed : %f in prepare %d EMPTY node database.\n", 
    //           (float)(end - start) / CLOCKS_PER_SEC, size);
    }
    
    if (rank == 0) 
    	start = clock();
  
    db.send();
    
    if (rank == 0) 
    	end = clock();

/*  
    if (rank == 0)
        printf("[%d process]Time Consumed : %f in sending %d node database\n",
              rank, (float)(end - start) / CLOCKS_PER_SEC, size);     
              */
    // printf("Beginning Receiving\n");
#if 1
    {
        start = clock();
    	db.receive();
        end = clock();    
        printf("[%d process]Time Consumed : %f seconds in gethering %d node database\n",
             rank,   (float)(end - start) / CLOCKS_PER_SEC,  size);	
    }


#endif

    MPI_Finalize();
   
    return 0;
}
