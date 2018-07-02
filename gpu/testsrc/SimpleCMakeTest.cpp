#include "easylogging++.h"
#include <nccl.h>

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[])
{
    LOG(INFO) << "Allocating memory on device..";

    ncclResult_t ncclGroupStart();
	
    LOG(INFO) << "Memory allocation done!";

    ncclResult_t ncclGroupEnd();

	return 0;
}
