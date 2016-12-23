#include "CSV.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string>
#include <math.h>

#define ASSERT(cond) do {if (!(cond)) fprintf(stderr, "Assertion %s failed (%s:%d)\n", #cond, __FILE__, __LINE__);} while (0)

int main(void)
{
    const char* line = "0x4141F22452,THUEM,0.44924902\n";
    std::string name;
    long ID;
    double value;
    ASSERT(parseCSVLine(line, "lsf", &ID, &name, &value) == CSVParseSuccess);
    ASSERT(name == "THUEM");
    ASSERT(ID == 0x4141F22452L);
    ASSERT(fabs(value - 0.44924902) < 0.00001);

    line = "4242,ABC,0.25";
    ASSERT(parseCSVLine(line, "lsf", &ID, &name, &value) == CSVParseSuccess);
    ASSERT(name == "ABC");
    ASSERT(ID == 4242);
    ASSERT(value == 0.25);

    line = "422.131,dfsdf,14";
    ASSERT(parseCSVLine(line, "lsf", &ID, &name, &value) == CSVParseTypeMismatch);

    fprintf(stderr, "All tests finished");
    return 0;
}
