/*******************************************************************************
 * Author: Mingxu Hu, Hongkun Yu, Siyuan Ren
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include "easylogging++.h"

#include "Macro.h"
#include "Precision.h"

namespace el = easyloggingpp;

#define INITIALIZE_EASYLOGGINGPP _INITIALIZE_EASYLOGGINGPP

#define REPORT_ERROR(msg) CLOG(FATAL, "LOGGER_SYS") << __FILE__ \
                                                    << ", " \
                                                    << __LINE__ \
                                                    << ", " \
                                                    << __FUNCTION__ \
                                                    << ": " \
                                                    << msg;

void loggerInit(int argc, const char* const * argv);

#define FGETS_ERROR_HANDLER(command) \
    do \
    { \
        if (command == NULL) \
        { \
            REPORT_ERROR("FAIL TO FGETS"); \
            abort(); \
        } \
    } while (0);

#define GETCWD_ERROR_HANDLER(command) \
    do \
    { \
        if (command == false) \
        { \
            REPORT_ERROR("FAIL TO GETCWD"); \
            abort(); \
        } \
    } while (0);

long memoryCheckParseLine(char* line)
{
    // This assumes that a digit will be found and the line ends in " Kb".
    long i = strlen(line);

    const char* p = line;

    while (*p <'0' || *p > '9') p++;

    line[i - 3] = '\0';

    i = atoi(p);

    return i;
};

long memoryCheckVM()
{
    FILE* file = fopen("/proc/self/status", "r");

    if (file == NULL)
    {
        REPORT_ERROR("FAIL TO OPEN /proc/self/status");

        abort();
    }

    long result = -1;

    char line[128];

    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmSize:", 7) == 0)
        {
            result = memoryCheckParseLine(line);

            break;
        }
    }

    fclose(file);

    return result;
};

long memoryCheckRM()
{
    FILE* file = fopen("/proc/self/status", "r");

    if (file == NULL)
    {
        REPORT_ERROR("FAIL TO OPEN /proc/self/status");

        abort();
    }

    long result = -1;

    char line[128];

    while (fgets(line, 128, file) != NULL)
    {
        if (strncmp(line, "VmRSS:", 6) == 0)
        {
            result = memoryCheckParseLine(line);

            break;
        }
    }

    fclose(file);

    return result;
};

#endif // LOGGING_H
