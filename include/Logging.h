//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu, Hongkun Yu
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

long memoryCheckParseLine(char* line);

long memoryCheckVM();

long memoryCheckRM();

#endif // LOGGING_H
