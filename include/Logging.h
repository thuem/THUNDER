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
#include <cmath>
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
void initLogger(const char *logFileFullName, int rank);
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
        if (strcmp(command, "FALSE") == 0) \
        { \
            REPORT_ERROR("FAIL TO GETCWD"); \
            abort(); \
        } \
    } while (0);

long memoryCheckParseLine(char* line);

long memoryCheckVM();

long memoryCheckRM();

#ifndef NAN_NO_CHECK

inline int IS_NAN(const RFLOAT x)
{
    if (TSGSL_isnan(x) || isinf(x))
        return 1;
    else
        return 0;
};

inline int IS_NAN(const double x)
{
    if (TSGSL_isnan(x) || isinf(x))
        return 1;
    else
        return 0;
}

#define POINT_NAN_CHECK(x) POINT_NAN_CHECK_RFLOAT(x)

#define POINT_NAN_CHECK_RFLOAT(x) \
    do \
    { \
        if (IS_NAN(x)) \
        { \
            REPORT_ERROR("NAN DETECTED"); \
            abort(); \
        } \
    } while(0);

#define POINT_NAN_CHECK_COMPLEX(x) \
    do \
    { \
        if ((IS_NAN(REAL(x))) || (IS_NAN(IMAG(x)))) \
        { \
            REPORT_ERROR("NAN DETECTED"); \
            abort(); \
        } \
    } while(0);

#define SEGMENT_NAN_CHECK(x, size)

#define SEGMENT_NAN_CHECK_RFLOAT(x, size) \
    do \
    { \
        for (size_t i = 0; i < size; i++) \
            if (IS_NAN((x)[i])) \
            { \
                REPORT_ERROR("NAN DETECTED"); \
                abort(); \
            } \
    } while(0);

#define SEGMENT_NAN_CHECK_COMPLEX(x, size) \
    do \
    { \
        for (size_t i = 0; i < size; i++) \
        { \
            if (IS_NAN(REAL((x)[i])) || IS_NAN(IMAG((x)[i]))) \
            { \
                REPORT_ERROR("NAN DETECTED"); \
                abort(); \
            } \
        } \
    } while(0);

#define NAN_CHECK_DMAT33(x) \
    do \
    { \
        const double* ptr = x.data(); \
        for (int i = 0; i < 9; i++) \
        { \
            if (IS_NAN(ptr[i])) \
            { \
                REPORT_ERROR("NAN DETECTED"); \
                abort(); \
            } \
        } \
    } while(0);

#define NAN_CHECK_DMAT22(x) \
    do \
    { \
        const double* ptr = x.data(); \
        for (int i = 0; i < 4; i++) \
        { \
            if (IS_NAN(ptr[i])) \
            { \
                REPORT_ERROR("NAN DETECTED"); \
                abort(); \
            } \
        } \
    } while(0);

//void NAN_CHECK(RFLOAT* x, const size_t size);

#endif

#endif // LOGGING_H
