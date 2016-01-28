/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef SQLITE3_ERROR
#define SQLITE3_ERROR

#include <cstdio>
#include <string>

#include <sqlite3.h>

#include "Error.h"

#define SQLITE3_HANDLE_ERROR(err) \
    [](const int _err) throw() \
    { \
        if ((_err != 0) && \
            (_err != 100) && \
            (_err != 101)) \
            REPORT_ERROR(std::string(sqlite3GetErrorString(_err))); \
    }(err)

const char* sqlite3GetErrorString(const int err);

#endif // SQLITE3_ERROR
