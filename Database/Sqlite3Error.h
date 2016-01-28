#include <cstdio>

#include <sqlite3.h>

#define SQLITE3_HANDLE_ERROR(err) (sqlite3HandleError(err, __FILE__, __LINE__))

void sqlite3HandleError(const int err,
                       const char* file,
                       const int line);

const char* sqlite3GetErrorString(const int err);
