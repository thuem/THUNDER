#include "Sqlite3Error.h"

void sqlite3HandleError(const int err,
                        const char* file,
                        const int line)
{
    if ((err != 0) &&
        (err != 100) &&
        (err != 101))
        printf("%s in %s at line %d\n", sqlite3GetErrorString(err), file, line);
}

const char* sqlite3GetErrorString(const int err)
{
    switch(err)
    {
        case 0:
            return "SQLITE_OK";
        case 1:
            return "SQLITE_ERROR";
        case 2:
            return "SQLITE_INTERNAL";
        case 3:
            return "SQLITE_PERM";
        case 4:
            return "SQLITE_ABORT";
        case 5:
            return "SQLITE_BUSY";
        case 6:
            return "SQLITE_LOCKED";
        case 7:
            return "SQLITE_NOMEM";
        case 8:
            return "SQLITE_READONLY";
        case 9:
            return "SQLITE_INTERRUPT";
        case 10:
            return "SQLITE_IOERR";
        case 11:
            return "SQLITE_CORRUPT";
        case 12:
            return "SQLITE_NOTFOUND";
        case 13:
            return "SQLITE_FULL";
        case 14:
            return "SQLITE_CANTOPEN";
        case 15:
            return "SQLITE_PROTOCOL";
        case 16:
            return "SQLITE_EMPTY";
        case 17:
            return "SQLITE_SCHEMA";
        case 18:
            return "SQLITE_TOOBIG";
        case 19:
            return "SQLITE_CONSTRAINT";
        case 20:
            return "SQLITE_MISMATCH";
        case 21:
            return "SQLITE_MISUSE";
        case 22:
            return "SQLITE_NOLFS";
        case 23:
            return "SQLITE_AUTH";
        case 24:
            return "SQLITE_FORMAT";
        case 25:
            return "SQLITE_RANGE";
        case 26:
            return "SQLITE_NOTADB";
        case 100:
            return "SQLITE_ROW";
        case 101:
            return "SQLITE_DONE";
        default:
            return "UNDEFINED";
    }
}
