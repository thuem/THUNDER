// clang-format

#include "SQLWrapper.h"

namespace sql {

static bool isOK(int code)
{
    return code == SQLITE_OK || code == SQLITE_DONE || code == SQLITE_ROW;
}

SQLite3Exception::SQLite3Exception(int _code, const char* _msg)
    : code(_code)
{
    snprintf(description, sizeof(description), "SQLite3 Error %d: %s", _code, _msg);
}

SQLite3Exception::SQLite3Exception(int _code)
    : SQLite3Exception(_code, sqlite3_errstr(_code))
{
}

SQLite3Exception::SQLite3Exception(sqlite3* db)
    : SQLite3Exception(sqlite3_errcode(db), sqlite3_errmsg(db))
{
}

SQLite3DB::SQLite3DB(const char* path, int flags)
{
    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(path, &db, flags, nullptr);
    if (rc != SQLITE_OK)
        throw SQLite3Exception(rc);
    handle.reset(db, &sqlite3_close);
}

SQLite3DB::~SQLite3DB()
{
}

void SQLite3DB::exec(const char* cmd)
{
    int rc = sqlite3_exec(handle.get(), cmd, nullptr, nullptr, nullptr);
    if (!isOK(rc))
        throw SQLite3Exception(handle.get());
}

void SQLite3Statement::check(int rc)
{
    if (!isOK(rc))
        throw SQLite3Exception(db.getNativeHandle());
}
SQLite3Statement::SQLite3Statement(const char* command, int nByte, SQLite3DB _db)
    : db(_db)
    , stmt(nullptr, &sqlite3_finalize)
{
    sqlite3_stmt* tmp = nullptr;
    int rc = sqlite3_prepare_v2(db.getNativeHandle(), command, nByte, &tmp, nullptr);
    check(rc);
    stmt.reset(tmp);
}
SQLite3Statement::~SQLite3Statement()
{
}
bool SQLite3Statement::step()
{
    int rc = sqlite3_step(stmt.get());
    if (rc == SQLITE_OK || rc == SQLITE_DONE)
        return false;
    if (rc == SQLITE_ROW)
        return true;
    throw SQLite3Exception(db.getNativeHandle());
}
}
