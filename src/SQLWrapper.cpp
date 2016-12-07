// clang-format

#include "SQLWrapper.h"

#include <utility>

namespace sql
{
static bool isOK(int code)
{
    return code == SQLITE_OK || code == SQLITE_DONE || code == SQLITE_ROW;
}

Exception::Exception(int _code, const char* _msg)
    : code(_code)
{
    snprintf(description, sizeof(description), "SQLite3 error %d: %s", _code, _msg);
}

Exception::Exception(int _code)
    : Exception(_code, sqlite3_errstr(_code))
{
}

Exception::Exception(sqlite3* db)
    : Exception(sqlite3_errcode(db), sqlite3_errmsg(db))
{
}

DB::DB(const char* path, int flags)
{
    if (flags == 0)
        flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE; // default behavior
    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(path, &db, flags, nullptr);
    if (rc != SQLITE_OK)
        throw Exception(rc);
    handle = new InternalHandle;
    handle->sqlDB = db;
    handle->refcount = 1;
}

void DB::clear()
{
    if (!handle)
        return;
    if (--handle->refcount <= 0)
    {
        sqlite3_close(handle->sqlDB);
        delete handle;
    }
}

void DB::exec(const char* cmd)
{
    int rc = sqlite3_exec(getNativeHandle(), cmd, nullptr, nullptr, nullptr);
    if (!isOK(rc))
        throw Exception(getNativeHandle());
}

void Statement::check(int rc)
{
    if (!isOK(rc))
        throw Exception(db.getNativeHandle());
}
Statement::Statement(const char* command, int nByte, DB& _db)
    : db(_db)
    , stmt(nullptr)
{
    int rc = sqlite3_prepare_v2(db.getNativeHandle(), command, nByte, &stmt, nullptr);
    check(rc);
}
Statement::~Statement()
{
    if (stmt)
        sqlite3_finalize(stmt);
}
bool Statement::step()
{
    int rc = sqlite3_step(stmt);
    if (rc == SQLITE_OK || rc == SQLITE_DONE)
        return false;
    if (rc == SQLITE_ROW)
        return true;
    throw Exception(db.getNativeHandle());
}
}
