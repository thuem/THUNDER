// clang-format

#include "SQLWrapper.h"

namespace sql {

static bool isOK(int code)
{
    return code == SQLITE_OK || code == SQLITE_DONE || code == SQLITE_ROW;
}

Exception::Exception(int _code, const char* _msg)
    : code(_code)
{
    snprintf(description, sizeof(description), " Error %d: %s", _code, _msg);
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
    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(path, &db, flags, nullptr);
    if (rc != SQLITE_OK)
        throw Exception(rc);
    handle.reset(db, &sqlite3_close);
}

DB::~DB()
{
}

void DB::exec(const char* cmd)
{
    int rc = sqlite3_exec(handle.get(), cmd, nullptr, nullptr, nullptr);
    if (!isOK(rc))
        throw Exception(handle.get());
}

void Statement::check(int rc)
{
    if (!isOK(rc))
        throw Exception(db.getNativeHandle());
}
Statement::Statement(const char* command, int nByte, DB _db)
    : db(_db)
    , stmt(nullptr, &sqlite3_finalize)
{
    sqlite3_stmt* tmp = nullptr;
    int rc = sqlite3_prepare_v2(db.getNativeHandle(), command, nByte, &tmp, nullptr);
    check(rc);
    stmt.reset(tmp);
}
Statement::~Statement()
{
}
bool Statement::step()
{
    int rc = sqlite3_step(stmt.get());
    if (rc == SQLITE_OK || rc == SQLITE_DONE)
        return false;
    if (rc == SQLITE_ROW)
        return true;
    throw Exception(db.getNativeHandle());
}
}
