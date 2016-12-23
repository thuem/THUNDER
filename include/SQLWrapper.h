// clang-format
#pragma once
#include <sqlite3.h>
#include <stddef.h>
#include <exception>
#include <string>
#include <cstddef>
#include <algorithm>

#include <boost/move/core.hpp>

namespace sql {
class Exception : public std::exception
{
private:
    char description[128];
    int code;

    void init(int code, const char* msg);
public:
    explicit Exception(int code);
    explicit Exception(int code, const char* msg);
    explicit Exception(sqlite3* db);
    int getCode() const
    {
        return code;
    }
    const char* what() const throw()
    {
        return description;
    }
    ~Exception() throw() {}
};

class DB
{
private:
    struct InternalHandle
    {
        sqlite3* sqlDB;
        ptrdiff_t refcount;
    };
    InternalHandle* handle;

public:
    explicit DB() : handle(NULL)
    {
    }
    explicit DB(const char* path, int flags);
    ~DB() { clear(); }
    void clear();
    DB(const DB& other) : handle(other.handle)
    {
        if (handle)
            ++handle->refcount;
    }
    DB& operator=(const DB& other)
    {
        if (this == &other || this->handle == other.handle)
            return *this;
        clear();
        handle = other.handle;
        if (handle)
            ++handle->refcount;
        return *this;
    }
    void swap(DB& other)
    {
        std::swap(handle, other.handle);
    }
    void exec(const char* cmd);
    void beginTransaction()
    {
        exec("begin transaction");
    }
    void endTransaction()
    {
        exec("end transaction");
    }
    void rollbackTransaction()
    {
        exec("rollback transaction");
    }
    sqlite3* getNativeHandle()
    {
        return handle == NULL ? NULL : handle->sqlDB;
    }
    bool empty() const
    {
        return handle == NULL || handle->sqlDB == NULL;
    }
};

class Statement
{
    BOOST_MOVABLE_BUT_NOT_COPYABLE(Statement)
private:
    DB db;
    sqlite3_stmt* stmt;

    void check(int code);

public:
    explicit Statement()
        : stmt(NULL)
    {
    }
    explicit Statement(const char* command, int nByte, DB& db);
    ~Statement();

    Statement(BOOST_RV_REF(Statement) other) : db(other.db), stmt(other.stmt)
    {
        other.db.clear();
        other.stmt = NULL;
    }

    Statement&operator=(BOOST_RV_REF(Statement) other)
    {
        if (this == &other)
            return *this;
        db.swap(other.db);
        std::swap(stmt, other.stmt);
        return *this;
    }

    bool empty()
    {
        return !stmt;
    }
    void reset()
    {
        sqlite3_reset(stmt);
    }
    bool step();
    sqlite3_stmt* getNativeHandle()
    {
        return stmt;
    }
    void bind_null(int index)
    {
        check(sqlite3_bind_null(getNativeHandle(), index));
    }
    void bind_int(int index, int val)
    {
        check(sqlite3_bind_int(getNativeHandle(), index, val));
    }
    void bind_text(int index, const char* str, int length, bool copy)
    {
        check(sqlite3_bind_text(getNativeHandle(), index, str, length, copy ? SQLITE_TRANSIENT : SQLITE_STATIC));
    }
    void bind_double(int index, double val)
    {
        check(sqlite3_bind_double(getNativeHandle(), index, val));
    }
    int get_int(int col)
    {
        return sqlite3_column_int(getNativeHandle(), col);
    }
    double get_double(int col)
    {
        return sqlite3_column_double(getNativeHandle(), col);
    }
    std::string get_text(int col)
    {
        const char* str = reinterpret_cast<const char*>(sqlite3_column_text(getNativeHandle(), col));
        return std::string(str, str + sqlite3_column_bytes(getNativeHandle(), col));
    }
};
}
