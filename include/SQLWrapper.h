// clang-format
#pragma once
#include <sqlite3.h>
#include <stddef.h>
#include <exception>
#include <memory>
#include <string>

namespace sql {
class Exception : public std::exception {
private:
    char description[128];
    int code;

public:
    explicit Exception(int code);
    explicit Exception(int code, const char* msg);
    explicit Exception(sqlite3* db);
    int getCode() const noexcept
    {
        return code;
    }
    const char* what() const noexcept override
    {
        return description;
    }
};

class DB {
private:
    std::shared_ptr<sqlite3> handle;

public:
    explicit DB(const char* path, int flags);
    ~DB();
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
    sqlite3* getNativeHandle() noexcept
    {
        return handle.get();
    }
};

class Statement {
private:
    DB db;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt;

    void check(int code);

public:
    explicit Statement(const char* command, int nByte, DB db);
    ~Statement();
    void reset()
    {
        sqlite3_reset(stmt.get());
    }
    bool step();
    sqlite3_stmt* getNativeHandle() noexcept
    {
        return stmt.get();
    }
    void bind_int(int index, int val)
    {
        check(sqlite3_bind_int(getNativeHandle(), index, val));
    }
    void bind_string(int index, const char* str, int length, bool copy)
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
    std::string get_string(int col)
    {
        auto str = reinterpret_cast<const char*>(sqlite3_column_text(getNativeHandle(), col));
        return std::string(str, str + sqlite3_column_bytes(getNativeHandle(), col));
    }
};
}
