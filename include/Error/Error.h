/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ERROR_H
#define ERROR_H

#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>

#define REPORT_ERROR(ErrMsg) throw Error(ErrMsg, __FILE__, __LINE__)

class Error : public std::exception
{
    private:

        std::string _errMsg;
        std::string _file;
        mutable std::string _totalMsg;

        int _line;

    public:

        Error(const std::string& errMsg,
              const std::string& file,
              const int line);

        friend std::ostream& operator<<(std::ostream& os, const Error& error);

        const char* what() const noexcept override;
};

#endif // ERROR_H
