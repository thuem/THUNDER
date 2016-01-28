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

#define REPORT_ERROR(ErrMsg) throw Error(ErrMsg, __FILE__, __LINE__)

class Error
{
    private:

        std::string _errMsg;
        std::string _file;
        int _line;

    public:

        Error(const std::string& errMsg,
              const std::string& file,
              const int line);

        friend std::ostream& operator<<(std::ostream& os, Error& error);
};

#endif // ERROR_H
