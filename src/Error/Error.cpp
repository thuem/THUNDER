/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Error.h"
#include <sstream>

Error::Error(const std::string& errMsg,
             const std::string& file,
             const int line)
{
    _errMsg = errMsg;
    _file = file;
    _line = line;
}

std::ostream& operator<<(std::ostream& os, const Error& error)
{
    os << "ERROR: " << error._errMsg << std::endl
       << "File: " << error._file <<std::endl
       << "Line: " << error._line <<std::endl;
    return os;
}

const char* Error::what() const throw()
{
    if (_totalMsg.empty())
    {
        std::stringstream ss;
        ss << *this;
        _totalMsg = ss.str();
    }
    return _totalMsg.c_str();
}
