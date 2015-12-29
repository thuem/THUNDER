/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency: 
 * Test:
 * Execution:
 * Description: an error reporting class
 *
 * Manual:
 * ****************************************************************************/

#include "Error.h"

Error::Error(const std::string& errMsg,
             const std::string& file,
             const int line)
{
    _errMsg = errMsg;
    _file = file;
    _line = line;
}

std::ostream& operator<<(std::ostream& os, Error& error)
{
    os << "ERROR: " << error._errMsg << std::endl
       << "File: " << error._file <<std::endl
       << "Line: " << error._line <<std::endl;
}
