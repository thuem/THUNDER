#ifndef UTILS_H
#define UTILS_H

#include <boost/container/vector.hpp>
#include <string>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

using std::string;
using boost::container::vector;

bool regexMatches(const char* str, const char* pattern);

const char* getTempDirectory(void);

#endif // UTILS_H
