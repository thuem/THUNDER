#ifndef UTILS_H
#define UTILS_H

#include <boost/container/vector.hpp>
#include <string>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <getopt.h>
#include <iostream>

#include "Precision.h"

using std::string;
using std::cout;
using std::endl;
using boost::container::vector;

bool regexMatches(const char* str, const char* pattern);

const char* getTempDirectory(void);

void optionCheck(char option[], int size, const struct option long_options[]);

#endif // UTILS_H
