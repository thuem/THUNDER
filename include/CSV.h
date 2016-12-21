#pragma once
#ifndef THUEM_CSV_H
#define THUEM_CSV_H

#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <string>

enum CSVParseState
{
    CSVParseSuccess,
    CSVParseTooManyFields,
    CSVParseTooFewFields,
    CSVParseOverflow,
    CSVParseTypeMismatch,
    CSVParseInvalidFormatString
};

/**
 * Parse a single line in a CSV file. Quoting and escaping are not supported.
 * @param buffer string buffer
 * @param length string length
 * @param format format specifiers consisting of 'd' (int) 'l' (long) 's' (std::string) or 'f' (double)
 * @param ... Variable args as pointers to variables to store the results
 * @return Parse state
 *
 * @example
 * int a; long b; std::string c; double d;
 * const char* line = "2,345308542354,Hello world,0.125\n";
 * parseCSVLine(line, strlen(line), "dlsf", &a, &b, &c, &d);
 * // a == 2; b == 345308542354L; c == "Hello world"; d == 0.125;
 */
CSVParseState parseCSVLine(const char* buffer, size_t length, const char* format, ...);

// A null terminated buffer, same as above
CSVParseState parseCSVLine(const char* buffer, const char* format, ...);

CSVParseState vParseCSVLine(const char* buffer, size_t length, const char* format, va_list ap);

#endif //THUEM_CSV_H
