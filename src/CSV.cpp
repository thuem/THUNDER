#include "CSV.h"

#include <stdexcept>
#include <regex.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <limits>

CSVParseState parseCSVLine(const char* buffer, size_t length, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    CSVParseState rc = vParseCSVLine(buffer, length, format, ap);
    va_end(ap);
    return rc;
}

CSVParseState parseCSVLine(const char* buffer, const char* format, ...)
{
    va_list ap;
    va_start(ap, format);
    CSVParseState rc = vParseCSVLine(buffer, strlen(buffer), format, ap);
    va_end(ap);
    return rc;
}

CSVParseState vParseCSVLine(const char* buffer, size_t length, const char* format, va_list ap)
{
    size_t last_index = 0, format_string_index = 0;
    for (size_t i = 0; i <= length; ++i) {
        if (i >= length || buffer[i] == ',' || buffer[i] == '\n') {
            if (last_index < i) {
                switch (format[format_string_index++]) {
                    case '\0':
                        return CSVParseTooManyFields;
                    case 's':
                    {
                        std::string* s = va_arg(ap, std::string*);
                        s->assign(buffer + last_index, buffer + i);
                        break;
                    }
                    case 'd':
                    {
                        int* s = va_arg(ap, int*);
                        char* end_ptr;
                        errno = 0;
                        long value = strtol(buffer + last_index, &end_ptr, 0);
                        if (errno == ERANGE
                            || value < std::numeric_limits<int>::min()
                            || value > std::numeric_limits<int>::max())
                            return CSVParseOverflow;
                        if (end_ptr != buffer + i)
                            return CSVParseTypeMismatch;
                        *s = static_cast<int>(value);
                        break;
                    }
                    case 'l':
                    {
                        long* s = va_arg(ap, long*);
                        char* end_ptr;
                        errno = 0;
                        long value = strtol(buffer + last_index, &end_ptr, 0);
                        if (errno == ERANGE)
                            return CSVParseOverflow;
                        if (end_ptr != buffer + i)
                            return CSVParseTypeMismatch;
                        *s = value;
                        break;
                    }
                    case 'f':
                    {
                        double* s = va_arg(ap, double*);
                        char* end_ptr;
                        errno = 0;
                        double value = strtod(buffer + last_index, &end_ptr);
                        if (errno == ERANGE)
                            return CSVParseOverflow;
                        if (end_ptr != buffer + i)
                            return CSVParseTypeMismatch;
                        *s = value;
                        break;
                    }
                    default:
                        return CSVParseInvalidFormatString;
                }
            }
            last_index = i + 1;
        }
    }
    return CSVParseSuccess;
}
