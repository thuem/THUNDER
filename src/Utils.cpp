#include "Utils.h"

#include <stdexcept>
#include <regex.h>
#include <sys/types.h>

namespace {
class Regex {
private:
    regex_t regex;

public:
    Regex(const char* pattern, int flags)
    {
        int rc = regcomp(&regex, pattern, flags);
        if (rc != 0) {
            char buf[3000];
            regerror(rc, nullptr, buf, sizeof(buf));
            throw std::invalid_argument(buf);
        }
    }

    regex_t* getInternal()
    {
        return &regex;
    }

    ~Regex()
    {
        regfree(&regex);
    }
};
}

bool regexMatches(const char* str, const char* pattern)
{
    Regex regex(pattern, REG_EXTENDED | REG_NOSUB);
    int rc = regexec(regex.getInternal(), str, 0, nullptr, 0);
    if (rc == 0)
        return true;
    if (rc == REG_NOMATCH)
        return false;

    char buf[3000];
    regerror(rc, regex.getInternal(), buf, sizeof(buf));
    throw std::runtime_error(buf);
}
