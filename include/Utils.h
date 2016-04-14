#pragma once

#define DISABLE_COPY_MOVE(cls) public: \
                               cls(const cls&) = delete; \
                               cls(cls&&) = delete; \
                               cls& operator=(const cls&) = delete; \
                               cls& operator=(cls&&) = delete; 

#define MAKE_DEFAULT_MOVE(cls) public: \
                               cls(const cls&) = delete; \
                               cls(cls&&) = default; \
                               cls& operator=(const cls&) = delete; \
                               cls& operator=(cls&&) = default;

bool regexMatches(const char* str, const char* pattern);
