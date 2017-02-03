/*******************************************************************************
 * Author: Mingxu Hu, Hongkun Yu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef LOGGING_H
#define LOGGING_H

#include <string>
#include <unistd.h>
#include "easylogging++.h"

#include "Macro.h"

namespace el = easyloggingpp;

#define INITIALIZE_EASYLOGGINGPP _INITIALIZE_EASYLOGGINGPP

#define REPORT_ERROR(msg) CLOG(FATAL, "LOGGER_SYS") << __FILE__ \
                                                    << ", " \
                                                    << __LINE__ \
                                                    << ", " \
                                                    << __FUNCTION__ \
                                                    << ": " \
                                                    << msg;

void loggerInit(int argc, const char* const * argv);

#endif // LOGGING_H
