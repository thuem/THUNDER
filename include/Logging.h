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


// Compatibility settings for
namespace el = easyloggingpp;
#define INITIALIZE_EASYLOGGINGPP _INITIALIZE_EASYLOGGINGPP

void loggerInit(int argc, const char* const * argv);

#endif // LOGGING_H
