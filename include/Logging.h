/*******************************************************************************
 * Author: Mingxu Hu
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

#include "easylogging++.h"

using namespace std;

static el::Configurations loggerConf;

static void loggerInit(el::Configurations conf,
                       const char loggerName[],
                       const char loggerFile[])
{
    conf.setToDefault();

    conf.set(el::Level::Global,
             el::ConfigurationType::Filename,
             loggerFile);

    conf.set(el::Level::Info,
             el::ConfigurationType::ToFile,
             "true");

    conf.set(el::Level::Info,
             el::ConfigurationType::ToStandardOutput,
             "false");

    el::Loggers::reconfigureLogger(loggerName, conf);
};

static void loggerInit()
{
    loggerInit(loggerConf, "LOGGER_SYS", "LOGGER_SYS.log");
    loggerInit(loggerConf, "LOGGER_INIT", "LOGGER_INIT.log");
    loggerInit(loggerConf, "LOGGER_ROUND", "LOGGER_ROUND.log");
    loggerInit(loggerConf, "LOGGER_COMPARE", "LOGGER_COMPARE.log");
    loggerInit(loggerConf, "LOGGER_RECO", "LOGGER_RECO.log");
};

static void loggerInit(int argc, char* argv[])
{
    string logger(argv[0]);
    loggerInit(loggerConf, "LOGGER_SYS", (logger + ".log").c_str());
    loggerInit(loggerConf, "LOGGER_INIT", (logger + ".log").c_str());
    loggerInit(loggerConf, "LOGGER_ROUND", (logger + ".log").c_str());
    loggerInit(loggerConf, "LOGGER_COMPARE", (logger + ".log").c_str());
    loggerInit(loggerConf, "LOGGER_RECO", (logger + ".log").c_str());
}

#endif // LOGGING_H
