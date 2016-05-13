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

#include "easylogging++.h"

static el::Configurations loggerAConf;
static el::Configurations loggerBConf;

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
    loggerInit(loggerAConf, "LOGGER_SYS", "LOGGER_SYS.log");
    loggerInit(loggerAConf, "LOGGER_INIT", "LOGGER_INIT.log");
    loggerInit(loggerBConf, "LOGGER_ROUND", "LOGGER_ROUND.log");
    loggerInit(loggerBConf, "LOGGER_COMPARE", "LOGGER_COMPARE.log");
};

#endif // LOGGING_H
