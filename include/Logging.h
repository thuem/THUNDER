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

using namespace std;

// Compatibility settings for
namespace el = easyloggingpp;
#define INITIALIZE_EASYLOGGINGPP _INITIALIZE_EASYLOGGINGPP

static el::Configurations loggerConf;

inline void loggerInit(el::Configurations conf,
                       const char loggerName[],
                       const char loggerFile[])
{
    conf.setToDefault();

    conf.set(el::Level::All,
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

inline void loggerInit()
{
    loggerInit(loggerConf, "LOGGER_SYS", "LOGGER_SYS.log");
    loggerInit(loggerConf, "LOGGER_INIT", "LOGGER_INIT.log");
    loggerInit(loggerConf, "LOGGER_ROUND", "LOGGER_ROUND.log");
    loggerInit(loggerConf, "LOGGER_COMPARE", "LOGGER_COMPARE.log");
    loggerInit(loggerConf, "LOGGER_RECO", "LOGGER_RECO.log");
    loggerInit(loggerConf, "LOGGER_MPI", "LOGGER_MPI.log");
    loggerInit(loggerConf, "LOGGER_FFT", "LOGGER_MPI.log");
};

inline void loggerInit(int argc, char* argv[])
{
    string loggername;

    char buf[FILE_NAME_LENGTH];
    getcwd(buf, sizeof(buf));
    loggername = buf;

    string appname(argv[0]);

    loggername += appname.substr(appname.rfind('/')) + ".log";

    loggerInit(loggerConf, "LOGGER_SYS", loggername.c_str());
    loggerInit(loggerConf, "LOGGER_INIT", loggername.c_str());
    loggerInit(loggerConf, "LOGGER_ROUND", loggername.c_str());
    loggerInit(loggerConf, "LOGGER_COMPARE", loggername.c_str());
    loggerInit(loggerConf, "LOGGER_RECO", loggername.c_str());
    loggerInit(loggerConf, "LOGGER_MPI", loggername.c_str());
    loggerInit(loggerConf, "LOGGER_FFT", loggername.c_str());
}

#endif // LOGGING_H
