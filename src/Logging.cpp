#include "Logging.h"
#include "Utils.h"

static el::Configurations conf;

void loggerInit(int argc, const char* const * argv)
{
    using std::string;

    string loggerFileName;

    char buf[FILE_NAME_LENGTH];
    getcwd(buf, sizeof(buf));
    loggerFileName = buf;
    string appname(argv[0]);

    if (appname.rfind('/') == std::string::npos)
        loggerFileName += '/' + appname + ".log";
    else
        loggerFileName += appname.substr(appname.rfind('/')) + ".log";

    std::cout << "Log File will be Put: " << loggerFileName << std::endl;

    conf.setToDefault();
    conf.set(el::Level::All,
             el::ConfigurationType::Filename,
             loggerFileName);
    conf.set(el::Level::Info,
             el::ConfigurationType::ToFile,
             "true");
    conf.set(el::Level::Info,
             el::ConfigurationType::ToStandardOutput,
             "false");
    el::Loggers::setDefaultConfigurations(conf, true);

    const char* loggerNames[] = {"LOGGER_SYS","LOGGER_INIT","LOGGER_ROUND","LOGGER_COMPARE",
                                 "LOGGER_RECO","LOGGER_MPI","LOGGER_FFT"};
    for (size_t i = 0; i < sizeof(loggerNames) / sizeof(*loggerNames); ++i) {
        el::Loggers::getLogger(loggerNames[i]); // Force creation of loggers
    }
}

