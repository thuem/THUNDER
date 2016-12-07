/*******************************************************************************
 * Author: Siyuan Ren, Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Random.h"

static pthread_key_t key;

static void tls_deallocate(void* p)
{
    if (p)
        gsl_rng_free(static_cast<gsl_rng*>(p));
}

static int module_init()
{
    pthread_key_create(&key, &tls_deallocate);
    return 0;
}

static const int MODULE_INITED = module_init();

static unsigned long urandom()
{
    int fd = open("/dev/urandom", O_RDONLY);

    if (fd < 0) CLOG(FATAL, "LOGGER_SYS") << "No /dev/urandom";

    unsigned long res;
    if (read(fd, &res, sizeof(res)) != sizeof(res))
    {
        close(fd);
        CLOG(FATAL, "LOGGER_SYS") << "Insufficient Read";
    }

    close(fd);
    return res;
}

gsl_rng* get_random_engine()
{
    // Supress warnings about unused variable
    (void)MODULE_INITED; 

    gsl_rng* engine = static_cast<gsl_rng*>(pthread_getspecific(key));

    if (engine) return engine;

    engine = gsl_rng_alloc(gsl_rng_mt19937);

    if (!engine) CLOG(FATAL, "LOGGER_SYS") << "Failure to Allocate Random Engine";

    gsl_rng_set(engine, urandom());

    if (pthread_setspecific(key, engine) != 0)
    {
        gsl_rng_free(engine);
        CLOG(FATAL, "LOGGER_SYS") << "Failure to Set Thread Local Storage";
    }

    return engine;
}
