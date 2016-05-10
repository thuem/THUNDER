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

static int module_init()
{
    pthread_key_create(&key,
                       [](void* p)
                       {
                           if (p) gsl_rng_free(static_cast<gsl_rng*>(p));
                       });
    return 0;
}

static const int MODULE_INITED = module_init();

static unsigned long urandom()
{
    int fd = open("/dev/urandom", O_RDONLY);

    if (fd < 0) LOG(FATAL) << "No /dev/urandom";

    unsigned long res;
    if (read(fd, &res, sizeof(res)) != sizeof(res))
    {
        close(fd);
        LOG(FATAL) << "Insufficient Read";
    }

    close(fd);
    return res;
}

gsl_rng* get_random_engine()
{
    // Supress warnings about unused variable
    (void)MODULE_INITED; 

    auto engine = static_cast<gsl_rng*>(pthread_getspecific(key));

    if (engine) return engine;

    engine = gsl_rng_alloc(gsl_rng_mt19937);

    if (!engine) LOG(FATAL) << "Failure to Allocate Random Engine";

    gsl_rng_set(engine, urandom());

    if (pthread_setspecific(key, engine) != 0)
    {
        gsl_rng_free(engine);
        LOG(FATAL) << "Failure to Set Thread Local Storage";
    }

    return engine;
}
