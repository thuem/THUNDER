#include <Functions/Random.h>

#include <stdexcept>
#include <pthread.h>

static pthread_key_t key;

static int module_init()
{
    pthread_key_create(&key, [] (void* p) { gsl_rng_free(static_cast<gsl_rng*>(p)); });
    return 0;
}

static int inited = module_init();

gsl_rng* get_random_engine()
{
    auto engine = static_cast<gsl_rng*>(pthread_getspecific(key));
    if (engine)
        return engine;
    engine = gsl_rng_alloc(gsl_rng_mt19937);
    if (!engine)
        throw std::runtime_error("Failure to allocate random engine");
    if (!pthread_setspecific(key, engine))
        throw std::runtime_error("Failure to set thread local storage");
    return engine;
}
