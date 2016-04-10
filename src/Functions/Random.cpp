#include <Functions/Random.h>

#include <stdexcept>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

static pthread_key_t key;

static int module_init()
{
    pthread_key_create(&key, [] (void* p) { 
        if (p) 
            gsl_rng_free(static_cast<gsl_rng*>(p)); 
    });
    return 0;
}

static const int MODULE_INITED = module_init();

static unsigned long urandom()
{
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("No /dev/urandom");
    unsigned long res;
    if (read(fd, &res, sizeof(res)) != sizeof(res)) {
        close(fd);
        throw std::runtime_error("Insufficient read");
    }
    close(fd);
    return res;
}


gsl_rng* get_random_engine()
{
    (void)MODULE_INITED; // Supress warnings about unused variable
    auto engine = static_cast<gsl_rng*>(pthread_getspecific(key));
    if (engine)
        return engine;
    engine = gsl_rng_alloc(gsl_rng_mt19937);
    if (!engine)
        throw std::runtime_error("Failure to allocate random engine");
    gsl_rng_set(engine, urandom());
    if (!pthread_setspecific(key, engine)) {
        gsl_rng_free(engine);
        throw std::runtime_error("Failure to set thread local storage");
    }
    return engine;
}
