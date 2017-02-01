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
#include "Logging.h"

#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

namespace
{
    class ThreadLocalRNG
    {
        private:

            pthread_key_t key;

            static void deallocate(void* p)
            {
                if (p) gsl_rng_free(static_cast<gsl_rng*>(p));
            }

        public:

            ThreadLocalRNG()
            {
                int rc = pthread_key_create(&key,
                                            &ThreadLocalRNG::deallocate);

                if (rc) CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__ 
                                                  << ": "
                                                  << strerror(rc);
            }

            ~ThreadLocalRNG()
            {
                pthread_key_delete(key);
            }

            gsl_rng* get()
            {
                gsl_rng* engine = static_cast<gsl_rng*>(pthread_getspecific(key));

                if (engine) return engine;

                engine = gsl_rng_alloc(gsl_rng_mt19937);

                if (!engine) CLOG(FATAL, "LOGGER_SYS") << "Failure to allocate Random Engine";

                unsigned long seed;

                (void)(seed_from_urandom(&seed) || seed_from_time(&seed));

                gsl_rng_set(engine, seed);

                int rc = pthread_setspecific(key, engine);
                if (rc) CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                                  << ": "
                                                  << strerror(rc);

                return engine;
            }

            static bool seed_from_time(unsigned long* out)
            {
                *out = 0xc96a3ea3d89ceb52UL;
                struct timeval tm;
                gettimeofday(&tm, NULL);
                *out ^= (unsigned long)tm.tv_sec;
                *out ^= (unsigned long)tm.tv_usec << 7;
                *out ^= (unsigned long)(uintptr_t)&tm;
                *out ^= (unsigned long)getpid() << 3;
                return true;
            }

            static bool seed_from_urandom(unsigned long* out)
            {
                int fd = open("/dev/urandom", O_RDONLY);
                if (fd < 0) return false;

                if (read(fd, out, sizeof(*out)) != sizeof(*out))
                {
                    close(fd);
                    return false;
                }
                close(fd);
                return true;
            }
    };
}

gsl_rng* get_random_engine()
{
    static ThreadLocalRNG rng;
    return rng.get();
}
