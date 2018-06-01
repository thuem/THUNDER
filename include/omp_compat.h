// Compatibility header to allow build with or without OpenMP
#pragma once

#ifdef _OPENMP
#include <omp.h>
#else

#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

typedef pthread_mutex_t omp_lock_t;

inline void omp_init_lock(omp_lock_t* lock) { if (pthread_mutex_init(lock, NULL)) abort();}
inline void omp_destroy_lock(omp_lock_t* lock) { if (pthread_mutex_destroy(lock)) abort();}
inline void omp_set_lock(omp_lock_t* lock) { if (pthread_mutex_lock(lock)) abort(); }
inline void omp_unset_lock(omp_lock_t* lock) { if (pthread_mutex_unlock(lock)) abort(); }
inline int omp_get_max_threads(void) { int rc = (int)sysconf(_SC_NPROCESSORS_ONLN); return rc < 1 ? 1 : rc; }
#endif
