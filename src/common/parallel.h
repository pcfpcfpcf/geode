#ifndef GEODE_PARALLEL_H
#define GEODE_PARALLEL_H

/* A pool that outlives the work handed to it. The probe spawns a thread per
   benchmark and joins it, which does not transfer here: one token is hundreds
   of parallel regions of a few hundred microseconds each, and thread creation
   alone would cost more than the arithmetic. */

typedef void (*ParallelFn)(void *state, int worker, int n_workers);

typedef struct ThreadPool ThreadPool;

/* Physical cores, not logical cpus: hyperthread siblings share the vector
   units that dequant saturates, so counting siblings oversubscribes and
   measurably slows the work down rather than speeding it up. */
int pool_default_workers(void);

ThreadPool *pool_start(int n_workers);
void pool_stop(ThreadPool *pool);
int pool_workers(const ThreadPool *pool);

/* Calls fn once per worker and returns when every one of them has returned.
   The caller's thread runs worker 0, so fn must not assume it is on a thread
   the pool owns. */
void pool_run(ThreadPool *pool, ParallelFn fn, void *state);

#endif
