#include "parallel.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>

/* Long enough to cover the handoff between two parallel regions, short enough
   that an oversubscribed box gives the cpu back instead of burning it. */
#define SPINS_BEFORE_YIELD 4096

struct ThreadPool {
    pthread_t *threads;
    struct Worker *workers;
    int n_workers;

    ParallelFn fn;
    void *state;
    _Atomic unsigned epoch;
    _Atomic int n_finished;
    _Atomic int stopping;
};

typedef struct Worker {
    ThreadPool *pool;
    int index;
} Worker;

static void spin(int *spins) {
    if (++*spins < SPINS_BEFORE_YIELD) return;
    sched_yield();
    *spins = 0;
}

static void *pool_worker(void *arg) {
    Worker *worker = arg;
    ThreadPool *pool = worker->pool;
    unsigned seen = 0;

    for (;;) {
        int spins = 0;
        unsigned epoch;
        while ((epoch = atomic_load_explicit(&pool->epoch,
                                             memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&pool->stopping, memory_order_acquire))
                return NULL;
            spin(&spins);
        }
        seen = epoch;
        if (atomic_load_explicit(&pool->stopping, memory_order_acquire))
            return NULL;

        pool->fn(pool->state, worker->index, pool->n_workers);
        atomic_fetch_add_explicit(&pool->n_finished, 1, memory_order_release);
    }
}

ThreadPool *pool_start(int n_workers) {
    if (n_workers < 1) n_workers = 1;

    ThreadPool *pool = calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->n_workers = n_workers;
    atomic_init(&pool->epoch, 0);
    atomic_init(&pool->n_finished, 0);
    atomic_init(&pool->stopping, 0);

    int n_spawned = n_workers - 1;
    if (n_spawned == 0) return pool;

    pool->threads = malloc((size_t)n_spawned * sizeof *pool->threads);
    pool->workers = malloc((size_t)n_spawned * sizeof *pool->workers);
    if (!pool->threads || !pool->workers) {
        pool_stop(pool);
        return NULL;
    }
    for (int i = 0; i < n_spawned; i++) {
        pool->workers[i].pool = pool;
        pool->workers[i].index = i + 1;
        if (pthread_create(&pool->threads[i], NULL, pool_worker,
                           &pool->workers[i])) {
            pool->n_workers = i + 1;
            break;
        }
    }
    return pool;
}

void pool_run(ThreadPool *pool, ParallelFn fn, void *state) {
    if (pool->n_workers == 1) {
        fn(state, 0, 1);
        return;
    }
    pool->fn = fn;
    pool->state = state;
    atomic_store_explicit(&pool->n_finished, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->epoch, 1, memory_order_release);

    fn(state, 0, pool->n_workers);

    int spins = 0;
    while (atomic_load_explicit(&pool->n_finished, memory_order_acquire) <
           pool->n_workers - 1)
        spin(&spins);
}

void pool_stop(ThreadPool *pool) {
    if (!pool) return;
    atomic_store_explicit(&pool->stopping, 1, memory_order_release);
    atomic_fetch_add_explicit(&pool->epoch, 1, memory_order_release);
    if (pool->threads)
        for (int i = 0; i < pool->n_workers - 1; i++)
            pthread_join(pool->threads[i], NULL);
    free(pool->threads);
    free(pool->workers);
    free(pool);
}

int pool_workers(const ThreadPool *pool) { return pool->n_workers; }
