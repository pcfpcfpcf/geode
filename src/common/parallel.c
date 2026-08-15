#include "parallel.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Long enough to cover the handoff between two parallel regions, short enough
   that an oversubscribed box gives the cpu back instead of burning it. */
#define SPINS_BEFORE_YIELD 4096

#define CACHE_LINE 64

/* Each worker reports done in its own slot, and the padding keeps sizeof over
   a cache line so that no two slots ever land on one. A shared counter puts
   every worker's release on the line the others are spinning on, which turns
   an idle wait into coherence traffic against the same interconnect the
   weight stream needs. */
typedef struct Worker {
    ThreadPool *pool;
    int index;
    int cpu;
    _Atomic unsigned finished_epoch;
    char pad[CACHE_LINE];
} Worker;

struct ThreadPool {
    pthread_t *threads;
    Worker *workers;
    int n_workers;

    ParallelFn fn;
    void *state;
    _Atomic unsigned epoch;
    _Atomic int stopping;
};

#define CORES_MAX 256

static int read_topology_id(int cpu, const char *field) {
    char path[128];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, field);
    FILE *file = fopen(path, "r");
    if (!file) return -1;
    int value;
    if (fscanf(file, "%d", &value) != 1) value = -1;
    fclose(file);
    return value;
}

/* Fills cpus with one logical cpu per physical core and returns how many there
   were, or 0 when the topology is not readable. */
static int physical_cores(int *cpus, int max) {
    long logical = sysconf(_SC_NPROCESSORS_ONLN);
    if (logical < 1) return 0;

    int packages[CORES_MAX], cores[CORES_MAX];
    int n_cores = 0;
    for (int cpu = 0; cpu < logical && cpu < CORES_MAX && n_cores < max; cpu++) {
        int package = read_topology_id(cpu, "physical_package_id");
        int core = read_topology_id(cpu, "core_id");
        if (package < 0 || core < 0) return 0;

        int seen = 0;
        for (int i = 0; i < n_cores; i++)
            if (packages[i] == package && cores[i] == core) seen = 1;
        if (!seen) {
            packages[n_cores] = package;
            cores[n_cores] = core;
            cpus[n_cores] = cpu;
            n_cores++;
        }
    }
    return n_cores;
}

int pool_default_workers(void) {
    int cpus[CORES_MAX];
    int n_cores = physical_cores(cpus, CORES_MAX);
    if (n_cores > 0) return n_cores;
    long logical = sysconf(_SC_NPROCESSORS_ONLN);
    return logical > 0 ? (int)logical : 1;
}

/* Pinned one worker per physical core, the calling thread included -- it runs
   worker 0 and so carries its share of every region. Two workers landing on
   one core's hyperthread siblings while another core idles costs about a
   tenth of the achievable streaming bandwidth.

   Left unpinned when the caller asked for more workers than there are cores:
   there is no assignment that keeps them off each other, and the scheduler
   moving them is better than us freezing a bad one in place. */
static void pin_to_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void spin(int *spins) {
    if (++*spins < SPINS_BEFORE_YIELD) return;
    sched_yield();
    *spins = 0;
}

static void *pool_worker(void *arg) {
    Worker *worker = arg;
    ThreadPool *pool = worker->pool;
    unsigned seen = 0;

    pin_to_cpu(worker->cpu);
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
        atomic_store_explicit(&worker->finished_epoch, epoch,
                              memory_order_release);
    }
}

ThreadPool *pool_start(int n_workers) {
    if (n_workers < 1) n_workers = 1;

    ThreadPool *pool = calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->n_workers = n_workers;
    atomic_init(&pool->epoch, 0);
    atomic_init(&pool->stopping, 0);

    int cpus[CORES_MAX];
    int n_cores = physical_cores(cpus, CORES_MAX);
    int pinning = n_cores >= n_workers;

    int n_spawned = n_workers - 1;
    if (n_spawned == 0) {
        if (pinning) pin_to_cpu(cpus[0]);
        return pool;
    }

    pool->threads = malloc((size_t)n_spawned * sizeof *pool->threads);
    pool->workers = calloc((size_t)n_spawned, sizeof *pool->workers);
    if (!pool->threads || !pool->workers) {
        pool_stop(pool);
        return NULL;
    }
    for (int i = 0; i < n_spawned; i++) {
        pool->workers[i].pool = pool;
        pool->workers[i].index = i + 1;
        pool->workers[i].cpu = pinning ? cpus[i + 1] : -1;
        atomic_init(&pool->workers[i].finished_epoch, 0);
        if (pthread_create(&pool->threads[i], NULL, pool_worker,
                           &pool->workers[i])) {
            pool->n_workers = i + 1;
            break;
        }
    }
    if (pinning) pin_to_cpu(cpus[0]);
    return pool;
}

void pool_run(ThreadPool *pool, ParallelFn fn, void *state) {
    if (pool->n_workers == 1) {
        fn(state, 0, 1);
        return;
    }
    pool->fn = fn;
    pool->state = state;
    unsigned epoch =
        atomic_fetch_add_explicit(&pool->epoch, 1, memory_order_release) + 1;

    fn(state, 0, pool->n_workers);

    for (int i = 0; i < pool->n_workers - 1; i++) {
        int spins = 0;
        while (atomic_load_explicit(&pool->workers[i].finished_epoch,
                                    memory_order_acquire) != epoch)
            spin(&spins);
    }
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
