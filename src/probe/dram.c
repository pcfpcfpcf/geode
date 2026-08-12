#include "dram.h"

#include <dirent.h>
#include <linux/mempolicy.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define NODE_MAX 64
#define WORKERS_MAX 8
#define WORKER_BUFFER_BYTES (16u << 20)
#define LATENCY_SLOTS (1u << 19)
#define LATENCY_STRIDE 64

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static volatile uint64_t sink;

static void pin_memory_to_node(void *addr, size_t len, int node, int n_nodes) {
    if (node < NODE_MAX) {
        unsigned long mask = 1UL << node;
        syscall(__NR_mbind, addr, len, MPOL_BIND, &mask, (unsigned long)n_nodes,
                MPOL_MF_STRICT);
    }
}

static void parse_cpulist(const char *s, int *cpus, int max, int *count) {
    const char *p = s;
    while (*p && *count < max) {
        while (*p == ',' || *p == ' ' || *p == '\n') p++;
        if (!*p) break;
        int lo, hi;
        if (sscanf(p, "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi && *count < max; c++) cpus[(*count)++] = c;
        } else {
            sscanf(p, "%d", &lo);
            cpus[(*count)++] = lo;
        }
        while (*p && *p != ',') p++;
    }
}

static int read_file(const char *path, char *buf, size_t sz) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, sz - 1, f);
    buf[n] = '\0';
    fclose(f);
    return 0;
}

static int collect_nodes(int *nodes, int max, int *count) {
    const char *possible = "/sys/devices/system/node/possible";
    char buf[256];
    if (read_file(possible, buf, sizeof buf)) return -1;
    char *end;
    long highest = strtol(buf, &end, 10);
    while (end && *end != '\0') {
        end++;
        long v = strtol(end, &end, 10);
        if (v > highest) highest = v;
    }
    *count = 0;
    for (int n = 0; n <= highest && n < max; n++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d", n);
        if (access(path, F_OK) == 0) nodes[(*count)++] = n;
    }
    return 0;
}

static int node_capacity(int node, unsigned long long *bytes) {
    char path[160];
    snprintf(path, sizeof path, "/sys/devices/system/node/node%d/meminfo", node);
    char buf[4096];
    if (read_file(path, buf, sizeof buf)) return -1;
    char *line = strstr(buf, "MemTotal:");
    if (!line) return -1;
    unsigned long long kb = 0;
    if (sscanf(line, "MemTotal: %llu kB", &kb) != 1 || !kb) return -1;
    *bytes = kb * 1024ull;
    return 0;
}

static double bench_read(uint64_t *p, size_t nw, double seconds) {
    double t0 = now_s(), t1 = t0;
    long long passes = 0;
    uint64_t sum = 0;
    do {
        for (size_t i = 0; i < nw; i++) sum += p[i];
        sink = sum;
        passes++;
        t1 = now_s();
    } while (t1 - t0 < seconds);
    return (double)passes * nw * 8 / (t1 - t0);
}

static double bench_copy(uint64_t *dst, const uint64_t *src, size_t nw,
                         double seconds) {
    double t0 = now_s(), t1 = t0;
    long long passes = 0;
    do {
        memcpy(dst, src, nw * 8);
        passes++;
        t1 = now_s();
    } while (t1 - t0 < seconds);
    return (double)passes * nw * 8 / (t1 - t0);
}

typedef struct {
    int cpu;
    int node;
    int n_nodes;
    double seconds;
    uint64_t *src;
    uint64_t *dst;
    double read_bw;
    double copy_bw;
    int ok;
} Worker;

static void *dram_worker(void *arg) {
    Worker *w = arg;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    if (sched_setaffinity(0, sizeof set, &set)) {
        w->ok = 0;
        return NULL;
    }
    int words = WORKER_BUFFER_BYTES / 8;
    if (posix_memalign((void **)&w->src, 4096, WORKER_BUFFER_BYTES) ||
        posix_memalign((void **)&w->dst, 4096, WORKER_BUFFER_BYTES)) {
        w->ok = 0;
        return NULL;
    }
    pin_memory_to_node(w->src, WORKER_BUFFER_BYTES, w->node, w->n_nodes);
    pin_memory_to_node(w->dst, WORKER_BUFFER_BYTES, w->node, w->n_nodes);
    for (int i = 0; i < words; i++) w->src[i] = i;
    for (int i = 0; i < words; i++) w->dst[i] = 0;
    bench_read(w->src, words, 0.1);
    bench_copy(w->dst, w->src, words, 0.1);
    w->read_bw = bench_read(w->src, words, w->seconds);
    w->copy_bw = bench_copy(w->dst, w->src, words, w->seconds);
    w->ok = 1;
    return NULL;
}

static void build_chain(char *buf, size_t nslots) {
    size_t mask = nslots - 1;
    size_t v = 0;
    for (size_t i = 0; i < nslots; i++) {
        size_t next = (v * 5 + 1) & mask;
        *(uint64_t *)(buf + v * LATENCY_STRIDE) = next;
        v = next;
    }
}

static double bench_chain(char *buf, size_t nslots, double seconds) {
    size_t next = 0;
    double t0 = now_s(), t1 = t0;
    long long hops = 0;
    do {
        for (size_t i = 0; i < nslots; i++) {
            next = *(uint64_t *)(buf + next * LATENCY_STRIDE);
            hops++;
        }
        t1 = now_s();
    } while (t1 - t0 < seconds);
    sink = next;
    return (t1 - t0) / hops * 1e9;
}

int dram_probe(DramNode **out) {
    int nodes[NODE_MAX], n_nodes = 0, count = 0;
    if (collect_nodes(nodes, NODE_MAX, &count)) {
        *out = NULL;
        return 0;
    }
    n_nodes = count ? nodes[count - 1] + 1 : 0;
    if (n_nodes > NODE_MAX) n_nodes = NODE_MAX;

    DramNode *result = calloc(count, sizeof *result);
    for (int i = 0; i < count; i++) {
        int node = nodes[i];
        DramNode *d = &result[i];
        d->node = node;
        unsigned long long cap = 0;
        d->capacity_bytes = node_capacity(node, &cap) == 0 ? cap : 0;

        char path[160];
        snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist",
                 node);
        char buf[1024];
        int cpus[NODE_MAX], ncpu = 0;
        if (read_file(path, buf, sizeof buf)) continue;
        parse_cpulist(buf, cpus, NODE_MAX, &ncpu);
        if (ncpu == 0) continue;

        int nworkers = ncpu < WORKERS_MAX ? ncpu : WORKERS_MAX;
        Worker *workers = calloc(nworkers, sizeof *workers);
        pthread_t *threads = malloc(nworkers * sizeof *threads);
        for (int w = 0; w < nworkers; w++) {
            workers[w].cpu = cpus[w];
            workers[w].node = node;
            workers[w].n_nodes = n_nodes;
            workers[w].seconds = 1.0;
            pthread_create(&threads[w], NULL, dram_worker, &workers[w]);
        }
        int ok = 1;
        for (int w = 0; w < nworkers; w++) {
            pthread_join(threads[w], NULL);
            if (!workers[w].ok) ok = 0;
        }
        if (ok) {
            for (int w = 0; w < nworkers; w++) {
                d->read_bw_bytes_s += workers[w].read_bw;
                d->copy_bw_bytes_s += workers[w].copy_bw;
            }
        }
        free(workers);
        free(threads);

        char *chain = malloc(LATENCY_SLOTS * LATENCY_STRIDE);
        pin_memory_to_node(chain, LATENCY_SLOTS * LATENCY_STRIDE, node,
                           n_nodes);
        build_chain(chain, LATENCY_SLOTS);
        d->load_latency_ns = bench_chain(chain, LATENCY_SLOTS, 0.4);
        free(chain);
    }
    *out = result;
    return count;
}

void dram_free(DramNode *nodes) { free(nodes); }