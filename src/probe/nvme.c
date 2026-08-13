#include "nvme.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#define READ_CHUNK (1u << 20)
#define PROBE_FILE_BYTES (128ull << 20)

typedef struct {
    unsigned major, minor;
} DevNum;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int is_real_drive(const char *name) {
    static const char *skip[] = {"loop", "dm-", "sr", "ram", "zram",
                                 "md",   "nbd", "fd", "vc"};
    for (size_t i = 0; i < sizeof skip / sizeof *skip; i++) {
        if (!strncmp(name, skip[i], strlen(skip[i]))) return 0;
    }
    return 1;
}

static void drive_devnums(const char *name, DevNum *devs, int max, int *count) {
    char base[512];
    snprintf(base, sizeof base, "/sys/block/%s", name);
    char buf[1024];
    DevNum d;
    *count = 0;
    snprintf(buf, sizeof buf, "%s/dev", base);
    FILE *f = fopen(buf, "r");
    if (f) {
        if (fscanf(f, "%u:%u", &d.major, &d.minor) == 2 && *count < max)
            devs[(*count)++] = d;
        fclose(f);
    }
    DIR *dir = opendir(base);
    struct dirent *e;
    size_t nlen = strlen(name);
    while ((e = readdir(dir)) && *count < max) {
        if (!strncmp(e->d_name, name, nlen) && strcmp(e->d_name, name)) {
            snprintf(buf, sizeof buf, "%s/%s/dev", base, e->d_name);
            f = fopen(buf, "r");
            if (f) {
                if (fscanf(f, "%u:%u", &d.major, &d.minor) == 2)
                    devs[(*count)++] = d;
                fclose(f);
            }
        }
    }
    closedir(dir);
}

static int on_drive(const char *dirpath, const DevNum *devs, int ndevs) {
    struct stat st;
    if (stat(dirpath, &st)) return 0;
    unsigned m = major(st.st_dev), n = minor(st.st_dev);
    for (int i = 0; i < ndevs; i++)
        if (devs[i].major == m && devs[i].minor == n) return 1;
    return 0;
}

static int writable_dir_on_drive(const DevNum *devs, int ndevs, char *out,
                                 size_t outsz) {
    const char *home = getenv("HOME");
    static const char *candidates[] = {"/tmp", "/var/tmp", "/"};
    int n = 0;
    if (home) {
        if (access(home, W_OK | X_OK) == 0 && on_drive(home, devs, ndevs))
            snprintf(out, outsz, "%s", home);
        else
            n = 1;
    } else {
        n = 1;
    }
    if (n) {
        for (size_t i = 0; i < sizeof candidates / sizeof *candidates; i++) {
            if (access(candidates[i], W_OK | X_OK) == 0 &&
                on_drive(candidates[i], devs, ndevs)) {
                snprintf(out, outsz, "%s", candidates[i]);
                n = 0;
                break;
            }
        }
    }
    return n == 0;
}

static double bench_seq_read(int fd, unsigned long long byte_limit,
                             double seconds, int loop) {
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, READ_CHUNK)) return 0.0;
    double t0 = now_s(), t1 = t0;
    unsigned long long total = 0;
    while (t1 - t0 < seconds) {
        ssize_t got = read(fd, buf, READ_CHUNK);
        if (got < 0) break;
        if (got == 0) {
            if (!loop || lseek(fd, 0, SEEK_SET)) break;
            continue;
        }
        total += (unsigned long long)got;
        if (byte_limit && total >= byte_limit) break;
        t1 = now_s();
    }
    free(buf);
    return total ? (double)total / (t1 - t0) : 0.0;
}

static double measure_raw_device(const char *name, unsigned long long capacity,
                                 double seconds) {
    char path[64];
    snprintf(path, sizeof path, "/dev/%s", name);
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return 0.0;
    unsigned long long limit = capacity < 4ull << 30 ? capacity : 4ull << 30;
    double bw = bench_seq_read(fd, limit, seconds, 0);
    close(fd);
    return bw;
}

static double measure_file_on_fs(const char *dirpath, double seconds) {
    struct statvfs vfs;
    if (statvfs(dirpath, &vfs)) return 0.0;
    unsigned long long free_bytes = (unsigned long long)vfs.f_bavail * vfs.f_frsize;
    if (free_bytes < PROBE_FILE_BYTES + (64ull << 20)) return 0.0;

    char path[1024];
    snprintf(path, sizeof path, "%s/.geode-probe-%ld", dirpath, (long)getpid());
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_DIRECT, 0600);
    if (fd < 0) return 0.0;

    void *buf = NULL;
    if (posix_memalign(&buf, 4096, READ_CHUNK)) {
        close(fd);
        unlink(path);
        return 0.0;
    }
    memset(buf, 0x5a, READ_CHUNK);

    unsigned long long written = 0;
    while (written < PROBE_FILE_BYTES) {
        ssize_t n = write(fd, buf, READ_CHUNK);
        if (n <= 0) break;
        written += (unsigned long long)n;
    }
    if (written < PROBE_FILE_BYTES) {
        free(buf);
        close(fd);
        unlink(path);
        return 0.0;
    }
    if (lseek(fd, 0, SEEK_SET)) {
        free(buf);
        close(fd);
        unlink(path);
        return 0.0;
    }
    double bw = bench_seq_read(fd, 0, seconds, 1);
    free(buf);
    close(fd);
    unlink(path);
    return bw;
}

int nvme_probe(NvmeDrive **out) {
    int count = 0, cap = 32;
    NvmeDrive *drives = calloc(cap, sizeof *drives);

    DIR *dir = opendir("/sys/block");
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (e->d_name[0] == '.' || !is_real_drive(e->d_name)) continue;
        if (count >= cap || strlen(e->d_name) >= sizeof drives[0].name) continue;

        NvmeDrive *d = &drives[count];
        snprintf(d->name, sizeof d->name, "%s", e->d_name);

        char path[320];
        snprintf(path, sizeof path, "/sys/block/%s/size", e->d_name);
        unsigned long long sectors = 0;
        FILE *f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%llu", &sectors) == 1)
                d->capacity_bytes = sectors * 512ull;
            fclose(f);
        }

        DevNum devs[64];
        int ndevs = 0;
        drive_devnums(e->d_name, devs, 64, &ndevs);

        d->read_bw_bytes_s = measure_raw_device(e->d_name, d->capacity_bytes,
                                                0.5);
        if (d->read_bw_bytes_s == 0.0) {
            char dirpath[1024] = "";
            if (writable_dir_on_drive(devs, ndevs, dirpath, sizeof dirpath))
                d->read_bw_bytes_s = measure_file_on_fs(dirpath, 0.5);
        }
        d->measured = d->read_bw_bytes_s > 0.0;
        count++;
    }
    closedir(dir);
    *out = drives;
    return count;
}

void nvme_free(NvmeDrive *drives) { free(drives); }