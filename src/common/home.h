#ifndef GEODE_HOME_H
#define GEODE_HOME_H

#include <stdio.h>
#include <stdlib.h>

/* ~/.geode/<name>, the cache the probe, manifest, planner and executor share.
   Each translation unit gets its own buffer; no caller holds the result across
   a second call. */
static inline const char *geode_home(const char *name) {
    static char path[1024];
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(path, sizeof path, "%s/.geode/%s", home, name);
    return path;
}

#endif
