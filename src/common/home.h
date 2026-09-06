#ifndef GEODE_HOME_H
#define GEODE_HOME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ~/.geode/<kind>-<model basename>.json. The manifest and plan describe one
   model, so they are keyed by it: switching models must not reuse the
   previous one's cache. The probe is hardware, not model, and stays shared. */
static inline const char *geode_model_home(const char *model,
                                            const char *kind) {
    static char path[1024];
    const char *base = strrchr(model, '/');
    base = base ? base + 1 : model;
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(path, sizeof path, "%s/.geode/%s-%s.json", home, kind, base);
    return path;
}

#endif
