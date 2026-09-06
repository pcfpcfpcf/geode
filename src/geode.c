#include "home.h"
#include "modules.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int have_cached(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && st.st_size > 0;
}

static void clean_cached(void) {
    const char *dir = geode_home("");
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "no cache at %s\n", dir);
        return;
    }
    struct dirent *entry;
    while ((entry = readdir(d))) {
        const char *name = entry->d_name;
        if (strcmp(name, "probe.json") == 0 ||
            strcmp(name, "manifest.json") == 0 || strcmp(name, "plan.json") == 0 ||
            strncmp(name, "manifest-", 9) == 0 ||
            strncmp(name, "plan-", 5) == 0) {
            char path[1024];
            snprintf(path, sizeof path, "%s%s", dir, name);
            remove(path);
            printf("%s deleted\n", name);
        }
    }
    closedir(d);
}

static int run_cached_planner(const char *argv0, int nargs, char **args,
                              const char *model) {
    if (!have_cached(geode_home("probe.json")) ||
        !have_cached(geode_model_home(model, "manifest"))) {
        fprintf(stderr,
                "no cached probe or manifest for %s; run '%s %s' once to "
                "build ~/.geode\n",
                model, argv0, model);
        return 2;
    }
    enum { MAX_ARGS = 16 };
    char *planner_argv[MAX_ARGS];
    planner_argv[0] = "geode planner";
    planner_argv[1] = (char *)model;
    int n = 2;
    for (int i = 0; i < nargs && n < MAX_ARGS; i++)
        planner_argv[n++] = args[i];
    return planner_main(n, planner_argv);
}

static int run_end_to_end(const char *model) {
    int rc;
    if (!have_cached(geode_home("probe.json"))) {
        char *probe_argv[] = {"geode probe"};
        rc = probe_main(1, probe_argv);
        if (rc) return rc;
    }
    if (!have_cached(geode_model_home(model, "manifest"))) {
        char *manifest_argv[] = {"geode manifest", (char *)model};
        rc = manifest_main(2, manifest_argv);
        if (rc) return rc;
    }
    if (!have_cached(geode_model_home(model, "plan"))) {
        char *planner_argv[] = {"geode planner", (char *)model};
        rc = planner_main(2, planner_argv);
        if (rc) return rc;
    }
    return exec_cli(model);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage:\n"
            "  %s MODEL.gguf         probe -> manifest -> planner, then chat\n"
            "  %s plan MODEL.gguf [opts]   rerun the planner for a model\n"
            "                        (--context N --batch N --out plan.json)\n"
            "  %s probe|manifest|planner [args...]   run one step\n"
            "  %s exec-run MODEL.gguf [PROMPT] [N]   generate, per the plan\n"
            "  %s exec MODEL.gguf                    exec commands and checks\n"
            "  %s serve MODEL.gguf [PORT]            OpenAI-compatible API on 127.0.0.1\n"
            "\n"
            "cache: ~/.geode/probe.json, manifest-<model>.json, plan-<model>.json\n",
            argv0, argv0, argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        usage(argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "plan") == 0) {
        if (argc < 3) {
            usage(argv[0]);
            return 2;
        }
        return run_cached_planner(argv[0], argc - 3, argv + 3, argv[2]);
    }
    if (strcmp(argv[1], "probe") == 0) return probe_main(argc - 1, argv + 1);
    if (strcmp(argv[1], "manifest") == 0)
        return manifest_main(argc - 1, argv + 1);
    if (strcmp(argv[1], "planner") == 0)
        return planner_main(argc - 1, argv + 1);
    if (strcmp(argv[1], "serve") == 0) return serve_main(argc - 1, argv + 1);
    if (exec_handles(argv[1])) return exec_main(argc - 1, argv + 1);
    if (strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 ||
        strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (strcmp(argv[1], "clean") == 0){
        clean_cached();
        return 0;
    }
    if (argc != 2) {
        usage(argv[0]);
        return 2;
    }
    return run_end_to_end(argv[1]);
}