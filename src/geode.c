#include "home.h"
#include "modules.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int have_cached(const char *name) {
    struct stat st;
    return stat(geode_home(name), &st) == 0 && st.st_size > 0;
}

static void clean_cached(){
    const char *cache[] = {"probe.json", "manifest.json", "plan.json"};
    for (int i = 0; i<3; i++){
        if (!have_cached(cache[i])){
            printf("%s doesn't exist!\n", cache[i]);
        } else{
            remove(geode_home(cache[i]));
            printf("%s deleted! \n", cache[i]);
        }
    }
}

static int run_cached_planner(const char *argv0, int nargs, char **args) {
    if (!have_cached("probe.json") || !have_cached("manifest.json")) {
        fprintf(stderr,
                "no cached model; run '%s MODEL.gguf' once to build "
                "~/.geode (probe + manifest)\n",
                argv0);
        return 2;
    }
    enum { MAX_ARGS = 16 };
    char *planner_argv[MAX_ARGS];
    planner_argv[0] = "geode planner";
    int n = 1;
    for (int i = 0; i < nargs && n < MAX_ARGS; i++)
        planner_argv[n++] = args[i];
    return planner_main(n, planner_argv);
}

static int run_end_to_end(const char *model) {
    int rc;
    if (!have_cached("probe.json")) {
        char *probe_argv[] = {"geode probe"};
        rc = probe_main(1, probe_argv);
        if (rc) return rc;
    }
    if (!have_cached("manifest.json")){
        char *manifest_argv[] = {"geode manifest", (char *)model};
        rc = manifest_main(2, manifest_argv);
        if (rc) return rc;
    }
    if (!have_cached("plan.json")) {
        char *planner_argv[] = {"geode planner"};
        rc = planner_main(1, planner_argv);
        if (rc) return rc;
    }
    return exec_cli(model);
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage:\n"
            "  %s MODEL.gguf         probe -> manifest -> planner, then chat\n"
            "  %s                    latest plan on the cached model\n"
            "  %s plan [opts]        same, with planner opts\n"
            "                        (--context N --batch N --out plan.json)\n"
            "  %s probe|manifest|planner [args...]   run one step\n"
            "  %s exec-run MODEL.gguf [PROMPT] [N]   generate, per the plan\n"
            "  %s exec MODEL.gguf                    exec commands and checks\n"
            "  %s serve MODEL.gguf [PORT]            OpenAI-compatible API on 127.0.0.1\n"
            "\n"
            "cache: ~/.geode/probe.json, manifest.json, plan.json\n",
            argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc == 1) return run_cached_planner(argv[0], 0, NULL);
    if (strcmp(argv[1], "plan") == 0)
        return run_cached_planner(argv[0], argc - 2, argv + 2);
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