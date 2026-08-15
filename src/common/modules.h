#ifndef GEODE_MODULES_H
#define GEODE_MODULES_H

int probe_main(int argc, char **argv);
int manifest_main(int argc, char **argv);
int planner_main(int argc, char **argv);
int exec_main(int argc, char **argv);

/* Which subcommands exec_main answers to, so the top-level dispatcher does not
   repeat their names. */
int exec_handles(const char *name);

#endif
