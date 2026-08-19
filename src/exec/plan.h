#ifndef GEODE_PLAN_H
#define GEODE_PLAN_H

#include <stddef.h>
#include <stdio.h>

#define PLAN_NAME_MAX 24
#define PLAN_MAX_PLACEMENTS 8
#define PLAN_MAX_CANDIDATES 8

typedef struct {
    char component[PLAN_NAME_MAX];
    char tier[PLAN_NAME_MAX];
    char policy[PLAN_NAME_MAX];
} Placement;

/* A strategy the planner scored. `scorable` is 0 for one it ruled out, in
   which case only `reason` is filled. */
typedef struct {
    char strategy[PLAN_NAME_MAX];
    int scorable;
    char reason[160];
    double decode_tok_s[2];
    double prefill_tok_s;
} Candidate;

typedef struct {
    char strategy[PLAN_NAME_MAX];
    int n_ctx;
    int batch;
    double predicted_tok_s[2];
    double predicted_prefill_tok_s;
    char numa_policy[PLAN_NAME_MAX];
    int numa_replicas;
    Placement placements[PLAN_MAX_PLACEMENTS];
    int n_placements;
    Candidate candidates[PLAN_MAX_CANDIDATES];
    int n_candidates;
} Plan;

/* What runs on a box the planner has never seen: everything on DRAM, and a
   context the caller sizes from the prompt. */
void plan_default(Plan *plan);

int plan_load(Plan *plan, const char *path, char *err, size_t errsz);
void plan_write(FILE *out, const Plan *plan);
void plan_print(const Plan *plan);

/* The candidate the planner scored for `strategy`, or NULL if it scored none.
   Carries the predicted decode range that strategy was promised. */
const Candidate *plan_candidate(const Plan *plan, const char *strategy);

#endif
