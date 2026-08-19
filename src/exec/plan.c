#include "plan.h"

#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_STRATEGY "CPU-STREAM"

static void copy_str(char *dst, size_t dstsz, const JVal *obj,
                     const char *key) {
    const JVal *value = json_get(obj, key);
    dst[0] = '\0';
    if (value && value->kind == JV_STR)
        snprintf(dst, dstsz, "%s", value->str);
}

static double number(const JVal *obj, const char *key, double fallback) {
    const JVal *value = json_get(obj, key);
    return value && value->kind == JV_NUM ? value->num : fallback;
}

/* The planner emits a [low, high] band; a plan carrying anything else leaves
   the band zeroed rather than half-read. */
static void copy_band(double *band, const JVal *obj, const char *key) {
    const JVal *value = json_get(obj, key);
    band[0] = band[1] = 0;
    if (!value || value->kind != JV_ARR || value->n_items != 2) return;
    for (int i = 0; i < 2; i++)
        if (value->items[i]->kind == JV_NUM) band[i] = value->items[i]->num;
}

void plan_default(Plan *plan) {
    memset(plan, 0, sizeof *plan);
    snprintf(plan->strategy, sizeof plan->strategy, "%s", DEFAULT_STRATEGY);
}

static void load_placements(Plan *plan, const JVal *root) {
    const JVal *array = json_get(root, "placement");
    if (!array || array->kind != JV_ARR) return;
    for (int i = 0; i < array->n_items && plan->n_placements < PLAN_MAX_PLACEMENTS;
         i++) {
        const JVal *item = array->items[i];
        if (item->kind != JV_OBJ) continue;
        Placement *placement = &plan->placements[plan->n_placements++];
        copy_str(placement->component, sizeof placement->component, item,
                 "component");
        copy_str(placement->tier, sizeof placement->tier, item, "tier");
        copy_str(placement->policy, sizeof placement->policy, item, "policy");
    }
}

static void load_candidates(Plan *plan, const JVal *root) {
    const JVal *array = json_get(root, "candidates");
    if (!array || array->kind != JV_ARR) return;
    for (int i = 0; i < array->n_items && plan->n_candidates < PLAN_MAX_CANDIDATES;
         i++) {
        const JVal *item = array->items[i];
        if (item->kind != JV_OBJ) continue;
        Candidate *candidate = &plan->candidates[plan->n_candidates++];
        copy_str(candidate->strategy, sizeof candidate->strategy, item,
                 "strategy");
        copy_str(candidate->reason, sizeof candidate->reason, item, "reason");
        candidate->scorable = (int)number(item, "scorable", 0);
        copy_band(candidate->decode_tok_s, item, "decode_tok_s");
        candidate->prefill_tok_s = number(item, "prefill_tok_s", 0);
    }
}

int plan_load(Plan *plan, const char *path, char *err, size_t errsz) {
    plan_default(plan);

    char *text = json_read_file(path, err, errsz);
    if (!text) return 0;

    JVal *root = json_parse(text, err, errsz);
    free(text);
    if (!root) return 0;
    if (root->kind != JV_OBJ) {
        snprintf(err, errsz, "%s: expected a JSON object", path);
        json_free(root);
        return 0;
    }

    copy_str(plan->strategy, sizeof plan->strategy, root, "strategy");
    if (!plan->strategy[0]) {
        snprintf(err, errsz, "%s: no \"strategy\" key", path);
        json_free(root);
        return 0;
    }

    const JVal *workload = json_get(root, "workload");
    if (workload && workload->kind == JV_OBJ) {
        plan->n_ctx = (int)number(workload, "context", 0);
        plan->batch = (int)number(workload, "batch", 0);
    }
    copy_band(plan->predicted_tok_s, root, "predicted_tok_s");
    plan->predicted_prefill_tok_s =
        number(root, "predicted_prefill_tok_s", 0);
    const JVal *numa = json_get(root, "numa");
    if (numa && numa->kind == JV_OBJ) {
        copy_str(plan->numa_policy, sizeof plan->numa_policy, numa, "policy");
        plan->numa_replicas = (int)number(numa, "replicas", 0);
    }
    load_placements(plan, root);
    load_candidates(plan, root);

    json_free(root);
    return 1;
}

const Candidate *plan_candidate(const Plan *plan, const char *strategy) {
    for (int i = 0; i < plan->n_candidates; i++)
        if (strcmp(plan->candidates[i].strategy, strategy) == 0)
            return &plan->candidates[i];
    return NULL;
}

void plan_print(const Plan *plan) {
    printf("plan:     %s", plan->strategy);
    if (plan->n_ctx) printf(" @ %d ctx", plan->n_ctx);
    for (int i = 0; i < plan->n_placements; i++)
        printf("%s%s->%s", i ? ", " : " (", plan->placements[i].component,
               plan->placements[i].tier);
    printf("%s\n", plan->n_placements ? ")" : "");
}

void plan_write(FILE *out, const Plan *plan) {
    Json j;
    json_begin(&j, out);
    json_u64(&j, "schema", 1);
    json_string(&j, "strategy", plan->strategy);
    json_open(&j, "workload", 0);
    json_u64(&j, "context", (unsigned long long)plan->n_ctx);
    json_u64(&j, "batch", (unsigned long long)plan->batch);
    json_close(&j);

    json_open(&j, "placement", 1);
    for (int i = 0; i < plan->n_placements; i++) {
        const Placement *p = &plan->placements[i];
        json_open(&j, NULL, 0);
        json_string(&j, "component", p->component);
        json_string(&j, "tier", p->tier);
        json_string(&j, "policy", p->policy);
        json_close(&j);
    }
    json_close(&j);

    json_open(&j, "numa", 0);
    json_string(&j, "policy", plan->numa_policy);
    json_u64(&j, "replicas", (unsigned long long)plan->numa_replicas);
    json_close(&j);

    json_open(&j, "predicted_tok_s", 1);
    json_double(&j, NULL, plan->predicted_tok_s[0]);
    json_double(&j, NULL, plan->predicted_tok_s[1]);
    json_close(&j);
    if (plan->predicted_prefill_tok_s > 0)
        json_double(&j, "predicted_prefill_tok_s",
                    plan->predicted_prefill_tok_s);

    json_open(&j, "candidates", 1);
    for (int i = 0; i < plan->n_candidates; i++) {
        const Candidate *c = &plan->candidates[i];
        json_open(&j, NULL, 0);
        json_string(&j, "strategy", c->strategy);
        json_u64(&j, "scorable", c->scorable);
        if (c->scorable) {
            json_open(&j, "decode_tok_s", 1);
            json_double(&j, NULL, c->decode_tok_s[0]);
            json_double(&j, NULL, c->decode_tok_s[1]);
            json_close(&j);
            if (c->prefill_tok_s > 0)
                json_double(&j, "prefill_tok_s", c->prefill_tok_s);
        } else {
            json_string(&j, "reason", c->reason);
        }
        json_close(&j);
    }
    json_close(&j);
    json_end(&j);
}
