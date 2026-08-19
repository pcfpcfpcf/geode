#include "gguf.h"
#include "home.h"
#include "kernels.h"
#include "model.h"
#include "modules.h"
#include "plan.h"
#include "quant.h"
#include "selftest.h"
#include "strategy.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PROMPT_TOKENS_MAX 2048
#define DEFAULT_PROMPT "The capital of France is"
#define DEFAULT_PREDICT_TOKENS 32

/* PROMPT, N_PREDICT, THREADS. */
#define MAX_OPTIONS 3

typedef struct {
    const char *prompt;
    int n_predict;
    int n_threads;
} RunArgs;

static const char *config_keys[] = {
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.kv_lora_rank",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.layer_norm_rms_epsilon",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "rope.scaling.yarn_log_multiplier",
    "expert_count",
    "expert_used_count",
    "expert_shared_count",
    "expert_feed_forward_length",
    "expert_weights_scale",
    "expert_weights_norm",
    "expert_gating_func",
    "leading_dense_block_count",
    "vocab_size",
};

static void print_config_key(const GgufFile *g, const char *arch, const char *key) {
    char full[160];
    unsigned long long integer;
    double real;
    snprintf(full, sizeof full, "%s.%s", arch, key);
    if (gguf_meta_u64(g, full, &integer))
        printf("  %-42s %llu\n", key, integer);
    else if (gguf_meta_f64(g, full, &real))
        printf("  %-42s %g\n", key, real);
}

static void print_layer_tensors(const GgufFile *g, int layer) {
    char prefix[16];
    snprintf(prefix, sizeof prefix, "blk.%d.", layer);
    size_t prefix_len = strlen(prefix);

    for (unsigned long long i = 0; i < g->n_tensors; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (strncmp(t->name, prefix, prefix_len) != 0) continue;
        printf("  %-34s %-5s [", t->name + prefix_len, quant_types[t->type].name);
        for (int d = 0; d < t->n_dims; d++)
            printf("%s%llu", d ? ", " : "", t->dims[d]);
        printf("]  %.2f MB\n", t->n_bytes / 1e6);
    }
}

static int describe_model(const GgufFile *g, const RunArgs *args) {
    (void)args;
    char arch[128] = "";
    char name[256] = "";
    gguf_meta_str(g, "general.architecture", arch, sizeof arch);
    gguf_meta_str(g, "general.name", name, sizeof name);
    printf("model: %s (%s), %llu tensors, file %.2f GB\n\n", name, arch,
           g->n_tensors, g->map_size / 1e9);

    printf("config:\n");
    for (unsigned i = 0; i < sizeof config_keys / sizeof *config_keys; i++)
        print_config_key(g, arch, config_keys[i]);

    unsigned long long histogram[64] = {0};
    unsigned long long bytes = 0;
    for (unsigned long long i = 0; i < g->n_tensors; i++) {
        histogram[g->tensors[i].type]++;
        bytes += g->tensors[i].n_bytes;
    }
    printf("\nquant histogram:");
    for (int type = 0; type < n_quant_types; type++)
        if (histogram[type])
            printf("  %s x%llu", quant_types[type].name, histogram[type]);
    printf("\ntensor bytes: %.2f GB (file %.2f GB)\n", bytes / 1e9,
           g->map_size / 1e9);

    printf("\nblk.0 (dense) tensors:\n");
    print_layer_tensors(g, 0);
    printf("\nblk.1 (moe) tensors:\n");
    print_layer_tensors(g, 1);
    return 0;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int argmax(const float *values, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

/* Prefills the prompt, then samples greedily until the model emits its end
   token or the budget runs out, printing tokens as they arrive. */
static void stream_tokens(const Strategy *strategy, Runtime *runtime,
                          const Model *model, const Tokenizer *tokenizer,
                          const int *ids, int n_prompt, int n_predict) {
    double started = now_seconds();
    const float *logits = NULL;
    for (int i = 0; i < n_prompt; i += PREFILL_CHUNK) {
        int n = n_prompt - i;
        if (n > PREFILL_CHUNK) n = PREFILL_CHUNK;
        logits = strategy->forward(runtime, ids + i, i, n);
    }
    double prefilled = now_seconds();

    int generated = 0;
    for (int i = 0; i < n_predict; i++) {
        int token = argmax(logits, model->n_vocab);
        if (token == tokenizer->eos_id) break;
        char text[512];
        tokenizer_decode(tokenizer, &token, 1, text, sizeof text);
        printf("%s", text);
        fflush(stdout);
        generated++;
        logits = strategy->forward(runtime, &token, n_prompt + i, 1);
    }
    double finished = now_seconds();

    double prefill_seconds = prefilled - started;
    printf("\n\nprefill: %d tokens in %.2fs (%.1f tok/s)\n", n_prompt,
           prefill_seconds, n_prompt / prefill_seconds);
    if (generated > 0)
        printf("decode:  %d tokens in %.2fs (%.2f tok/s)\n", generated,
               finished - prefilled, generated / (finished - prefilled));
}

static int generate(const GgufFile *g, const RunArgs *args) {
    char err[256];
    Plan plan;
    int planned = plan_load(&plan, geode_home("plan.json"), err, sizeof err);
    if (!planned) plan_default(&plan);

    double predicted[2];
    char note[256];
    const Strategy *strategy =
        strategy_choose(&plan, predicted, note, sizeof note);
    if (!strategy) {
        fprintf(stderr, "%s\n", note);
        return 1;
    }

    if (planned) plan_print(&plan);
    else printf("plan:     none (%s)\n", err);
    printf("%s\n", note);
    if (predicted[1] > 0)
        printf("predict:  %.1f-%.1f tok/s decode\n", predicted[0], predicted[1]);

    Model model;
    if (!model_load(&model, g, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    Tokenizer tokenizer;
    if (!tokenizer_init(&tokenizer, g, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        model_free(&model);
        return 1;
    }

    int ids[PROMPT_TOKENS_MAX];
    int n_prompt = tokenizer_encode_prompt(&tokenizer, args->prompt, ids,
                                           PROMPT_TOKENS_MAX);
    int rc = 1;
    if (n_prompt < 1) {
        fprintf(stderr, "prompt encoded to %d tokens; expected at least 1\n",
                n_prompt);
    } else {
        if (n_prompt == PROMPT_TOKENS_MAX)
            fprintf(stderr, "warning: prompt truncated to %d tokens\n",
                    n_prompt);

        Runtime *runtime =
            strategy->start(&model, &plan, n_prompt + args->n_predict,
                            args->n_threads, err, sizeof err);
        if (!runtime) {
            fprintf(stderr, "%s\n", err);
        } else {
            printf("%s", args->prompt);
            fflush(stdout);
            stream_tokens(strategy, runtime, &model, &tokenizer, ids, n_prompt,
                          args->n_predict);
            strategy->stop(runtime);
            rc = 0;
        }
    }

    tokenizer_free(&tokenizer);
    model_free(&model);
    return rc;
}

static int run_kernels(const GgufFile *g, const RunArgs *a) {
    (void)a;
    return selftest_kernels(g);
}

static int run_tokenizer(const GgufFile *g, const RunArgs *a) {
    (void)a;
    return selftest_tokenizer(g);
}

static int run_prefill(const GgufFile *g, const RunArgs *a) {
    (void)a;
    return selftest_prefill(g);
}

typedef struct {
    const char *name;
    const char *options; /* NULL for a command taking only a model */
    const char *summary;
    int (*run)(const GgufFile *g, const RunArgs *args);
} Command;

static const Command commands[] = {
    {"exec", NULL, "print config, quant histogram and layer tensors",
     describe_model},
    {"exec-run", "[PROMPT] [N_PREDICT] [THREADS]",
     "generate, using the cached plan to pick an executor", generate},
    {"exec-kernels", NULL, "check the quant kernels against a dequant reference",
     run_kernels},
    {"exec-tokenize", NULL, "round-trip a few prompts through the tokenizer",
     run_tokenizer},
    {"exec-prefill", NULL, "check chunked prefill against one token at a time",
     run_prefill},
};

static const Command *find_command(const char *name) {
    for (unsigned i = 0; i < sizeof commands / sizeof *commands; i++)
        if (strcmp(commands[i].name, name) == 0) return &commands[i];
    return NULL;
}

int exec_handles(const char *name) { return find_command(name) != NULL; }

static void print_usage(void) {
    fprintf(stderr, "usage: geode COMMAND MODEL.gguf [options]\n\n");
    for (unsigned i = 0; i < sizeof commands / sizeof *commands; i++) {
        fprintf(stderr, "  %-14s %s\n", commands[i].name, commands[i].summary);
        if (commands[i].options)
            fprintf(stderr, "  %-14s options: %s\n", "", commands[i].options);
    }
}

int exec_main(int argc, char **argv) {
    const Command *command = find_command(argv[0]);
    int n_options = argc - 2;
    if (!command || argc < 2 ||
        n_options > (command->options ? MAX_OPTIONS : 0)) {
        print_usage();
        return 2;
    }

    RunArgs args = {DEFAULT_PROMPT, DEFAULT_PREDICT_TOKENS, 0};
    if (n_options > 0) args.prompt = argv[2];
    if (n_options > 1) args.n_predict = atoi(argv[3]);
    if (n_options > 2) args.n_threads = atoi(argv[4]);
    if (args.n_predict < 1) {
        fprintf(stderr, "N_PREDICT is %d; expected at least 1\n",
                args.n_predict);
        return 2;
    }

    char err[256];
    GgufFile g;
    if (!gguf_open(&g, argv[1], err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    int rc = command->run(&g, &args);
    gguf_close(&g);
    return rc;
}

int exec_cli(char *path){

    char err[256];
    GgufFile g;
    if (!gguf_open(&g, path, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    RunArgs args = {"", DEFAULT_PREDICT_TOKENS, 0};
    
    int convo = 1;
    printf("Talk To Your Model On Your Machine\n\n\n\n");
    while (convo){
        char msg[PROMPT_TOKENS_MAX];
        printf("> ");
        scanf("%s", msg);
         
        if (!strcmp(msg, "q")) {
            printf("\nQuitting...\n");
            convo=0;
        }
    }

    return 0;
}