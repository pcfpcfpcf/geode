#include "gguf.h"
#include "home.h"
#include "kernels.h"
#include "modules.h"
#include "quant.h"
#include "selftest.h"
#include "session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_TOKENS_MAX 2048
#define DEFAULT_PROMPT "The capital of France is"
#define DEFAULT_PREDICT_TOKENS 32

/* PROMPT, N_PREDICT, THREADS. */
#define MAX_OPTIONS 3

#define CLI_CONTEXT_TOKENS 4096
#define CLI_REPLY_TOKENS 256
#define CLI_LINE_MAX 4096
#define CLI_QUIT "q"

typedef struct {
    const char *prompt;
    int n_predict;
    int n_threads;
    const char *model;
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

static int print_token(void *ctx, const char *text, int length) {
    (void)ctx;
    fwrite(text, 1, length, stdout);
    fflush(stdout);
    return 0;
}

static int stream_tokens(Session *session, Runtime *runtime, const int *ids,
                         int n_prompt, int position, int n_predict) {
    StreamStats stats;
    int end = session_stream(session, runtime, ids, n_prompt, position,
                             n_predict, print_token, NULL, &stats);
    printf("\n\nprefill: %d tokens in %.2fs (%.1f tok/s)\n", stats.n_prompt,
           stats.prefill_seconds, stats.n_prompt / stats.prefill_seconds);
    if (stats.n_generated > 0)
        printf("decode:  %d tokens in %.2fs (%.2f tok/s)\n",
               stats.n_generated, stats.decode_seconds,
               stats.n_generated / stats.decode_seconds);
    return end;
}

static int generate(const GgufFile *g, const RunArgs *args) {
    char err[256];
    Session session;
    if (!session_open(&session, g, args->model, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    int ids[PROMPT_TOKENS_MAX];
    int n_prompt = tokenizer_encode_prompt(&session.tokenizer, args->prompt,
                                           ids, PROMPT_TOKENS_MAX);
    int rc = 1;
    if (n_prompt < 1) {
        fprintf(stderr, "prompt encoded to %d tokens; expected at least 1\n",
                n_prompt);
    } else {
        if (n_prompt == PROMPT_TOKENS_MAX)
            fprintf(stderr, "warning: prompt truncated to %d tokens\n",
                    n_prompt);

        Runtime *runtime = session.strategy->start(
            &session.model, &session.plan, n_prompt + args->n_predict,
            args->n_threads, err, sizeof err);
        if (!runtime) {
            fprintf(stderr, "%s\n", err);
        } else {
            printf("%s", args->prompt);
            fflush(stdout);
            stream_tokens(&session, runtime, ids, n_prompt, 0,
                          args->n_predict);
            session.strategy->stop(runtime);
            rc = 0;
        }
    }

    session_close(&session);
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

    RunArgs args = {DEFAULT_PROMPT, DEFAULT_PREDICT_TOKENS, 0, argv[1]};
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

/* One line of input without its newline, or NULL at end of input. A line
   wider than the buffer is refused rather than half-read into a turn. */
static const char *read_turn(char *line, int max) {
    if (!fgets(line, max, stdin)) return NULL;
    int n = (int)strcspn(line, "\n");
    if (line[n] == '\n') {
        line[n] = '\0';
        return line;
    }
    int c;
    while ((c = fgetc(stdin)) != '\n' && c != EOF) {}
    fprintf(stderr, "line over %d bytes; send a shorter one\n", max - 1);
    line[0] = '\0';
    return line;
}

int exec_cli(const char *path) {
    char err[256];
    GgufFile g;
    if (!gguf_open(&g, path, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    Session session;
    if (!session_open(&session, &g, path, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        gguf_close(&g);
        return 1;
    }
    if (!session.has_chat) {
        fprintf(stderr,
                "model has no role markers in its vocabulary, so it takes no "
                "conversation; use 'exec-run MODEL.gguf PROMPT' to complete "
                "text with it instead\n");
        session_close(&session);
        gguf_close(&g);
        return 1;
    }

    Runtime *runtime =
        session.strategy->start(&session.model, &session.plan,
                                CLI_CONTEXT_TOKENS, 0, err, sizeof err);
    if (!runtime) {
        fprintf(stderr, "%s\n", err);
        session_close(&session);
        gguf_close(&g);
        return 1;
    }

    printf("\ntalk to your model on your machine\n"
           "%d token context, %d token replies, '%s' quits\n",
           CLI_CONTEXT_TOKENS, CLI_REPLY_TOKENS, CLI_QUIT);

    int ids[PROMPT_TOKENS_MAX];
    int position = 0;
    for (;;) {
        char line[CLI_LINE_MAX];
        printf("\n> ");
        fflush(stdout);
        const char *turn = read_turn(line, sizeof line);
        if (!turn || strcmp(turn, CLI_QUIT) == 0) break;
        if (!turn[0]) continue;

        int n_turn = chat_encode_turn(&session.chat, &session.tokenizer, turn,
                                      position == 0, ids, PROMPT_TOKENS_MAX);
        if (n_turn < 1) {
            fprintf(stderr, "that encoded to no tokens; send some text\n");
            continue;
        }
        if (position + n_turn + CLI_REPLY_TOKENS > CLI_CONTEXT_TOKENS) {
            fprintf(stderr,
                    "context full at %d of %d tokens; '%s' and start again\n",
                    position, CLI_CONTEXT_TOKENS, CLI_QUIT);
            continue;
        }
        position = stream_tokens(&session, runtime, ids, n_turn, position,
                                 CLI_REPLY_TOKENS);
    }

    session.strategy->stop(runtime);
    session_close(&session);
    gguf_close(&g);
    return 0;
}
