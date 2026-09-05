#include "gguf.h"
#include "json.h"
#include "modules.h"
#include "session.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_PORT 8080
#define REQUEST_BODY_MAX (1 << 20)
#define CONTEXT_TOKENS 4096
#define REPLY_TOKENS_MAX 2048
#define DEFAULT_REPLY_TOKENS 256
#define MAX_MESSAGES 32
#define READ_TIMEOUT_MS 10000
#define HEADERS_MAX 8192
#define CHUNK_MAX 8192

#define FINISH_STOP "\"stop\""
#define FINISH_LENGTH "\"length\""

static const char *status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 500: return "Internal Server Error";
    default: return "Error";
    }
}

static int write_all(int fd, const char *buf, int len) {
    while (len > 0) {
        int r = (int)write(fd, buf, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        buf += r;
        len -= r;
    }
    return 1;
}

static int read_some(int fd, char *buf, int max) {
    struct pollfd pfd = {fd, POLLIN, 0};
    int r = poll(&pfd, 1, READ_TIMEOUT_MS);
    if (r <= 0) return -1;
    return (int)read(fd, buf, max);
}

/* The start of one header line, or NULL. Header names are case-insensitive. */
static const char *find_header(const char *headers, const char *sep,
                               const char *name) {
    const char *p = headers;
    while (p < sep) {
        const char *eol = strstr(p, "\r\n");
        if (!eol || eol > sep) break;
        if (strncasecmp(p, name, strlen(name)) == 0) return p;
        p = eol + 2;
    }
    return NULL;
}

static int read_request(int fd, char *headers, int headers_max, char *body,
                        int body_max, int *body_len) {
    int n = 0;
    for (;;) {
        int r = read_some(fd, headers + n, headers_max - 1 - n);
        if (r <= 0) return 0;
        n += r;
        headers[n] = '\0';
        if (strstr(headers, "\r\n\r\n")) break;
        if (n >= headers_max - 1) return 0;
    }

    char *sep = strstr(headers, "\r\n\r\n");
    int header_end = (int)(sep - headers) + 4;
    int have = n - header_end;

    if (find_header(headers, sep, "transfer-encoding:")) return 0;
    const char *cl = find_header(headers, sep, "content-length:");
    long content_length = cl ? atol(cl + 15) : 0;
    if (content_length < 0 || content_length > body_max) return 0;

    memcpy(body, headers + header_end, have);
    *body_len = have;
    while (*body_len < content_length) {
        int r = read_some(fd, body + *body_len,
                          (int)content_length - *body_len);
        if (r <= 0) return 0;
        *body_len += r;
    }
    return 1;
}

static void respond_json(int fd, int status, const char *body) {
    char head[256];
    int n = snprintf(head, sizeof head,
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: application/json\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     status, status_text(status), strlen(body));
    write_all(fd, head, n);
    write_all(fd, body, strlen(body));
}

static void respond_sse(int fd) {
    static const char head[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n";
    write_all(fd, head, sizeof head - 1);
}

/* The preflight a browser sends before a cross-origin POST. */
static void respond_options(int fd) {
    static const char head[] =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\n"
        "Connection: close\r\n"
        "\r\n";
    write_all(fd, head, sizeof head - 1);
}

static int respond_error(int fd, int status, const char *type,
                         const char *message) {
    char *body = NULL;
    size_t body_len = 0;
    FILE *f = open_memstream(&body, &body_len);
    Json j;
    json_begin(&j, f);
    json_open(&j, "error", 0);
    json_string(&j, "message", message);
    json_string(&j, "type", type);
    json_null(&j, "param");
    json_null(&j, "code");
    json_close(&j);
    json_end(&j);
    fclose(f);
    respond_json(fd, status, body);
    free(body);
    return status;
}

static void timings(const StreamStats *stats, double *prefill, double *decode) {
    *prefill = stats->prefill_seconds > 0
                   ? stats->n_prompt / stats->prefill_seconds
                   : 0;
    *decode = stats->decode_seconds > 0
                  ? stats->n_generated / stats->decode_seconds
                  : 0;
}

/* One SSE event: a chat completion chunk, compact enough for the `data:`
   line. `finish_json` is the quoted reason or NULL for an open chunk; `stats`
   adds the measured rates to the chunk that closes the reply. */
static int sse_chunk(char *out, int max, const char *id, long created,
                     const char *model, const char *role, const char *content,
                     const char *finish_json, const StreamStats *stats) {
    int n = snprintf(out, max,
                     "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                     "\"created\":%ld,\"model\":\"",
                     id, created);
    n += json_escape(model, out + n, max - n);
    n += snprintf(out + n, max - n, "\",\"choices\":[{\"index\":0,\"delta\":{");
    if (role) {
        n += snprintf(out + n, max - n, "\"role\":\"");
        n += json_escape(role, out + n, max - n);
        n += snprintf(out + n, max - n, "\"");
    }
    if (content) {
        n += snprintf(out + n, max - n, "%s\"content\":\"", role ? "," : "");
        n += json_escape(content, out + n, max - n);
        n += snprintf(out + n, max - n, "\"");
    }
    n += snprintf(out + n, max - n, "},\"finish_reason\":%s}]",
                  finish_json ? finish_json : "null");
    if (stats) {
        double prefill, decode;
        timings(stats, &prefill, &decode);
        n += snprintf(out + n, max - n,
                      ",\"timings\":{\"prefill_tok_s\":%.1f,"
                      "\"decode_tok_s\":%.1f}",
                      prefill, decode);
    }
    n += snprintf(out + n, max - n, "}\n\n");
    return n;
}

typedef struct {
    char *data;
    int len;
    int cap;
} Buffer;

static void buffer_append(Buffer *b, const char *text, int length) {
    if (b->len + length + 1 > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 1024;
        while (b->len + length + 1 > b->cap) b->cap *= 2;
        b->data = realloc(b->data, b->cap);
    }
    memcpy(b->data + b->len, text, length);
    b->len += length;
    b->data[b->len] = '\0';
}

static int collect_sink(void *ctx, const char *text, int length) {
    buffer_append(ctx, text, length);
    return 0;
}

typedef struct {
    int fd;
    const char *id;
    long created;
    const char *model;
    int failed;
} SseCtx;

static int sse_sink(void *ctx, const char *text, int length) {
    SseCtx *c = ctx;
    char content[512];
    if (length >= (int)sizeof content) length = (int)sizeof content - 1;
    memcpy(content, text, length);
    content[length] = '\0';
    char buf[CHUNK_MAX];
    int n = sse_chunk(buf, sizeof buf, c->id, c->created, c->model, NULL,
                      content, NULL, NULL);
    if (!write_all(c->fd, buf, n)) {
        c->failed = 1;
        return 1;
    }
    return 0;
}

static void log_tokens(const StreamStats *stats) {
    fprintf(stderr, "  %d prompt + %d completion tokens\n", stats->n_prompt,
            stats->n_generated);
}

static int plain_reply(Session *session, Runtime *runtime,
                       const char *model_name, int fd, const char *id,
                       long created, const int *ids, int n_prompt,
                       int max_tokens) {
    Buffer content = {0};
    StreamStats stats;
    session_stream(session, runtime, ids, n_prompt, 0, max_tokens,
                   collect_sink, &content, &stats);

    char *body = NULL;
    size_t body_len = 0;
    FILE *f = open_memstream(&body, &body_len);
    Json j;
    json_begin(&j, f);
    json_string(&j, "id", id);
    json_string(&j, "object", "chat.completion");
    json_u64(&j, "created", (unsigned long long)created);
    json_string(&j, "model", model_name);
    json_open(&j, "choices", 1);
    json_open(&j, NULL, 0);
    json_u64(&j, "index", 0);
    json_open(&j, "message", 0);
    json_string(&j, "role", "assistant");
    json_string(&j, "content", content.data ? content.data : "");
    json_close(&j);
    json_string(&j, "finish_reason",
                stats.n_generated == max_tokens ? "length" : "stop");
    json_close(&j);
    json_close(&j);
    json_open(&j, "usage", 0);
    json_u64(&j, "prompt_tokens", (unsigned long long)stats.n_prompt);
    json_u64(&j, "completion_tokens", (unsigned long long)stats.n_generated);
    json_u64(&j, "total_tokens",
             (unsigned long long)(stats.n_prompt + stats.n_generated));
    json_close(&j);
    double prefill, decode;
    timings(&stats, &prefill, &decode);
    json_open(&j, "timings", 0);
    json_double(&j, "prefill_tok_s", prefill);
    json_double(&j, "decode_tok_s", decode);
    json_close(&j);
    json_end(&j);
    fclose(f);

    respond_json(fd, 200, body);
    log_tokens(&stats);
    free(body);
    free(content.data);
    return 200;
}

static int stream_reply(Session *session, Runtime *runtime,
                        const char *model_name, int fd, const char *id,
                        long created, const int *ids, int n_prompt,
                        int max_tokens) {
    respond_sse(fd);

    char buf[CHUNK_MAX];
    int n = sse_chunk(buf, sizeof buf, id, created, model_name, "assistant",
                      NULL, NULL, NULL);
    if (!write_all(fd, buf, n)) return 200;

    SseCtx ctx = {fd, id, created, model_name, 0};
    StreamStats stats;
    session_stream(session, runtime, ids, n_prompt, 0, max_tokens, sse_sink,
                   &ctx, &stats);
    if (!ctx.failed) {
        n = sse_chunk(buf, sizeof buf, id, created, model_name, NULL, NULL,
                      stats.n_generated == max_tokens ? FINISH_LENGTH
                                                      : FINISH_STOP,
                      &stats);
        if (write_all(fd, buf, n)) write_all(fd, "data: [DONE]\n\n", 15);
    }
    log_tokens(&stats);
    return 200;
}

static int handle_models(int fd, const char *model_name, long created) {
    char *body = NULL;
    size_t body_len = 0;
    FILE *f = open_memstream(&body, &body_len);
    Json j;
    json_begin(&j, f);
    json_string(&j, "object", "list");
    json_open(&j, "data", 1);
    json_open(&j, NULL, 0);
    json_string(&j, "id", model_name);
    json_string(&j, "object", "model");
    json_u64(&j, "created", (unsigned long long)created);
    json_string(&j, "owned_by", "geode");
    json_close(&j);
    json_close(&j);
    json_end(&j);
    fclose(f);
    respond_json(fd, 200, body);
    free(body);
    return 200;
}

static int handle_chat_completions(Session *session, Runtime *runtime,
                                   const char *model_name, int fd,
                                   const char *body) {
    char err[256];
    JVal *req = json_parse(body, err, sizeof err);
    if (!req) return respond_error(fd, 400, "invalid_request_error", err);

    const JVal *messages = json_get(req, "messages");
    if (!messages || messages->kind != JV_ARR || messages->n_items < 1) {
        json_free(req);
        return respond_error(fd, 400, "invalid_request_error",
                             "messages must be a non-empty array");
    }

    ChatMessage list[MAX_MESSAGES];
    int n_messages = 0;
    for (int i = 0; i < messages->n_items; i++) {
        if (n_messages == MAX_MESSAGES) {
            json_free(req);
            return respond_error(fd, 400, "invalid_request_error",
                                 "too many messages");
        }
        const JVal *m = messages->items[i];
        const JVal *role = json_get(m, "role");
        const JVal *content = json_get(m, "content");
        if (!role || role->kind != JV_STR || !content ||
            content->kind != JV_STR) {
            json_free(req);
            return respond_error(fd, 400, "invalid_request_error",
                                 "each message needs a string role and "
                                 "content");
        }
        list[n_messages++] = (ChatMessage){role->str, content->str};
    }

    int stream = 0;
    const JVal *stream_val = json_get(req, "stream");
    if (stream_val && stream_val->kind == JV_BOOL)
        stream = stream_val->num != 0;

    float temperature = SAMPLER_TEMPERATURE;
    const JVal *temp_val = json_get(req, "temperature");
    if (temp_val && temp_val->kind == JV_NUM) {
        temperature = (float)temp_val->num;
        if (temperature < 0) temperature = 0;
        if (temperature > 2) temperature = 2;
    }

    int max_tokens = DEFAULT_REPLY_TOKENS;
    const JVal *max_val = json_get(req, "max_tokens");
    if (max_val && max_val->kind == JV_NUM) max_tokens = (int)max_val->num;
    if (max_tokens < 1 || max_tokens > REPLY_TOKENS_MAX) {
        char msg[128];
        snprintf(msg, sizeof msg, "max_tokens must be between 1 and %d",
                 REPLY_TOKENS_MAX);
        json_free(req);
        return respond_error(fd, 400, "invalid_request_error", msg);
    }

    const JVal *n_val = json_get(req, "n");
    if (n_val && n_val->kind == JV_NUM && n_val->num != 1) {
        json_free(req);
        return respond_error(fd, 400, "invalid_request_error", "n must be 1");
    }

    int ids[CONTEXT_TOKENS];
    int n_prompt = chat_encode_messages(&session->chat, &session->tokenizer,
                                        list, n_messages, 1, ids,
                                        CONTEXT_TOKENS);
    json_free(req);
    if (n_prompt < 1)
        return respond_error(fd, 400, "invalid_request_error",
                             "messages encoded to no tokens");
    if (n_prompt >= CONTEXT_TOKENS) {
        char msg[128];
        snprintf(msg, sizeof msg, "prompt too long for the %d-token context",
                 CONTEXT_TOKENS);
        return respond_error(fd, 400, "invalid_request_error", msg);
    }
    if (n_prompt + max_tokens > CONTEXT_TOKENS) {
        char msg[128];
        snprintf(msg, sizeof msg,
                 "prompt plus max_tokens exceeds the %d-token context",
                 CONTEXT_TOKENS);
        return respond_error(fd, 400, "invalid_request_error", msg);
    }

    sampler_init(&session->sampler, temperature, SAMPLER_TOP_P,
                 SAMPLER_REPETITION_PENALTY, SAMPLER_SEED);

    char id[32];
    static int next_id = 0;
    snprintf(id, sizeof id, "chatcmpl-%d", ++next_id);
    long created = time(NULL);

    if (stream)
        return stream_reply(session, runtime, model_name, fd, id, created,
                            ids, n_prompt, max_tokens);
    return plain_reply(session, runtime, model_name, fd, id, created, ids,
                       n_prompt, max_tokens);
}

static void handle_client(Session *session, Runtime *runtime,
                          const char *model_name, long created, int fd) {
char headers[HEADERS_MAX];
    char body[REQUEST_BODY_MAX + 1];
    int body_len = 0;
    if (!read_request(fd, headers, sizeof headers, body, sizeof body,
                      &body_len)) {
        respond_error(fd, 400, "invalid_request_error", "malformed request");
        return;
    }
    body[body_len] = '\0';

    char method[16], path[256];
    if (sscanf(headers, "%15s %255s", method, path) != 2) {
        respond_error(fd, 400, "invalid_request_error",
                      "malformed request line");
        return;
    }
    char *query = strchr(path, '?');
    if (query) *query = '\0';

    int status;
    if (strcmp(method, "OPTIONS") == 0) {
        respond_options(fd);
        status = 204;
    } else if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
        status = handle_models(fd, model_name, created);
    } else if (strcmp(method, "POST") == 0 &&
               strcmp(path, "/v1/chat/completions") == 0) {
        status = handle_chat_completions(session, runtime, model_name, fd,
                                         body);
    } else {
        status = respond_error(fd, 404, "invalid_request_error",
                               "no such endpoint");
    }
    fprintf(stderr, "%s %s -> %d\n", method, path, status);
}

static int serve_loop(Session *session, Runtime *runtime,
                      const char *model_name, long created, int port) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "socket: %s\n", strerror(errno));
        return 1;
    }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof addr) < 0) {
        fprintf(stderr, "cannot bind 127.0.0.1:%d: %s\n", port,
                strerror(errno));
        close(listen_fd);
        return 1;
    }
    if (listen(listen_fd, 8) < 0) {
        fprintf(stderr, "listen: %s\n", strerror(errno));
        close(listen_fd);
        return 1;
    }

    signal(SIGPIPE, SIG_IGN);

    printf("serving %s on http://127.0.0.1:%d/v1\n", model_name, port);
    printf("  GET  /v1/models\n");
    printf("  POST /v1/chat/completions\n");
    fflush(stdout);

    for (;;) {
        int client = accept(listen_fd, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "accept: %s\n", strerror(errno));
            break;
        }
        handle_client(session, runtime, model_name, created, client);
        close(client);
    }
    close(listen_fd);
    return 0;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s MODEL.gguf [PORT]\n"
            "  open an OpenAI-compatible API on 127.0.0.1 (default port %d)\n",
            argv0, DEFAULT_PORT);
}

int serve_main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        usage(argv[0]);
        return 2;
    }
    int port = DEFAULT_PORT;
    if (argc == 3) {
        port = atoi(argv[2]);
        if (port < 1 || port > 65535) {
            fprintf(stderr, "PORT is %d; expected 1-65535\n", port);
            return 2;
        }
    }

    char err[256];
    GgufFile g;
    if (!gguf_open(&g, argv[1], err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    Session session;
    if (!session_open(&session, &g, err, sizeof err)) {
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
        session.strategy->start(&session.model, &session.plan, CONTEXT_TOKENS,
                                0, err, sizeof err);
    if (!runtime) {
        fprintf(stderr, "%s\n", err);
        session_close(&session);
        gguf_close(&g);
        return 1;
    }

    char model_name[256] = "";
    gguf_meta_str(&g, "general.name", model_name, sizeof model_name);
    if (!model_name[0]) {
        const char *base = strrchr(argv[1], '/');
        snprintf(model_name, sizeof model_name, "%s",
                 base ? base + 1 : argv[1]);
    }

    struct stat st;
    long created = 0;
    if (stat(argv[1], &st) == 0) created = (long)st.st_mtime;

    int rc = serve_loop(&session, runtime, model_name, created, port);

    session.strategy->stop(runtime);
    session_close(&session);
    gguf_close(&g);
    return rc;
}
