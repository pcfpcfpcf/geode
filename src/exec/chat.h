#ifndef GEODE_CHAT_H
#define GEODE_CHAT_H

#include "tokenizer.h"

/* A conversation the model recognises, rendered as role-labelled messages.
   Two formats are known:

     DeepSeek:  ROLE <|role_sep|> \n TEXT <|message_sep|> \n\n
     Qwen3:     <|im_start|> ROLE \n TEXT <|im_end|> \n

   and a reply is prompted by opening an `assistant` message and letting the
   model fill it in, up to the marker that closes it. A Qwen3 reply that
   carries its reasoning markers is prompted past them, so the model answers
   directly instead of thinking out loud. */
typedef enum {
    CHAT_DEEPSEEK,
    CHAT_QWEN3,
} ChatFormat;

typedef struct {
    ChatFormat format;
    int start;    /* deepseek role_sep, qwen3 im_start */
    int end;     /* deepseek message_sep, qwen3 im_end */
    int thinking; /* qwen3 reasoning-open marker, else -1 */
    int response; /* qwen3 reasoning-close marker, else -1 */
} Chat;

/* 0 when the vocabulary spells neither format's markers, which means this model
   does not use either and nothing here would render a prompt it understands. */
int chat_init(Chat *chat, const Tokenizer *tokenizer);

typedef struct {
    const char *role;
    const char *text;
} ChatMessage;

/* The messages plus the opening of the reply, ready to prefill. `opening`
   starts a fresh conversation rather than continuing the one in the cache. */
int chat_encode_messages(const Chat *chat, const Tokenizer *tokenizer,
                         const ChatMessage *messages, int n_messages,
                         int opening, int *ids, int max_ids);

/* One user message plus the opening of the reply, ready to prefill. `opening`
   starts a fresh conversation rather than continuing the one in the cache. */
int chat_encode_turn(const Chat *chat, const Tokenizer *tokenizer,
                     const char *text, int opening, int *ids, int max_ids);

#endif
