#ifndef GEODE_CHAT_H
#define GEODE_CHAT_H

#include "tokenizer.h"

/* A conversation the model recognises, rendered as role-labelled messages:

     ROLE <|role_sep|> \n TEXT <|message_sep|> \n\n

   and a reply is prompted by opening an `assistant` message and letting the
   model fill it in, up to the `<|message_sep|>` that closes it. */
typedef struct {
    int role_sep;
    int message_sep;
} Chat;

/* 0 when the vocabulary spells neither marker, which means this model does not
   use the format above and nothing here would render a prompt it understands. */
int chat_init(Chat *chat, const Tokenizer *tokenizer);

/* One user message plus the opening of the reply, ready to prefill. `opening`
   starts a fresh conversation rather than continuing the one in the cache. */
int chat_encode_turn(const Chat *chat, const Tokenizer *tokenizer,
                     const char *text, int opening, int *ids, int max_ids);

#endif
