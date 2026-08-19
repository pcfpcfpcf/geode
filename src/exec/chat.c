#include "chat.h"

/* Both markers carry their own trailing whitespace into the vocabulary, so a
   turn never spells the newlines around them itself. */
#define ROLE_SEP "<|role_sep|>\n"
#define MESSAGE_SEP "<|message_sep|>\n\n"

int chat_init(Chat *chat, const Tokenizer *tokenizer) {
    chat->role_sep = tokenizer_token_id(tokenizer, ROLE_SEP);
    chat->message_sep = tokenizer_token_id(tokenizer, MESSAGE_SEP);
    return chat->role_sep >= 0 && chat->message_sep >= 0;
}

typedef struct {
    const Chat *chat;
    const Tokenizer *tokenizer;
    int *ids;
    int n;
    int max;
} Turn;

static void put_token(Turn *turn, int id) {
    if (turn->n < turn->max) turn->ids[turn->n++] = id;
}

static void put_text(Turn *turn, const char *text) {
    turn->n += tokenizer_encode(turn->tokenizer, text, turn->ids + turn->n,
                                turn->max - turn->n);
}

static void open_message(Turn *turn, const char *role) {
    put_text(turn, role);
    put_token(turn, turn->chat->role_sep);
}

static void close_message(Turn *turn) {
    put_token(turn, turn->chat->message_sep);
}

int chat_encode_turn(const Chat *chat, const Tokenizer *tokenizer,
                     const char *text, int opening, int *ids, int max_ids) {
    Turn turn = {chat, tokenizer, ids, 0, max_ids};

    /* Either way one message is left hanging: the empty system message that
       opens a conversation, or the reply that generation stopped short of its
       separator. */
    if (opening) {
        if (tokenizer->add_bos) put_token(&turn, tokenizer->bos_id);
        open_message(&turn, "system");
    }
    close_message(&turn);

    open_message(&turn, "user");
    put_text(&turn, text);
    close_message(&turn);
    open_message(&turn, "assistant");
    return turn.n;
}
