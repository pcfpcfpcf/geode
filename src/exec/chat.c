#include "chat.h"

/* Both markers carry their own trailing whitespace into the vocabulary, so a
   turn never spells the newlines around them itself. */
#define ROLE_SEP "<|role_sep|>\n"
#define MESSAGE_SEP "<|message_sep|>\n\n"

int chat_init(Chat *chat, const Tokenizer *tokenizer) {
    int role_sep = tokenizer_token_id(tokenizer, ROLE_SEP);
    int message_sep = tokenizer_token_id(tokenizer, MESSAGE_SEP);
    if (role_sep >= 0 && message_sep >= 0) {
        chat->format = CHAT_DEEPSEEK;
        chat->start = role_sep;
        chat->end = message_sep;
        chat->thinking = -1;
        chat->response = -1;
        return 1;
    }
    int im_start = tokenizer_token_id(tokenizer, "<|im_start|>");
    int im_end = tokenizer_token_id(tokenizer, "<|im_end|>");
    if (im_start >= 0 && im_end >= 0) {
        chat->format = CHAT_QWEN3;
        chat->start = im_start;
        chat->end = im_end;
        /* Two Qwen3 vocab generations spell the reasoning markers with and
           without a leading space; take whichever this model carries. */
        chat->thinking = tokenizer_token_id(tokenizer, "<think>");
        if (chat->thinking < 0)
            chat->thinking = tokenizer_token_id(tokenizer, " thinking");
        chat->response = tokenizer_token_id(tokenizer, "</think>");
        if (chat->response < 0)
            chat->response = tokenizer_token_id(tokenizer, " response");
        return 1;
    }
    return 0;
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
    if (turn->chat->format == CHAT_QWEN3) {
        put_token(turn, turn->chat->start);
        put_text(turn, role);
        put_text(turn, "\n");
    } else {
        put_text(turn, role);
        put_token(turn, turn->chat->start);
    }
}

static void close_message(Turn *turn) {
    put_token(turn, turn->chat->end);
    if (turn->chat->format == CHAT_QWEN3) put_text(turn, "\n");
}

int chat_encode_messages(const Chat *chat, const Tokenizer *tokenizer,
                         const ChatMessage *messages, int n_messages,
                         int opening, int *ids, int max_ids) {
    Turn turn = {chat, tokenizer, ids, 0, max_ids};

    /* Either way one message is left hanging: the empty system message that
       opens a conversation, or the reply that generation stopped short of its
       separator. */
    if (opening) {
        if (tokenizer->add_bos) put_token(&turn, tokenizer->bos_id);
        open_message(&turn, "system");
    }
    close_message(&turn);

    for (int i = 0; i < n_messages; i++) {
        open_message(&turn, messages[i].role);
        put_text(&turn, messages[i].text);
        close_message(&turn);
    }
    open_message(&turn, "assistant");
    if (chat->format == CHAT_QWEN3 && chat->thinking >= 0 && chat->response >= 0) {
        put_token(&turn, chat->thinking);
        put_text(&turn, "\n\n");
        put_token(&turn, chat->response);
        put_text(&turn, "\n\n");
    }
    return turn.n;
}

int chat_encode_turn(const Chat *chat, const Tokenizer *tokenizer,
                     const char *text, int opening, int *ids, int max_ids) {
    ChatMessage messages[1];
    messages[0] = (ChatMessage){"user", text};
    /* `chat_encode_messages` opens the empty system message itself when
       `opening` is set; passing one here would render it twice. */
    return chat_encode_messages(chat, tokenizer, messages, 1, opening, ids,
                                max_ids);
}
