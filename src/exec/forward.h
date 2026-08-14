#ifndef GEODE_FORWARD_H
#define GEODE_FORWARD_H

#include "model.h"

typedef struct Runtime Runtime;

Runtime *runtime_start(const Model *model, int n_ctx, int n_threads, char *err,
                       size_t errsz);
void runtime_stop(Runtime *runtime);

/* Runs one token at `position` through every layer, appending to the kv cache
   as it goes. Returns logits over the vocabulary, valid until the next call.
   Positions must be fed in order from 0: the cache holds no gaps. */
const float *forward(Runtime *runtime, int token, int position);

#endif
