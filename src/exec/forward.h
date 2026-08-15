#ifndef GEODE_FORWARD_H
#define GEODE_FORWARD_H

#include "model.h"

typedef struct Runtime Runtime;

Runtime *runtime_start(const Model *model, int n_ctx, int n_threads, char *err,
                       size_t errsz);
void runtime_stop(Runtime *runtime);

/* The widest chunk `forward` accepts. Prefill reads every weight once per
   chunk instead of once per token, so a prompt fed one token at a time costs
   what decoding it would; feeding it in chunks is what makes prefill cheaper
   than decode rather than the same price.

   Wider is not uniformly better, and the curve here is shallow -- 16, 32, 64
   and 100 measured 34, 36, 38 and 36 tok/s on the target box. Two effects
   cross. Attention and the shared expert amortize across the whole chunk, and
   a chunk this wide gives each routed expert about a tile of tokens rather
   than the one or two that fall back to a vector at a time. Against that, 64
   tokens already route to nearly every expert in the stack, so past here an
   extra token brings in as much expert weight as it saves, and the activations
   still have to stay in cache for the row-held-across-vectors loop to pay. */
#define PREFILL_CHUNK 64

/* Runs `n_tokens` consecutive tokens starting at `position` through every
   layer, appending to the kv cache as it goes. Returns logits over the
   vocabulary for the *last* token of the chunk, valid until the next call --
   the earlier ones are never sampled, and the output matrix is the widest in
   the model. Positions must be fed in order from 0: the cache holds no gaps. */
const float *forward(Runtime *runtime, const int *tokens, int position,
                     int n_tokens);

#endif
