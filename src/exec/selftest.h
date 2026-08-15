#ifndef GEODE_SELFTEST_H
#define GEODE_SELFTEST_H

#include "gguf.h"

/* Each prints what it compared and returns 0 when the check passed, 1 when it
   failed -- the shell's convention, since each is its own command. */
int selftest_kernels(const GgufFile *g);
int selftest_tokenizer(const GgufFile *g);
int selftest_prefill(const GgufFile *g);

#endif
