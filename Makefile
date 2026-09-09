CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
           -Isrc/common -Isrc/probe -Isrc/manifest -Isrc/planner -Isrc/exec
LDLIBS   = -ldl -pthread -lm

GEODE_SRC = src/geode.c \
            src/probe/probe.c src/probe/dram.c src/probe/flops.c \
            src/probe/nvme.c src/probe/gpu.c \
            src/manifest/manifest.c \
            src/planner/planner.c \
            src/exec/gguf.c src/exec/kernels.c src/exec/tokenizer.c \
            src/exec/model.c src/exec/forward.c \
            src/exec/plan.c src/exec/strategy.c src/exec/cpu_stream.c \
            src/exec/trace.c \
            src/exec/cuda.c src/exec/hybrid.c \
            src/exec/chat.c src/exec/sampler.c \
            src/exec/session.c src/exec/serve.c \
            src/exec/selftest.c src/exec/exec.c \
            src/common/json.c src/common/quant.c src/common/parallel.c

BIN   := bin/geode

all: $(BIN)

$(BIN): $(GEODE_SRC) src/common/*.h src/probe/*.h src/exec/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(GEODE_SRC) $(LDLIBS)

clean:
	rm -rf bin

.PHONY: all clean
