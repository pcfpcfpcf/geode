CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
           -Isrc/common -Isrc/probe -Isrc/manifest -Isrc/planner -Isrc/exec
LDLIBS   = -ldl -pthread -lm

GEODE_SRC = src/geode.c \
            src/probe/probe.c src/probe/dram.c src/probe/flops.c \
            src/probe/nvme.c src/probe/gpu.c \
            src/manifest/manifest.c \
            src/planner/planner.c \
            src/exec/gguf.c src/exec/exec.c \
            src/common/json.c src/common/quant.c

BIN   := bin/geode

all: $(BIN)

$(BIN): $(GEODE_SRC) src/common/json.h src/common/modules.h \
         src/common/quant.h src/probe/*.h src/exec/*.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(GEODE_SRC) $(LDLIBS)

clean:
	rm -rf bin

.PHONY: all clean
