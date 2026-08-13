CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Isrc/common
LDLIBS   = -ldl -pthread

PROBE_SRC = src/probe/probe.c src/probe/dram.c \
            src/probe/nvme.c src/probe/gpu.c src/common/json.c

all: probe manifest planner

probe: $(PROBE_SRC) src/probe/*.h src/common/json.h
	$(CC) $(CFLAGS) -o $@ $(PROBE_SRC) $(LDLIBS)

manifest: src/manifest/manifest.c src/common/json.c src/common/json.h
	$(CC) $(CFLAGS) -o $@ src/manifest/manifest.c src/common/json.c

planner: src/planner/planner.c src/common/json.c src/common/json.h
	$(CC) $(CFLAGS) -o $@ src/planner/planner.c src/common/json.c -lm

clean:
	rm -f probe manifest planner

.PHONY: all clean
