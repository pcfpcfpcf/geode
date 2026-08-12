CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDLIBS   = -ldl -pthread

PROBE_SRC = src/probe/probe.c src/probe/json.c src/probe/dram.c \
            src/probe/nvme.c src/probe/gpu.c

all: probe

probe: $(PROBE_SRC) src/probe/*.h
	$(CC) $(CFLAGS) -o $@ $(PROBE_SRC) $(LDLIBS)

clean:
	rm -f probe

.PHONY: all clean