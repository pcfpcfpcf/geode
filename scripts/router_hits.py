#!/usr/bin/env python3
"""Turn a GEODE_ROUTER_DUMP file into the hit-rate-vs-pool-size curve.

    GEODE_ROUTER_DUMP=dump.txt ./bin/geode exec-run MODEL.gguf "..." 400
    scripts/router_hits.py dump.txt

The question this answers: over the decoded tokens, what share of a layer's
routed-expert reads would a resident pool of K experts have served? If that
share tracks the uniform line K/n_expert, load balancing has flattened the
skew and the caching stages have nothing to trade on. If it sits well above,
the skew is real and a pool is worth its VRAM.

Two policies per layer:
  static  -- the pool holds that layer's K globally-hottest experts, fixed.
             The ceiling for any frequency-driven residency.
  lru     -- demand-paged pool of K, evicting least-recently-used. What a
             real prefetch-on-miss pool gets without knowing the future.
"""

import sys
from collections import Counter, OrderedDict


def load(path):
    n_layer = n_expert = n_used = None
    per_layer = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#"):
                for field in line[1:].split():
                    if "=" in field:
                        key, value = field.split("=")
                        if key == "n_layer":
                            n_layer = int(value)
                        elif key == "n_expert":
                            n_expert = int(value)
                        elif key == "n_expert_used":
                            n_used = int(value)
                continue
            parts = line.split()
            if not parts:
                continue
            layer = int(parts[0])
            per_layer.setdefault(layer, []).append([int(p) for p in parts[1:]])
    if n_expert is None:
        sys.exit("no header line in dump")
    return n_layer, n_expert, n_used, per_layer


def static_hit_rate(tokens, pool_size):
    counts = Counter()
    for picks in tokens:
        counts.update(picks)
    resident = {expert for expert, _ in counts.most_common(pool_size)}
    hits = total = 0
    for picks in tokens:
        total += len(picks)
        hits += sum(1 for expert in picks if expert in resident)
    return hits / total if total else 0.0


def lru_hit_rate(tokens, pool_size):
    resident = OrderedDict()
    hits = total = 0
    for picks in tokens:
        for expert in picks:
            total += 1
            if expert in resident:
                hits += 1
                resident.move_to_end(expert)
            else:
                resident[expert] = True
                if len(resident) > pool_size:
                    resident.popitem(last=False)
    return hits / total if total else 0.0


def carryover_rate(tokens):
    """Share of a token's experts also chosen by the previous token."""
    hits = total = 0
    previous = set()
    for picks in tokens[1:]:
        current = set(picks)
        total += len(current)
        hits += len(current & previous)
        previous = current
    if not total:
        previous = set(tokens[0]) if tokens else set()
        return 0.0
    return hits / total


def bar(value, width=32):
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    n_layer, n_expert, n_used, per_layer = load(sys.argv[1])
    layers = sorted(per_layer)
    n_tokens = min(len(per_layer[layer]) for layer in layers)

    pool_sizes = sorted({
        n_used,
        n_used * 2,
        max(n_used, n_expert // 8),
        n_expert // 4,
        n_expert // 2,
        (3 * n_expert) // 4,
        n_expert,
    })
    pool_sizes = [k for k in pool_sizes if 0 < k <= n_expert]

    print(f"dump:      {sys.argv[1]}")
    print(f"model:     {n_layer} layers, {n_expert} experts, "
          f"{n_used}/token")
    print(f"decoded:   {n_tokens} tokens "
          f"({n_tokens * n_used} routed reads/layer)")
    carry = sum(carryover_rate(per_layer[l]) for l in layers) / len(layers)
    print(f"carryover: {carry:6.1%}  (experts shared with the previous token; "
          f"uniform baseline {n_used / n_expert:.1%})")
    print()

    header = f"{'pool K':>7} {'K/N':>6} │ {'static hit':>11} {'lru hit':>9}"
    print(header)
    print("─" * len(header))
    for pool_size in pool_sizes:
        static = [static_hit_rate(per_layer[l], pool_size) for l in layers]
        lru = [lru_hit_rate(per_layer[l], pool_size) for l in layers]
        static_mean = sum(static) / len(static)
        lru_mean = sum(lru) / len(lru)
        uniform = pool_size / n_expert
        print(f"{pool_size:>7} {uniform:>6.1%} │ "
              f"{static_mean:>10.1%} {lru_mean:>8.1%}  "
              f"{bar(static_mean)}  (min layer {min(static):.0%})")

    print()
    lift_size = max(n_used, n_expert // 8)
    static_lift = sum(
        static_hit_rate(per_layer[l], lift_size) for l in layers
    ) / len(layers)
    uniform = lift_size / n_expert
    ratio = static_lift / uniform if uniform else 0.0
    print(f"at K={lift_size} ({uniform:.0%} of the stack): "
          f"static {static_lift:.0%} vs uniform {uniform:.0%} "
          f"— {ratio:.2f}x")
    if ratio < 1.15:
        print("verdict: routing is flat. A resident pool serves no more than "
              "its share of VRAM buys. The caching thesis is dead here.")
    elif ratio < 1.6:
        print("verdict: mild skew. A pool helps but the planner's h=0.50 "
              "assumption needs the measured number, not a guess.")
    else:
        print("verdict: strong skew. Caching is live; size the pool from the "
              "static curve above.")


if __name__ == "__main__":
    main()
