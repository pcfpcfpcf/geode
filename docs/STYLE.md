# STYLE.md

How to write code here. Written for AI agents working in this repository;
humans welcome. Drop it into any project — nothing below is domain-specific.

Two ideas sit underneath every rule:

**Code is liability.** Every line is something to read, maintain, and get wrong.
Less code is less to keep in your head, less to break, fewer bugs.

**The next reader has no context.** They don't know what you were thinking,
which alternatives you rejected, or what the ticket said. Write for them.

Where this file and the repo's existing conventions disagree, **the repo wins.**

---

## 0. Read before you write

Spend your first minutes reading the code around your change. Match its naming,
its structure, its error handling, its comment density, its test style. A change
that is clean in isolation but foreign to its surroundings is a defect — it makes
the codebase harder to read even though the diff looks good.

Do not introduce a second way to do something the repo already does. Do not add a
dependency, a framework, or a build step to save yourself a few lines. Do not
reformat code you aren't changing; it buries your actual change in noise.

## 1. Make it obvious

Sophistication is not the goal. Obviousness is. Most unreadable code is not
clever algorithms — it is vague names, functions that do four things, and numbers
that appear from nowhere.

**Names carry the meaning a comment would.**

- Booleans read as assertions: `shared`, `expired`, `has_pending`.
- Functions are verbs, values are nouns.
- No abbreviations. `cfg` and `id` are fine; `mgr`, `tmp2`, `dat` are not.
- Prefer the domain's word over a technical one. Say what the thing *is* in the
  problem, not what container it lives in.
- A name that needs a comment to be understood is the wrong name.

**Every number gets a name.** A literal in the middle of logic is a fact nobody
can search for and nobody can change safely. Hoist it to a named constant or a
config field, and put related constants together so the tunable surface of the
module is visible in one place.

**One function, one job, one level of abstraction.** If you cannot describe what
it does without "and", split it. Nesting past three levels usually means an early
return or an extracted helper is waiting.

## 2. Say it once

Duplication means a future change has to be made in several places, and one of
them will be missed.

**Parameterize, don't fork.** Two near-identical functions that must always change
together should be one function with a parameter. The moment you write
`render_compact()` next to `render_full()`, you have created two things to keep
in sync.

**Data that must agree lives in one structure.** If a symbol has a label, a
colour, and a description, that is one table with three fields — not three
dictionaries that can drift apart.

**Generate anything derived.** Help text, docs, summaries and validation messages
that restate values should be produced from those values. A hand-written copy of
a number is stale the day someone changes it.

**But duplication of knowledge is the problem, not duplication of characters.**
Two identical lines that will change for different reasons at different times are
fine, and forcing them together couples things that should move independently.
Ask whether the two sites encode the *same fact*. If yes, unify. If they merely
look alike today, leave them.

## 3. Fewer moving parts

Prefer the boring solution that a stranger can follow.

- **Delete rather than deprecate.** When something is replaced, remove it — no
  dead branch, no unused flag, no commented-out block. Version control remembers.
- **One representation per concept.** If a value is a dict here, an object there,
  and a tuple in the third place, most of your code is conversion. Pick one shape
  and carry it end to end.
- **No class where a function will do. No layer where a parameter will do. No new
  route, hook, or abstraction where an existing one extends naturally.**
- Do not build for a requirement nobody has stated. Generality you don't need is
  cost you pay now for a benefit that may never arrive.

## 4. One error path

- Raise a typed error at the point of failure. Convert it to a user-facing
  message in **one** place per boundary.
- Never mix styles — returning `None` for failure in one function and raising in
  its neighbour makes every call site a guess.
- Don't let a low-level exception escape to a user or an API client. A stack
  trace is not an error message.
- Error text should say what went wrong *and* what would work: not `invalid
  input`, but `expected a number between 1 and 12`.
- Never swallow an error to make a red test go green.

## 5. Comments

Almost never. The code should say it.

A comment earns its place only when it records something the code *cannot* say: a
non-obvious invariant, why an obvious approach was rejected, a workaround for an
external bug, a subtle ordering requirement. If you want to explain *what* a
block does, rename or extract instead.

Never write narration (`// loop over users`), section banners, changelog comments,
or commented-out code. If the repo has a heavier docstring convention, follow the
repo — consistency beats this rule.

## 6. Output is an interface

Log lines, error messages, CLI output and API payloads are read by people, by
other programs, and increasingly by models. Design them.

- **Aggregate.** Never emit twenty near-identical lines; group by cause and report
  a count.
- **Cap unbounded lists** and say how many were omitted.
- **Keep machine-readable shapes stable.** Adding a field is cheap; renaming one
  breaks every consumer.
- Say what happened, not what function ran.

## 7. Respect the boundaries

Lower layers must know nothing about the layers above them. Domain logic should
not know it is being served over HTTP; storage should not know about the UI.

Cross a boundary through a named, documented function — not by reaching into
another module's internals. When one layer needs to attach information the other
doesn't own (request ids, session state, pagination), put it in the envelope
around the domain object, never inside it.

## 8. State that moves on its own

If the code has background work, shared mutable state, or timers, these are where
the bugs actually live:

- Guard shared mutable state with **one** discipline. Don't mix synchronous and
  asynchronous access to the same data and hope.
- Periodic work belongs to its own scheduler at its own cadence. Folding a
  fixed-interval job into a variable-interval loop means it fires at the wrong
  rate the moment the loop slows down.
- Schedule against a **monotonic deadline**, not repeated sleeps — sleeps
  accumulate drift over a long run.
- A bounded queue carrying full-state snapshots should drop the **oldest** entry
  when it fills. Dropping the newest pins a slow consumer permanently behind.

## 9. Verify before you claim

There is no credit for code that looks right.

- **Run it.** Not "this should work" — execute the path you changed.
- **Test at the layer where bugs live.** Pure logic is usually the easy part;
  wiring, concurrency, serialization and fan-out are where things actually break.
- **Make time cheap.** Inject the clock, seed the RNG, and shrink intervals so a
  five-minute lifecycle can be verified in twenty seconds. Never verify a slow
  behaviour by waiting for it once.
- **Determinism is a feature.** A seeded run that reproduces exactly is worth more
  than a realistic one you can't repeat.
- **Prove the contract, not the appearance.** Check that the shape a consumer
  destructures is really the shape produced. Confirm every case in an enum has a
  handler, so a future addition fails loudly rather than silently.
- **Re-run the check that catches your specific risk** after you change anything
  near it.
- If a test fails, say so and show the output. If you skipped a step, say which.
  Never describe unverified work as done.

## 10. Scope and honesty

- Deliver what was asked — don't quietly narrow it, widen it, or turn it into a
  different task. Finish every part; if something is blocked, complete the rest
  and say plainly what you left out and why.
- Found an adjacent problem? Mention it in your reply. Don't silently fix it in
  the same change, and don't build infrastructure for it.
- Disagree with the request? Say it in a sentence or two, then do the work under
  stated assumptions. If it's reaffirmed, that's the decision — proceed fully.
- Make routine judgment calls yourself; ask only when two readings lead to
  materially different work.
- Report outcomes faithfully. State assumptions where they matter. Correct
  mistakes plainly and move on — no ceremony, no tally.
