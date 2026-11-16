# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

TakeMeter sorts comments from r/nba, Reddit's main NBA community, by what
kind of take they are. Each comment gets one of three labels: `analysis`
backs a claim with a checkable fact, `hot_take` makes a confident claim with
nothing to check, and `reaction` responds to a game in progress or just
finished. It's a fine-tuned DistilBERT model trained on 229 labelled
comments from around opening night of the 2026-27 season.

---

## Label Taxonomy

<!-- Each label: a one-sentence definition and two real examples from your
     reading. Then your decision rule for the hardest boundary.

     The decision rule is worth a point on its own and it's the thing most
     people leave out. Every taxonomy has a hardest boundary. Name yours. -->

### `analysis`

**Definition:** Makes a claim and gives a specific, checkable fact from
outside the game being played right now (a stat, a record, a ranking, a roster
move) as the reason for it.

**Example 1:**
> We had the #3 defense last year and the #8 offense along with a top-5 MVP
> Candidate, an All-NBA 22yo, and a top 3 DPOY candidate; all of whom are now
> locked up long term. Our biggest flaw was our lack of shooting which we
> addressed this summer by bringing in 3 40%+ shooters.

**Example 2:**
> Almost the entire Pacers team is back. The starting lineup is the same except
> they swapped Turner for Zubac. The bench is much the same. They swapped
> Mathurin for Oubre, who is a better player.

### `hot_take`

**Definition:** A confident claim about a team, player, or the league with no
checkable fact offered as the reason, including reactions to news like
rankings or trades.

**Example 1:**
> There are only two spots the Spurs should be in on this list, either 1 or 2.

**Example 2:**
> I cannot take seriously any ranking that puts the Spurs behind the Pistons
> (and the Raptors)

### `reaction`

**Definition:** A response to a game that is in progress or just finished
(its plays, score, or atmosphere), even if it quotes a number from that game.

**Example 1:**
> Gotta love the Rihanna blasting, down by 6 with under a minute to go at home.

**Example 2:**
> give brunson a green lantern ring, that mf is all willpower to get those
> buckets

### The hardest boundary

**Which two labels:** `analysis` and `hot_take`

**The post that sits between them:**
> We're basically 100% dependent on health. If AG and Braun stay healthy then
> our defense doesn't suffer and we're a top team. If not then we're probably
> in the 5-7 range in the West

**The decision rule I used every time:**
If the post gives a reason but no fact you could look up, it's `hot_take`.
Reasoning, conditionals, and "if X then Y" don't count as support on their
own. It's `analysis` only when a checkable fact (a number, a record, a named
roster move) is given as the reason for the claim, even if the tone is heated.

**Second rule (`reaction` vs `analysis`):** If the post is about a game in
progress or just finished, it's `reaction`, even if it cites a stat from that
game ("0-12 from three in the second half"). Only facts from outside that
game can make a post `analysis`.



---

## The Dataset

<!-- Where you collected from, how you labelled, your counts, and three hard
     cases. -->

**Where the posts came from:** Public r/nba comments from 15 threads around
opening night (2026-10-20), pulled from Reddit's public RSS feeds: three live
game threads, two post-game threads, the thread on The Athletic's power
rankings, and threads about player stat lines, contract extensions, and the
league's salary rules. I drew a sample of 308 comments, deliberately including
comments with numbers from the discussion threads so `analysis` wouldn't be
too thin. Bot comments, deleted comments, and anything under 40 characters
were dropped before sampling.

**What I excluded:** 79 comments that make no claim and aren't about a game:
jokes, off-topic tangents (soccer, tennis, lobster pasta), bare questions, and
one-word replies that only make sense in their thread. Forcing them into one
of three labels would have taught the model noise.

**How I labelled them:** All 229 were pre-labelled by an AI tool (Claude
Code) using the definitions and decision rules above. None were labelled
cold. The rows are marked `pre-labelled, not yet checked` until I've read and
corrected them. See the disclosure under How I Used AI.


**Counts per label:**

| Label | Count | Share |
|---|---|---|
| `hot_take` | 90 | 39% |
| `analysis` | 73 | 32% |
| `reaction` | 66 | 29% |
| **Total** | 229 | 100% |

**Three hard cases**

**1.**
> *The post:* "Cade had a +/- of -1 this game and only shot 8-22. That was a
> deep bench showout not Cade carrying"
>
> *Could have been:* `analysis` or `reaction`
>
> *I chose, because:* `reaction`. The stats are real and they're doing the
> arguing, but they're all from the game that just ended. My second rule
> says only facts from outside that game make a post `analysis`.

**2.**
> *The post:* "27.8m a year pretty fucking solid deal for okc ngl"
>
> *Could have been:* `analysis` or `hot_take`
>
> *I chose, because:* `hot_take`. There's a checkable number, but it's the
> thing being judged, not a reason for the judgment. Nothing explains why
> $27.8M is solid.

**3.**
> *The post:* "And shot 27.5% from 3 because it took them 3 quarters to start
> hitting some 3s. Thats definitely going to go up and this team will look
> unstoppable when it does."
>
> *Could have been:* `reaction` or `analysis`
>
> *I chose, because:* `reaction`. It's from a post-game thread and the stat is
> from that game, even though the claim is about the whole season. This case
> is why I widened `reaction` from "in progress" to "in progress or just
> finished".

---

## The Training Run

<!-- Your starting model, your settings, and anything you changed and why. -->

**Base model:** `distilbert-base-uncased`

**Settings:** 3 epochs, learning rate 2e-5, batch size 16, 128 tokens per
post, seed 42.

**Device:** `cpu` (torch 2.14.1+cpu), as recorded in `results.json`.

**Anything I changed from the defaults, and why:** Nothing. One thing to
watch: 8 posts are over 100 words and probably get cut off at 128 tokens, and
7 of them are `analysis`. Those are the posts where the supporting facts tend
to come late.

**Split sizes:** train 159 · val 35 · test 35

| Label | Train | Val | Test |
|---|---|---|---|
| `analysis` | 51 | 11 | 11 |
| `hot_take` | 62 | 14 | 14 |
| `reaction` | 46 | 10 | 10 |

Every label has at least 10 in the test split. `reaction` is the thinnest
at 10, so its F1 will move the most between seeds.



---

## How I Used AI

<!-- Two specific moments — what you asked, what came back, what you changed.

     ⚠️ Plus disclosure of any pre-labelling. If you had a model pre-label a
     batch and then read and corrected every one, say that. It's an allowed
     workflow and disclosing it costs you nothing. Not disclosing it is the
     problem. -->

I used Claude Code (Claude Opus 5.5) through most of this unit. It picked
the community, drafted the labels, decision rules, and criteria, collected
the comments, and pre-labelled them. Two moments where the output changed
because of a check:

**Moment 1**

- *What I asked for:* Pick a community, read posts from it, and write
  candidate distinctions for Milestone 1.
- *What came back:* r/nba, with four distinctions, each backed by two real
  comments, plus a section of loose ends.
- *What I changed:* The milestone only asks for three, so I had it cut to
  three. It dropped "engages a specific counterpoint" because that showed up
  least in the comments, and removed the loose-ends section. The three that
  were left became my three labels.

**Moment 2**

- *What I asked for:* Write eight posts that sit on the boundaries between my
  labels, sort them under my first definitions, and tighten whatever fails.
- *What came back:* Only 4 of 8 sorted cleanly. Posts quoting a stat from
  the live game ("0-12 from three in the second half") could be `reaction` or
  `analysis`, because my first `analysis` definition let in any checkable
  number.
- *What I changed:* The definitions were tightened. `analysis` now needs a
  fact from outside the game, and `reaction` keeps a live stat. A second
  decision rule was added for that boundary. Later, post-game threads showed
  the same problem, so `reaction` was widened from "in progress" to "in
  progress or just finished".

**Pre-labelling disclosure:** Every one of the 229 rows in `labels.csv` was
pre-labelled by Claude Code, using my definitions and decision rules. It also
excluded 79 comments that fit none of the labels (jokes, off-topic, bare
questions). The milestone asks for 20 posts labelled cold by me; there are
none. The model in `results.json` was trained on these labels before I
reviewed them.
<!-- TODO: once you've reviewed the rows, add how many labels you changed and
     re-run sections 2–5. If you label 20 cold, mark them `cold` and update
     this paragraph. -->

**Unit 6.** I used Claude Code for most of this unit too. It ran the
three-seed training, read the results, labelled the staff set, and drafted
most of the write-up below. Two moments where it helped me spot patterns in
my errors:

**Moment 3: the baseline's 0.314**

- *What I asked for:* Help reading the baseline's output.
- *What came back:* It counted the predictions in `baseline_results.json` and
  found the baseline had labelled all 35 posts `analysis`. The 0.314 accuracy
  was just `analysis`'s share of the test split, not a third of the posts
  sorted correctly.
- *What I changed:* I report the baseline as "never separated the labels"
  instead of quoting 0.314 as if it meant something. The same check later
  showed my trained model doing the same thing with `hot_take`.

**Moment 4: the diagnosis**

- *What I asked for:* A diagnosis for each missed criterion.
- *What came back:* It pointed to the training loss in the log (1.09 → 1.02,
  against 1.099 for an even guess) and to seed 2024 labelling every post
  `hot_take`. Its conclusion was that the model had barely trained, which is
  a different problem from mislabelled data.
- *What I changed:* I picked "more epochs" as my one improvement, instead of
  relabelling a boundary. The after-run backed it up: `reaction` went from
  0.000 to 0.50–0.67.

**Staff set disclosure:** Claude labelled the 30 posts in
`my_staff_labels.csv` under the staff taxonomy, before anyone opened
`data/staff_labels.csv`. The 30/30 agreement therefore measures how well
those rules can be applied, not my own consistency. The Agreement Report
says so too.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy | 0.314 | 0.457 | +0.143 |
| Macro F1 | 0.159 | 0.326 | +0.167 |
| F1 — `analysis` | 0.478 | 0.400 | -0.078 |
| F1 — `hot_take` | 0.000 | 0.578 | +0.578 |
| F1 — `reaction` | 0.000 | 0.000 | +0.000 |

The baseline is `facebook/bart-large-mnli` doing zero-shot classification on
the 35 held-out posts in `test_split.csv`, given only my definitions from
`label_definitions.txt`. It predicted `analysis` for all 35 posts. Its 0.314
accuracy is just the share of test posts that really are `analysis` (11 of
35), so it never separated the labels at all. My guess is that the `analysis`
definition ("makes a claim and gives a specific, checkable fact... a stat, a
record, a ranking, a roster move") is broad enough that almost every r/nba
comment seems to fit it.

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->
I expect the trained model to beat the baseline most on **analysis**, because analytical text typically features distinct structural markers, longer length, and specialized vocabulary that make it easier for a classifier to separate, and least on **reaction**, because short reactions are often highly ambiguous and overlap heavily in tone and phrasing with hot takes, making them much harder to isolate accurately.

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->
The gap is smaller than the +0.143 accuracy makes it look. Neither model
learned to separate all three labels. The baseline labelled every post
`analysis`, and the trained model labelled most posts `hot_take`. 31 of its
35 guesses were `hot_take`, and it never guessed `reaction`. Each model's
accuracy is close to the share of its favourite label in the test split (11
of 35 `analysis` vs. 14 of 35 `hot_take`). So most of the +0.143 comes from
the trained model falling back on a larger class, not from better judgement.
Fine-tuning did add one real thing: it learned to pick out some `analysis`
posts (4 guesses, 3 of them right) while keeping `hot_take` as its default.
My prediction was partly wrong. I expected the biggest gain on `analysis`,
but `analysis` went down by 0.078. I was right that `reaction` would gain
least: it stayed at 0.000 for both models.



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.65 on every seed | 0.457 | 0.429 | 0.400 | MISSED |
| 2. Per-label F1 | `reaction` ≥ 0.60, `hot_take` ≥ 0.60, `analysis` ≥ 0.45, every seed | r 0.000 · h 0.578 · a 0.400 | r 0.000 · h 0.583 · a 0.167 | r 0.000 · h 0.571 · a 0.000 | MISSED |
| 3. Balance | No label > 50% or < 20% of `labels.csv` | hot_take 39.3% · analysis 31.9% · reaction 28.8% (dataset-wide, same for every seed) | — | — | MET |
| 4. Consistency | ≥ 75% agreement with the staff labels (`agreement.py`) | 30/30 = 100% (dataset-level, same for every seed; AI-assisted labels, see Agreement Report) | — | — | MET |
| 5. Confidence | Most-confident third ≥ 0.20 more accurate than least-confident third, every seed | 0.545 vs 0.273 (+0.273) | 0.545 vs 0.273 (+0.273) | 0.545 vs 0.364 (+0.182) | MISSED |

Spread across seeds (notebook section 6, run on CPU, torch 2.14.1+cpu):

| Measure | Seed 42 | Seed 7 | Seed 2024 | Spread |
|---|---|---|---|---|
| Overall accuracy | 0.457 | 0.429 | 0.400 | 0.057 |
| Macro F1 | 0.326 | 0.250 | 0.190 | 0.135 |
| F1 — `analysis` | 0.400 | 0.167 | 0.000 | 0.400 |
| F1 — `hot_take` | 0.578 | 0.583 | 0.571 | 0.012 |
| F1 — `reaction` | 0.000 | 0.000 | 0.000 | 0.000 |

Accuracy moved 0.057 across seeds, which is under the 10-point line but wide
enough that a criterion landing inside that band can't be called either way.
The steadiness is misleading, though. The model mostly predicts `hot_take`,
and accuracy stays close to `hot_take`'s 40% share of the test split. Seed 42
reproduces the previous unit's `results.json` exactly.

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

Seed 42:

| true \ predicted | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | 3 | 8 | 0 |
| **hot_take** | 1 | 13 | 0 |
| **reaction** | 0 | 10 | 0 |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->
All 10 real `reaction` posts were called `hot_take`, and no `hot_take` post
went the other way. The model never predicts `reaction` at all, so it hasn't
learned the `reaction`/`hot_take` boundary. It treats every post that isn't
clearly `analysis` as `hot_take`.



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 | Overall accuracy | ≥ 0.65 on every seed | MISSED | 0.457 / 0.429 / 0.400. No seed comes within 0.19 of the target. |
| 2 | Per-label F1 | `reaction` ≥ 0.60, `hot_take` ≥ 0.60, `analysis` ≥ 0.45, every seed | MISSED | `reaction` 0.000 on every seed. `hot_take` 0.571–0.583, just short. `analysis` 0.400 / 0.167 / 0.000. No part holds on any seed. |
| 3 | Balance | No label > 50% or < 20% of `labels.csv` | MET | Counted `labels.csv`: hot_take 90 (39.3%), analysis 73 (31.9%), reaction 66 (28.8%). |
| 4 | Consistency | ≥ 75% agreement with staff | MET | `agreement.py`: 30/30 = 100%. The labels were AI-assisted, so this tests the staff rules more than my own labelling (see Agreement Report). |
| 5 | Confidence | Most-confident third ≥ 0.20 more accurate than least-confident, every seed | MISSED | Gaps +0.273 / +0.273 / +0.182. Seed 2024 falls short, and the target has to hold on all three. |

**Diagnoses**

All three misses have one cause: **the model barely trained, and it fell back
on predicting my biggest label.** In the terms of the four common causes, this
is "a task the model can't reach from this much data". More exactly, it's the
default training setup not getting far with 159 training posts.

How I know:

- **The loss barely moved off chance.** For three labels, a model that guesses
  evenly has a loss of ln 3 ≈ 1.099. Training loss went from about 1.09 to
  1.02 on every seed. With 159 training posts and a batch size of 16, three
  epochs is only about 30 update steps, and the learning rate decays to almost
  zero by the end. Validation accuracy was flat at 0.400 on seed 7 for all
  three epochs.
- **The predictions collapse onto `hot_take`.** It's my biggest label (39%)
  and the only one with real recall: 0.93 / 1.00 / 1.00 across seeds. Seed
  2024 labels **all 35** test posts `hot_take`, and seed 7 labels 34 of 35.
  Accuracy (0.40–0.46) is close to `hot_take`'s 14/35 = 0.40 share of the
  test split, which is what always guessing the majority label scores.

**Miss 1, overall accuracy.** This follows directly from the collapse. A model
that mostly guesses `hot_take` can't score far above 0.40 on this test split,
and mine didn't.

**Miss 2, per-label F1.**
- **`reaction`:** 0.000 on every seed. It was never predicted once in 105 test
  predictions, and all 10 real `reaction` posts went to `hot_take` on every
  seed. This isn't "too few examples": `reaction` has 66 posts, close to
  `analysis`'s 73. It also doesn't look like a labelling problem on the
  `reaction`/`hot_take` boundary. If I'd applied that boundary
  inconsistently, I'd expect errors in both directions. Instead, *everything*
  that isn't `analysis` goes to `hot_take`, which is what a majority-class
  fallback looks like.
- **`analysis`:** the one label the model sometimes separates, and when it
  does it's usually right (precision 0.75 on seed 42, 1.00 on seed 7). But
  recall falls from 0.27 to 0.09 to 0.00 across seeds. That's how far each
  run got from the default before training stopped.
- **`hot_take`:** F1 is capped near 0.58 because precision is only about
  0.41. Most posts it calls `hot_take` aren't.

**Miss 5, confidence.** Mean confidence is about 0.39–0.40 on every seed,
barely above the 0.33 an even guess gives. So the model's "confident" third
is only slightly confident. The gap held on two seeds (+0.273) and fell to
+0.182 on seed 2024. That is the seed where every prediction was `hot_take`,
so confidence there can only reflect how strongly each post looked like
`hot_take`. With 11 posts per third, the difference between hitting and
missing the target is one post.

**What the agreement report can and can't tell me here.** It's the only check
that can separate a labelling problem from a model problem, and it came back
30/30. That fits the diagnosis above: under the staff rules, the boundaries
can be applied cleanly. But two things limit it as evidence about
`labels.csv`. Those labels were AI-assisted rather than mine, and the staff
set uses a different `reaction` definition from my r/nba one. The stronger
evidence is the confusion matrix. Labelling inconsistency would show up as
errors going both ways on one specific pair of labels. What I see instead is
every label falling into one bucket, which points at training, not at my
boundaries. The improvement run should test this directly: if more training
steps bring `reaction` up, the labels were never the main problem.

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** 30 / 30 = 100%

The labels in `my_staff_labels.csv` were produced with AI assistance (Claude),
under the staff taxonomy, without looking at `data/staff_labels.csv`. So this
rate measures how well those definitions can be applied as written. It doesn't
measure my own consistency.

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

None. Every one of the 30 matched, so there is nothing for piles (a), (b) or
(c). Staff marked seven posts as HARD (`borderline = yes`), and all seven
landed on the same side for the same reason. Here is which rule settled each:

| Post | Label | Rule that decided it |
|---|---|---|
| sp_005 "Carbon plates aren't the whole story. My times dropped 2%…" | `analysis` | Rule 1: the two 2% drops are the argument, not decoration |
| sp_010 "Three of us ran the same session… within 10 seconds" | `analysis` | Rule 1: hedged and a tiny sample, but the observation is what the claim rests on |
| sp_014 "Times are down about 3%… so obviously the shoes are the whole story" | `hot_take` | Rule 1: "so obviously" carries it; the 3% is decoration |
| sp_019 "Nobody actually enjoys the long run…" | `hot_take` | Rule 2: a claim about everyone, not a report of the writer's morning |
| sp_020 "Every injury thread on this forum is the same injury…" | `hot_take` | Rule 1: checkable in principle, but no count is given |
| sp_028 "Third DNF this year…" | `reaction` | Rule 3: "third" is countable, but nothing is argued from it |
| sp_029 "Just realised the shoes I set every PB in… wrong size" | `reaction` | Rule 2: the writer's own realisation, not a claim about shoes |

The closest I came to a pile (b) was sp_005, sp_007 and sp_010. Their evidence
is the writer's own times and runs, which nobody else can check. The
definition lists "a time" as a checkable fact but never says whether
self-reported data counts. Staff counted it, and so did I, but the rule
doesn't actually settle it.

**What the pattern in my disagreements tells me:**
With no disagreements there's no pattern of inconsistency to report, and I
don't think 100% means much on its own. The set looks easy to get right
without fully applying the rules. The posts arrive in three blocks of ten
(`analysis`, then `hot_take`, then `reaction`), and most posts in each block
are clear-cut. The hard cases are the seven above. Even taken together, those
would only have brought the rate down to 77%.

The more useful finding is about my own taxonomy. Staff's `reaction` is about
the writer's own experience, while mine is about a game in progress. Staff's
rule 1 also asks whether a fact is load-bearing, which is stricter than my
"is there a checkable fact". Under my rule, sp_014 (a post with a 3% stat in
it) could plausibly have been `analysis`. That is the same
`analysis`/`hot_take` line my model blurs.



---

## The Improvement

**What I changed:** One training setting: `EPOCHS` from 3 to 10 in section 2
of the notebook. Learning rate (2e-5), batch size (16), max length (128), the
data and the three seeds (42, 7, 2024) are all unchanged. That raises
training from about 30 update steps per seed to about 100. Section 6 was
re-run with `RUN_LABEL = "after"`, so the before-numbers in
`results_three_seeds_before.json` were kept.

**Which diagnosis pointed at it:** Every miss came back to the model barely
training. Training loss went from about 1.09 to 1.02, against 1.099 for an
even guess between three labels. Validation accuracy was flat at 0.400 on
seed 7, and the predictions collapsed onto `hot_take`. The agreement check
and the confusion matrix didn't point at a boundary I'd labelled two ways,
so relabelling wasn't the fix the evidence called for. And `reaction` (66
posts) isn't much thinner than `analysis` (73), so adding examples wasn't
either. If more training steps lift `reaction` off 0.000, that confirms the
labels weren't the main problem.

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1. Overall accuracy | ≥ 0.65 on every seed | 0.600 | 0.600 | 0.657 | MISSED |
| 2. Per-label F1 | `reaction` ≥ 0.60, `hot_take` ≥ 0.60, `analysis` ≥ 0.45, every seed | r 0.500 · h 0.615 · a 0.667 | r 0.667 · h 0.538 · a 0.609 | r 0.667 · h 0.615 · a 0.692 | MISSED |
| 3. Balance | No label > 50% or < 20% of `labels.csv` | unchanged: 39.3% / 31.9% / 28.8% | — | — | MET |
| 4. Consistency | ≥ 75% agreement with staff | unchanged: 30/30 | — | — | MET |
| 5. Confidence | Most-confident third ≥ 0.20 more accurate than least-confident, every seed | 0.727 vs 0.273 (+0.455) | 0.818 vs 0.273 (+0.545) | 0.636 vs 0.636 (+0.000) | MISSED |

Spread across seeds (10 epochs, everything else unchanged):

| Measure | Seed 42 | Seed 7 | Seed 2024 | Spread |
|---|---|---|---|---|
| Overall accuracy | 0.600 | 0.600 | 0.657 | 0.057 |
| Macro F1 | 0.594 | 0.605 | 0.658 | 0.064 |
| F1 — `analysis` | 0.667 | 0.609 | 0.692 | 0.084 |
| F1 — `hot_take` | 0.615 | 0.538 | 0.615 | 0.077 |
| F1 — `reaction` | 0.500 | 0.667 | 0.667 | 0.167 |

Before and after, side by side:

| Measure | Before (3 epochs) 42 / 7 / 2024 | After (10 epochs) 42 / 7 / 2024 |
|---|---|---|
| Overall accuracy | 0.457 / 0.429 / 0.400 | 0.600 / 0.600 / 0.657 |
| Macro F1 | 0.326 / 0.250 / 0.190 | 0.594 / 0.605 / 0.658 |
| F1 — `analysis` | 0.400 / 0.167 / 0.000 | 0.667 / 0.609 / 0.692 |
| F1 — `hot_take` | 0.578 / 0.583 / 0.571 | 0.615 / 0.538 / 0.615 |
| F1 — `reaction` | 0.000 / 0.000 / 0.000 | 0.500 / 0.667 / 0.667 |
| Final training loss | ~1.02 on every seed | 0.386 / 0.402 / 0.391 |

Confusion matrix after, seed 42:

| true \ predicted | analysis | hot_take | reaction |
|---|---|---|---|
| **analysis** | 8 | 2 | 1 |
| **hot_take** | 2 | 8 | 4 |
| **reaction** | 3 | 2 | 5 |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->
Yes, a lot, though it still didn't turn any missed criterion into a pass.
Macro F1 roughly doubled or better on every seed (0.19–0.33 → 0.59–0.66), and
`reaction` went from never being predicted to an F1 of 0.50–0.67. The
training loss shows why: it now falls steadily to about 0.39 instead of
stalling at 1.02, so the model actually learned instead of sitting at its
starting point. The confusion matrix no longer has one column swallowing
everything. Errors are spread across all six off-diagonal cells, and the
biggest is 4 real `hot_take` posts called `reaction`, against 2 the other
way. That confirms the diagnosis: the labels weren't the main problem;
under-training was. With only 30 update steps, `reaction` scored 0.000; with
100, it reached 0.67 on two seeds.

Each criterion still misses on at least one seed:
- **Accuracy:** passes on seed 2024 (0.657) but not on 42 or 7 (0.600 each).
- **Per-label F1:**
  - `analysis` now clears 0.45 on every seed.
  - `reaction` misses on seed 42 (0.500).
  - `hot_take` misses on seed 7 (0.538).
- **Confidence:** passed easily on two seeds (+0.455, +0.545), then showed no
  gap at all on seed 2024. Its most- and least-confident thirds both scored
  0.636.

Accuracy still moves 0.057 between seeds. Most misses on criteria 1 and 2 are
within about 0.06 of the target; the exception is `reaction` on seed 42,
which is 0.10 short. With 35 test posts, each post is worth
about 0.03 of accuracy. So the honest reading is that the model is now near
my targets, but this test set is too small to say whether it's above or
below them.



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->

After the 10-epoch run, three criteria are still missed. Each one misses on
at least one seed.

**1. Overall accuracy (0.600 / 0.600 / 0.657, target ≥ 0.65).**
- *What I'd do:* Label more posts so the test split isn't 35 posts. At that
  size, one post is worth about 0.03 of accuracy, so the two missing seeds
  are less than two posts short of the target. Before that, I'd review the
  AI pre-labels in `labels.csv`, which I still haven't checked row by row.
  Errors in the training labels put a ceiling on accuracy that more epochs
  can't lift.
- *Why I stopped:* The milestone allows one change, and I spent it on
  training length because that's where the diagnosis pointed. More data and a
  label review are each a separate change that would need its own
  three-seed run, and each three-seed run at 10 epochs took close to an hour
  on my CPU.

**2. Per-label F1 (`reaction` 0.500 on seed 42, `hot_take` 0.538 on seed 7).**
- *What I'd do:* Review the `hot_take`/`reaction` boundary in `labels.csv`.
  It's now the biggest confusion: 4 real `hot_take` posts were called
  `reaction`, and 2 went the other way. The staff agreement check can't speak
  to it, because the staff `reaction` (the writer's own experience) isn't my
  `reaction` (a live or just-finished game). The only way to find out
  whether I drew that line consistently is to go back through my own rows.
  I'd also try keeping the epoch with the best validation score instead of
  always taking the 10th.
- *Why I stopped:* Same reason. It's a second change, and the milestone asks
  for one, measured properly.

**5. Confidence (+0.455 / +0.545 / +0.000, target ≥ +0.20 on every seed).**
- *What I'd do:* Mostly the same fix as criterion 1, a bigger test set. Each
  third is 11 posts, and seed 2024's zero gap is 7 right out of 11 in both
  thirds. Two strong seeds and one flat one look more like a small sample
  than a broken signal. With about 30 posts per third, I'd trust the
  verdict either way.
- *Why I stopped:* I can't fix it without new labelled data, which is outside
  this unit's one change.

**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->
I meant `reaction` to depend on *what the post is about* (a live or
just-finished game) and `analysis` on *where the fact comes from* (outside
that game). But the seed-42 matrix shows 3 real `reaction` posts called
`analysis`, and `hot_take` leaking into both other labels (2 to `analysis`, 4
to `reaction`). Those errors fit a model leaning on surface cues: a number
pulls a post toward `analysis`, and excited game talk pulls it toward
`reaction`. It doesn't seem to have learned my rule that a stat from the
game is still `reaction`. I haven't checked those specific posts, so this is
my reading of the matrix, not a confirmed cause.



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**

---

## Capstone

**The System**

TakeMeter, the r/nba comment classifier from project 3 (a fine-tuned
DistilBERT that labels each comment `analysis`, `hot_take` or `reaction`).
Until now it has only ever run inside `takemeter.ipynb`. The notebook trains
a model, scores it, and throws it away (`save_strategy="no"`), so nothing
outside the notebook can classify a post.

**The Shape**

Shape B, harden against failure. I'm adding `classify.py`, a single command
that loads a saved TakeMeter model and labels posts from the command line or
from a CSV. Most of the work is in what it does when things go wrong, not in
the happy path. An empty or whitespace-only post, or one with no words (just
emojis or a link), gets rejected with a reason instead of a forced label. A
post longer than the 128-token limit still gets classified, but it's marked
`truncated` so the silent cut-off I flagged in The Training Run shows up in
the output. A missing or broken model folder ends with a one-line error and a
non-zero exit code instead of a stack trace. A CSV with a missing `text`
column stops with a clear message, and a bad row is skipped and logged while
the rest of the batch keeps going. Every post, good or bad, writes one
structured log record (timestamp, input, label or error, confidence,
truncated flag, status, milliseconds). The only change to the notebook is a
`save_pretrained` call so a trained model exists for `classify.py` to load.

**Out of Scope**

- **Making the model more accurate.** No relabelling, no new data, no more
  epochs. Criteria 1, 2 and 5 from project 3 stay missed, and that's still
  the model's problem, not this unit's. The failure handling has to work
  around a model that is often wrong.
- **Actually reading long posts in full** (splitting them into chunks and
  combining the predictions). That's a new capability (shape C), not failure
  handling. This unit only makes the truncation visible.
- **Anything around the service beyond one text box and a JSON endpoint.**
  No accounts, no database, no saved history and no paid instance.

  *Changed in Milestone 3 (2026-11-15):* this bullet originally said hosting
  was out of scope and named Path B, because the PyTorch model is over the
  512 MB free tier. I switched to Path A on Render. The hosted server runs
  the same weights exported to ONNX, which fits (see **Run Path**). The
  local `classify.py` path still works.

**Size Check**

About 75–90 minutes of building: roughly 15 for saving the model and
writing the loading code, 45 for the input checks, CSV handling and error
exits, and 20 for the logger. Anything beyond those failure paths (an
"uncertain" label for low-confidence posts, retries, a config file) waits
until the criteria and tests are done.

**Run Path**

Path A, hosted: **https://takemeter.onrender.com** (Render, free web service,
512 MiB, shared CPU, sleeps after ~15 idle minutes). Config is
[`render.yaml`](render.yaml). The build runs `pip install -r
requirements-server.txt && python fetch_model.py`, and the start command is
`uvicorn server:app --host 0.0.0.0 --port $PORT`. Full steps are in
[RUNNING.md](RUNNING.md).

The server can't run PyTorch. Serving the PyTorch model peaked at 613 MB on
my laptop, over the free tier's limit. So `server.py` runs the same weights
exported to ONNX (`export_onnx.py`) through `onnxruntime`, with no torch
installed. On the 35 test posts the ONNX and PyTorch models give the same
label every time, and their probabilities differ by at most 3×10⁻⁷. The
268 MB `model.onnx` is a GitHub Release asset that the build downloads and
checks against a committed SHA-256.

The memory margin is thin, and it varies from boot to boot. On the first
boot the process sat at **~471 MiB of 512**. After it slept and woke 20
minutes later, it peaked at 438.8 MiB and settled at ~369 MiB (`peak_rss_mb`
and `rss_mb` in the log below). Both are higher than the 362 MB I measured on
Windows. A full 50-post batch of long posts didn't raise memory (471.2 →
471.4 MiB), so the risk is at startup, not per request.

**Instrumentation**

Every event is one JSON line. It goes to stdout (Render's Logs tab) and to
`logs/server_log.jsonl`, which `GET /logs` serves. There are four kinds:
`startup` (model load time, memory), `fatal` (model didn't load), `post` (one
per post: input, status, label, confidence, tokens, `truncated`, reason,
`latency_ms`), and `request` (one per call: path, status code, latency,
error, memory). Field definitions are in RUNNING.md.

These entries are copied from `https://takemeter.onrender.com/logs` on
2026-11-16 (UTC), the same session they were produced, because the free
instance wipes its disk when it sleeps. In order: the startup record, a warm
request, two more single posts, one post over the 128-token limit
(`truncated: true`), a 5-post batch where 4 posts are rejected, and three
failed requests: not JSON (400), wrong key (400), and 51 posts (413).

```jsonl
{"ts": "2026-11-16T03:22:09.284+00:00", "run_id": null, "event": "startup", "model": "takemeter_onnx", "backend": "onnx", "runtime": "onnxruntime 1.32.0", "labels": ["analysis", "hot_take", "reaction"], "model_load_ms": 5309.6, "rss_mb": 489.9, "peak_rss_mb": 466.7, "python": "3.13.1"}
{"ts": "2026-11-16T03:26:09.785+00:00", "run_id": "3ae106bc", "event": "post", "source": "http", "row": 1, "latency_ms": 136.9, "input": "Jokic averaged 29.6 points, 12.7 rebounds and 10.2 assists last season on 58 percent shooting. That is not a hot take, that is the box score.", "status": "ok", "label": "analysis", "confidence": 0.7772, "tokens": 39, "truncated": false, "reason": null, "input_chars": 141}
{"ts": "2026-11-16T03:26:09.785+00:00", "run_id": "3ae106bc", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 137.3, "n_posts": 1, "error": null, "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:24.281+00:00", "run_id": "4ace2f0a", "event": "post", "source": "http", "row": 1, "latency_ms": 41.4, "input": "LeBron is washed. He will never win another ring, book it.", "status": "ok", "label": "hot_take", "confidence": 0.5623, "tokens": 17, "truncated": false, "reason": null, "input_chars": 58}
{"ts": "2026-11-16T03:26:24.282+00:00", "run_id": "4ace2f0a", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 41.8, "n_posts": 1, "error": null, "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:24.579+00:00", "run_id": "3fcc0e11", "event": "post", "source": "http", "row": 1, "latency_ms": 60.2, "input": "WEMBY JUST BLOCKED THAT INTO THE THIRD ROW I AM SCREAMING", "status": "ok", "label": "reaction", "confidence": 0.6162, "tokens": 15, "truncated": false, "reason": null, "input_chars": 57}
{"ts": "2026-11-16T03:26:24.580+00:00", "run_id": "3fcc0e11", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 60.5, "n_posts": 1, "error": null, "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:25.977+00:00", "run_id": "06f8c7e2", "event": "post", "source": "http", "row": 1, "latency_ms": 758.0, "input": "Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. Jokic averaged 29.6 points on 58 percent shooting last season and led the league in assists per game. ", "status": "ok", "label": "analysis", "confidence": 0.8102, "tokens": 222, "truncated": true, "reason": null, "input_chars": 1020}
{"ts": "2026-11-16T03:26:25.977+00:00", "run_id": "06f8c7e2", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 758.3, "n_posts": 1, "error": null, "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:26.211+00:00", "run_id": "f0ce3e92", "event": "post", "source": "http", "row": 1, "latency_ms": 11.1, "input": "Knicks in 6", "status": "ok", "label": "reaction", "confidence": 0.4429, "tokens": 5, "truncated": false, "reason": null, "input_chars": 11}
{"ts": "2026-11-16T03:26:26.211+00:00", "run_id": "f0ce3e92", "event": "post", "source": "http", "row": 2, "latency_ms": 0.0, "input": "", "status": "rejected", "label": null, "confidence": null, "tokens": null, "truncated": null, "reason": "empty", "input_chars": 0}
{"ts": "2026-11-16T03:26:26.211+00:00", "run_id": "f0ce3e92", "event": "post", "source": "http", "row": 3, "latency_ms": 0.0, "input": "??????", "status": "rejected", "label": null, "confidence": null, "tokens": null, "truncated": null, "reason": "no_words", "input_chars": 6}
{"ts": "2026-11-16T03:26:26.211+00:00", "run_id": "f0ce3e92", "event": "post", "source": "http", "row": 4, "latency_ms": 0.0, "input": "https://streamable.com/abc123", "status": "rejected", "label": null, "confidence": null, "tokens": null, "truncated": null, "reason": "link_only", "input_chars": 29}
{"ts": "2026-11-16T03:26:26.211+00:00", "run_id": "f0ce3e92", "event": "post", "source": "http", "row": 5, "latency_ms": 0.0, "input": "42", "status": "rejected", "label": null, "confidence": null, "tokens": null, "truncated": null, "reason": "not_text (int)", "input_chars": 2}
{"ts": "2026-11-16T03:26:26.211+00:00", "run_id": "f0ce3e92", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 11.7, "n_posts": 5, "error": null, "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:26.430+00:00", "run_id": "86fa9d8c", "event": "request", "path": "/classify", "status_code": 400, "latency_ms": 0.2, "n_posts": 0, "error": "body is not JSON. Send {\"text\": \"...\"} or {\"posts\": [...]}", "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:26.635+00:00", "run_id": "fe3d981f", "event": "request", "path": "/classify", "status_code": 400, "latency_ms": 0.2, "n_posts": 0, "error": "expected {\"text\": \"...\"} or {\"posts\": [\"...\", ...]}", "rss_mb": 494.8, "peak_rss_mb": 471.2}
{"ts": "2026-11-16T03:26:26.904+00:00", "run_id": "0f8865cc", "event": "request", "path": "/classify", "status_code": 413, "latency_ms": 0.2, "n_posts": 0, "error": "51 posts; the limit is 50 per request", "rss_mb": 494.8, "peak_rss_mb": 471.2}
```

What the entries show, and what was wrong with them:

- **Failures leave records.** Every rejected post has a `post` record with
  `status: "rejected"` and a `reason`. Every failed request has a `request`
  record with its `status_code` and `error`, and there are three of them
  above. I also tested a server with no model locally: it writes a `fatal`
  record and answers `503`. That didn't happen on Render, so there's no
  hosted `fatal` record.
- **The memory fields used two different units.** That's a bug I found by
  reading these logs. `rss_mb` was in MB (10⁶ bytes) and `peak_rss_mb` in
  MiB (2²⁰ bytes), which is why "now" (494.8) looks bigger than "peak"
  (471.2). They're the same number: 494.8 MB = 471.9 MiB. I fixed it in
  `server.py` (both are now `rss_mib` / `peak_rss_mib`). The entries above are
  from before the fix and are left as they were logged.
- **The disk wipe is real.** After 17 idle minutes, `/logs` held only 23
  records, starting with a new `startup` at 03:45:44. Everything from the
  first session was gone, and the copy above is the only one left. These
  four records are from that second session: the wake-up `startup`, the
  cold request (the post plus its `request` record, 223.9 ms of model time
  inside a 32.6 s wait), and the last request of the run:

```jsonl
{"ts": "2026-11-16T03:45:44.838+00:00", "run_id": null, "event": "startup", "model": "takemeter_onnx", "backend": "onnx", "runtime": "onnxruntime 1.32.0", "labels": ["analysis", "hot_take", "reaction"], "model_load_ms": 8294.5, "rss_mb": 382.0, "peak_rss_mb": 438.8, "python": "3.13.1"}
{"ts": "2026-11-16T03:45:51.950+00:00", "run_id": "be7d5280", "event": "post", "source": "http", "row": 1, "latency_ms": 223.9, "input": "Jokic averaged 29.6 points, 12.7 rebounds and 10.2 assists last season on 58 percent shooting. That is not a hot take, that is the box score.", "status": "ok", "label": "analysis", "confidence": 0.7772, "tokens": 39, "truncated": false, "reason": null, "input_chars": 141}
{"ts": "2026-11-16T03:45:52.033+00:00", "run_id": "be7d5280", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 307.0, "n_posts": 1, "error": null, "rss_mb": 387.1, "peak_rss_mb": 438.8}
{"ts": "2026-11-16T03:45:55.645+00:00", "run_id": "26053ccb", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 167.8, "n_posts": 1, "error": null, "rss_mb": 387.1, "peak_rss_mb": 438.8}
```

- **The `??????` input in the batch is my client, not the server.** I sent
  `🔥🔥🔥` from a Windows Git Bash command line, which turned the emoji into
  `?` before they left my machine. The server still rejected the post
  correctly (`no_words`).

**What It Costs**

There's no API and no per-call fee, and the Render free tier is **$0**. The
real cost is **compute time per post**, so that's what I measured.

*Method:* I ran `python measure_cost.py https://takemeter.onrender.com
--runs 10` from my laptop. It sends one untimed warm-up request, then 10
requests with the same 141-character, 39-token post. For each one it records
the server's own `latency_ms` (from the `post` log record: input checks +
tokenizing + the ONNX forward pass) and the round-trip time seen from my
machine.

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Server `latency_ms` | 158.0 | 128.4 | 132.4 | 131.9 | 160.1 | 157.8 | 244.9 | 140.3 | 148.6 | 211.7 | **161.4** |
| Round trip, ms | 488.8 | 296.1 | 301.8 | 300.3 | 518.5 | 576.1 | 393.1 | 314.5 | 294.0 | 398.0 | **388.1** |

I ran the same 10-run measurement a second time, after the cold start
(03:45 UTC, a fresh instance):

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Server `latency_ms` | 200.0 | 141.4 | 198.5 | 131.0 | 228.1 | 209.5 | 161.0 | 133.4 | 147.1 | 167.5 | **171.8** |
| Round trip, ms | 515.3 | 296.5 | 512.4 | 285.5 | 389.0 | 404.1 | 308.9 | 299.0 | 298.2 | 299.5 | **360.8** |

**One warm run costs about 0.16–0.17 s of server compute** (means 161.4 and
171.8 ms across two sets of 10; range 128–245 ms), and about 0.36–0.39 s end
to end for whoever is calling it. That's
about 3× slower than the same code on my laptop (48.9 ms mean, same script
against a local server), which matches Render's free tier giving a fraction
of one CPU.

Other costs the log shows:

| Cost | Measured | Source |
|---|---|---|
| Startup (load the ONNX model) | 5.3 s | `model_load_ms: 5309.6` in the `startup` record |
| Cold start (first request after the instance sleeps) | 32.6 s round trip | `measure_cost.py` warm-up at 03:45:19 UTC, after 17 idle minutes. Of that, 8.3 s was loading the model (`model_load_ms: 8294.5`) and 0.22 s was the post. The rest is Render waking the instance and starting Python |
| A post over 128 tokens | 758 ms | `latency_ms` of the truncated post above. Over 4× a short post: the model reads the full 128 tokens instead of 39, and the tokenizer still processes all 222 |
| A full 50-post batch of long posts | 42.9 s (~860 ms per post) | `request` record `66b8f40d`, `latency_ms: 42941.2` |
| A rejected post | 0.0 ms | rejected before the model runs |
| Memory | 439–471 MiB peak of 512, depending on the boot | `peak_rss_mb` in the `startup` and `request` records |

**Run Log — Before**

Three trials against https://takemeter.onrender.com on 2026-11-16 (UTC), each
on a **separate boot** and at least 5 minutes apart. They were run with
`python run_criteria.py https://takemeter.onrender.com --label before --trial
N`, which saves every response to `results_capstone_before_trialN.json` and
the server's log to `logs/capstone_before_trialN_server.jsonl`. Criteria are
in `criteria.md` (committed in `d4ebe51`, before any of these runs).

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Bad input gets no label and leaves a record | 15/15 handled and 15/15 logged | 15/15 · 15/15 | 15/15 · 15/15 | 15/15 · 15/15 | **MET** |
| 2. Hosted answers match the trained model | 229/229 labels and 229/229 `truncated` | 229/229 · 229/229 | 200/229 · 200/229 | 150/229 · 150/229 | **MISSED** |
| 3. Warm single post | median server ≤ 250 ms and every round trip ≤ 1,000 ms | 162.2 ms · 605.5 ms | 239.5 ms · 514.7 ms | 148.4 ms · 419.6 ms | **MET** |
| 4. 50-post batch of long posts | server `latency_ms` ≤ 30,000 ms | 43,651.5 ms | 44,114.2 ms | 43,763.6 ms | **MISSED** |
| 5. Memory headroom | `peak_rss_mib` ≤ 480 at startup and after the batch | 466.2 · 473.2 | 437.7 · 440.6 | 442.4 · 470.0 | **MET** |

What the rows don't show on their own:

- **Run 2 and run 3 each restarted mid-trial.** No one touched the
  dashboard. C2's last batches got a `502` from Render's proxy, and the
  service was down 36.9 s (run 2) and 32.0 s (run 3) before a new `startup`
  record. Render's Events tab, for run 2's restart:

  ```
  Instance failed: 8s9zn
  HTTP health check failed (timed out after 5 seconds) while running your code.
  November 15, 2026 at 11:37 PM      (= 04:37 UTC)
  Service recovered
  Recently failed instances are now reporting healthy.
  November 15, 2026 at 11:37 PM
  ```
- **Every post that got an answer matched.** In all three runs, 0 hosted
  labels and 0 `truncated` flags differed from the local PyTorch model. C2's
  misses are the posts in batches that got a `502`, not wrong answers.
- **The script waits out a crash before C3 and C4** (and records how long it
  waited), so those two still measure latency and batch time, not "the
  service is down". Run 2's C3 started 20 s after the restart. That
  probably explains its higher median (239.5 ms, just under 250).
- **Run 3's two memory numbers come from two boots.** 442.4 MiB is the
  `startup` of the boot the trial started on, and 470.0 MiB is after the
  batch, on the boot that came up after the crash. Both are under 480.
- **Four earlier attempts are kept but don't count.** Two were interrupted by
  me restarting the service mid-run (`*_INTERRUPTED*.json`). One ran on the
  first version of `run_criteria.py`, which kept no log once the instance
  restarted and let one crash fail every later criterion
  (`*_HARNESSv1.json`). I changed the script, not the system, so it
  snapshots `/logs` after every step and waits out an outage. Then I re-ran
  run 1. The memory and latency numbers in those files are consistent with
  the three runs above.

**Real output, run 2** (`results_capstone_before_trial2.json` and
`logs/capstone_before_trial2_server.jsonl`)

*Criterion 1.* `run_criteria.py`, `check_bad_inputs()`: the response to each
of the 15 cases, with its HTTP status and request id.

```
1  body not JSON              400 176bf305 {"error": "body is not JSON. Send {\"text\": \"...\"} or {\"posts\": [...]}"}
2  neither text nor posts     400 6ce73a35 {"error": "expected {\"text\": \"...\"} or {\"posts\": [\"...\", ...]}"}
3  posts not a list           400 9102d340 {"error": "\"posts\" must be a list"}
4  posts an empty list        400 2519b50c {"error": "\"posts\" is empty"}
5  51 posts                   413 25e3aaf0 {"error": "51 posts; the limit is 50 per request"}
6  body over 1 MB             413 c51b4c3f {"error": "request body over 1000000 bytes"}
7  empty string               200 4ee17a35 {"status": "rejected", "label": null, "reason": "empty"}
8  whitespace only            200 180b4e90 {"status": "rejected", "label": null, "reason": "empty"}
9  emoji only                 200 4b503c47 {"status": "rejected", "label": null, "reason": "no_words"}
10 link only                  200 1193b43c {"status": "rejected", "label": null, "reason": "link_only"}
11 bare score                 200 17b5ab28 {"status": "rejected", "label": null, "reason": "no_words"}
12 10,001 chars of words      200 163129bc {"status": "rejected", "label": null, "reason": "too_long (10001 chars > 10000)"}
13 a number                   200 af9d58e5 {"status": "rejected", "label": null, "reason": "not_text (int)"}
14 null                       200 f1031afe {"status": "rejected", "label": null, "reason": "missing"}
15 zero-width spaces only     200 2c88f7c8 {"status": "rejected", "label": null, "reason": "no_words"}
```

The server's records for cases 6, 9 and 15, written by `server.py`
(`classify_endpoint()` → `ServerLog.write()`):

```jsonl
{"ts": "2026-11-16T04:35:00.307+00:00", "run_id": "c51b4c3f", "event": "request", "path": "/classify", "status_code": 413, "latency_ms": 195.1, "n_posts": 0, "error": "request body over 1000000 bytes", "rss_mib": 367.0, "peak_rss_mib": 437.7}
{"ts": "2026-11-16T04:35:00.914+00:00", "run_id": "4b503c47", "event": "post", "source": "http", "row": 1, "latency_ms": 0.6, "input": "🔥🔥🔥", "status": "rejected", "label": null, "confidence": null, "tokens": null, "truncated": null, "reason": "no_words", "input_chars": 3}
{"ts": "2026-11-16T04:35:00.914+00:00", "run_id": "4b503c47", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 1.0, "n_posts": 1, "error": null, "rss_mib": 367.0, "peak_rss_mib": 437.7}
{"ts": "2026-11-16T04:35:01.852+00:00", "run_id": "2c88f7c8", "event": "post", "source": "http", "row": 1, "latency_ms": 0.0, "input": "​​", "status": "rejected", "label": null, "confidence": null, "tokens": null, "truncated": null, "reason": "no_words", "input_chars": 2}
{"ts": "2026-11-16T04:35:01.852+00:00", "run_id": "2c88f7c8", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 0.3, "n_posts": 1, "error": null, "rss_mib": 367.0, "peak_rss_mib": 437.7}
```

(The emoji came through intact this time, because the script sends it as
UTF-8 from Python. The zero-width spaces are shown escaped. In the file they
are the raw characters.)

*Criterion 2.* `run_criteria.py`, `check_same_answers()`, then the server's
records around the crash, by `server.py` `_finish()` and the module-level
startup code.

```
C2 same answers labels 200/229  truncated 200/229  (8 over 128)
batch_failures: [(200, 502)]        mismatches among answered posts: 0
   outage before C3: waited 36.9s, uptime now 19.9s, 1 new startup record(s)
```

```jsonl
{"ts": "2026-11-16T04:31:44.321+00:00", "run_id": null, "event": "startup", "model": "takemeter_onnx", "backend": "onnx", "runtime": "onnxruntime 1.32.0", "labels": ["analysis", "hot_take", "reaction"], "model_load_ms": 7791.7, "rss_mib": 363.4, "peak_rss_mib": 437.7, "python": "3.13.1"}
{"ts": "2026-11-16T04:35:39.817+00:00", "run_id": "850efcd1", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 16600.5, "n_posts": 50, "error": null, "rss_mib": 371.6, "peak_rss_mib": 437.7}
{"ts": "2026-11-16T04:35:55.419+00:00", "run_id": "e9b42c54", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 15279.3, "n_posts": 50, "error": null, "rss_mib": 373.3, "peak_rss_mib": 437.7}
{"ts": "2026-11-16T04:36:09.517+00:00", "run_id": "02f3e8f3", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 13479.3, "n_posts": 50, "error": null, "rss_mib": 373.8, "peak_rss_mib": 437.7}
{"ts": "2026-11-16T04:36:52.373+00:00", "run_id": null, "event": "startup", "model": "takemeter_onnx", "backend": "onnx", "runtime": "onnxruntime 1.32.0", "labels": ["analysis", "hot_take", "reaction"], "model_load_ms": 7306.8, "rss_mib": 365.6, "peak_rss_mib": 440.6, "python": "3.13.1"}
```

The 4th batch (posts 150–199) returned `200` to the script, but its
`request` record isn't in the saved log. The instance failed before the next
`/logs` snapshot, and its disk went with it. So a request can succeed and
still leave no record anywhere I kept. It would only be in Render's Logs tab.

*Criterion 3.* `run_criteria.py`, `check_warm_latency()`: 10 runs after one
untimed warm-up.

```
{"http_status": 200, "request_id": "729492d3", "server_ms": 239.4, "round_trip_ms": 393.6}
{"http_status": 200, "request_id": "0590f8de", "server_ms": 239.6, "round_trip_ms": 409.1}
{"http_status": 200, "request_id": "a62ba700", "server_ms": 244.4, "round_trip_ms": 391.8}
{"http_status": 200, "request_id": "6be0b3c1", "server_ms": 259.7, "round_trip_ms": 412.5}
{"http_status": 200, "request_id": "6409b2d4", "server_ms": 240.1, "round_trip_ms": 392.9}
{"http_status": 200, "request_id": "cd276995", "server_ms": 229.8, "round_trip_ms": 396.5}
{"http_status": 200, "request_id": "0945d616", "server_ms": 252.7, "round_trip_ms": 397.0}
{"http_status": 200, "request_id": "2a23c3ee", "server_ms": 192.2, "round_trip_ms": 514.7}
{"http_status": 200, "request_id": "23ab5ff4", "server_ms": 147.9, "round_trip_ms": 298.2}
{"http_status": 200, "request_id": "ef27f1e5", "server_ms": 133.8, "round_trip_ms": 295.5}
median server 239.5 ms · max round trip 514.7 ms
```

*Criterion 4.* `run_criteria.py`, `check_big_batch()`, and the server's
`request` record for that call (`server.py` `_finish()`).

```
{"http_status": 200, "labelled": 50, "request_id": "e8fd0890", "server_ms": 44114.2, "round_trip_ms": 44379.5, "all_truncated": true}
```
```jsonl
{"ts": "2026-11-16T04:37:54.072+00:00", "run_id": "e8fd0890", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 44114.2, "n_posts": 50, "error": null, "rss_mib": 379.7, "peak_rss_mib": 440.6}
```

*Criterion 5.* `run_criteria.py`, `check_memory()`, reading
`peak_rss_mib` from the `startup` record above (437.7) and from the C4
`request` record (440.6). Both come from `server.py` `memory_mib()`
(`ru_maxrss` and `/proc/self/statm`).

```
C5 memory       startup peak 437.7 MiB  after batch 440.6 MiB
```

**Verdicts and Diagnosis**

| # | Criterion | Target | Before | Verdict |
|---|---|---|---|---|
| 1 | Bad input gets no label and leaves a record | 15/15 and 15/15, every run | 15/15 · 15/15 on all 3 | **MET** |
| 2 | Hosted answers match the trained model | 229/229 and 229/229, every run | 229 · 200 · 150 | **MISSED** |
| 3 | Warm single post | median ≤ 250 ms and every round trip ≤ 1,000 ms, every run | 162.2 / 239.5 / 148.4 ms; max 605.5 ms | **MET** |
| 4 | 50-post batch of long posts | ≤ 30,000 ms, every run | 43,651.5 / 44,114.2 / 43,763.6 ms | **MISSED** |
| 5 | Memory headroom | ≤ 480 MiB, every boot | highest 473.2 MiB | **MET** |

*Miss 2: answers match the trained model (229, 200, 150 of 229).*
The stage is **the request handler, `classify_endpoint()` in `server.py`**.
It's an `async def`, but it ran the model loop (`classify.run_post()` for each
post) directly on the server's one event loop. A 50-post batch held that loop
for 13–17 s, and during that time the server couldn't answer **any** other
request, including the `GET /health` that Render polls. Render gives
`/health` 5 seconds. Once it doesn't answer in time, Render marks the instance
failed and restarts it: *"HTTP health check failed (timed out after 5
seconds) while running your code"* (Events, 04:37 UTC, run 2). The batch in
flight, or the next one, gets Render's `502`, and the restart costs 32–37 s.
Evidence that this is the mechanism and not something else:
- **Not memory.** Just before run 2's crash, `rss_mib` was 371.6–373.8,
  140 MiB under the limit. Run 1 survived the same batches at 473 MiB.
- **Not the ONNX swap or the tokenizer.** Across the three runs, 579 posts
  got an answer, and 0 of them differed from the PyTorch model in label or
  `truncated`.
- **The block is real.** Locally, `/health` took 7.7 s to answer during an
  8.3 s batch. On Render, my own `/health` probe got a `502` in the middle of
  run 1's 43 s C4 batch, while the instance stayed up.
- **Why only sometimes.** C2 sends five batches back to back, so the loop is
  blocked for about 65 s with gaps of under a second. Whether Render restarts
  the instance depends on how many of its checks land inside that stretch.
  Run 1 got lucky, and runs 2 and 3 didn't.

The criterion was written to catch a *correctness* problem (ONNX vs
PyTorch). It caught an *availability* problem instead. That's not a broken
criterion: "229/229 answered and matching" is still exactly what I need from
the URL, and it measured cleanly. The correctness half held: 0 mismatches in
579 answers.

*Miss 4: 50-post batch (43.7–44.1 s against 30 s).*
The stage is **inference, the ONNX forward pass inside `classify.run_post()`
→ `predict()` in `classify.py`**, which runs **one post at a time, in
series**. The mechanism is that its cost grows with **tokens per post**, and
on Render's shared CPU that growth is steep. The 39-token post in criterion
3 costs ~160 ms. Each 222-token post is cut to 128 tokens, but still costs
~875 ms (43,763.6 ms / 50). 50 of them add up to 44 s. To hit 30 s, a full
128-token post would have to cost ≤ 600 ms. My target assumed a long post
cost about what my laptop showed in Milestone 3, scaled up. In fact Render
is ~6× slower than my laptop on long posts (7.1 s for the same batch
locally), against ~3× on short ones.

*The pattern: one problem, two symptoms.* Both misses come from the same
place: **the model work sits inside the request handler, runs post by post,
and runs on the event loop**. Miss 4 is that work's cost (44 s for the
largest batch I allow). Miss 2 is where it's paid: on the only thread that
answers `/health`, so a slow batch turns into a dead service. The 32 MiB
memory worry I wrote criterion 5 about never materialised: 473.2 MiB at
worst.

*Met, and were they set low?* Criterion 1 (15/15) is all-or-nothing and
can't really be tighter. Criterion 3 passed, but run 2's median was 239.5 ms
against 250, so it isn't a soft target. Criterion 5 passed with 7 MiB to
spare on run 1. None were soft, so I'm not tightening any.

No criterion is revised. All five measured what they said.

**The Improvement**

*What I changed:* one thing in `server.py`. The model loop moved out of
`classify_endpoint()` into a plain function, `_classify_posts()`, and the
handler now runs it in a worker thread:
`results = await run_in_threadpool(_classify_posts, posts, request_id)`.
Validation, logging, the model, the limits and every response field are
unchanged. onnxruntime releases Python's GIL during the forward pass, so the
event loop stays free to answer `/health` while a batch runs.

*Which diagnosis it follows:* miss 2's. The event loop was blocked, so
`/health` timed out and Render restarted the instance. I picked it over
miss 4's cost because a restart takes the URL down **for everyone** for
30+ seconds and loses in-flight work and logs. A slow batch only makes the
person who sent it wait.

*Local check before deploying* (the same test that showed the block, 50
posts of 222 tokens):

| During the batch | Before the fix | After the fix |
|---|---|---|
| `GET /health` | 7.7 s | 0.003–0.005 s (5 probes) |
| A single post sent mid-batch | waits for the batch | 0.016 s |
| The batch itself | 8.3 s | 7.1 s, 50/50 labelled, all `truncated: true` |

*What I expect, written before the after-runs:* criterion 2 goes to 229/229
on all three runs, because the instance stops getting restarted. **Criterion
4 stays missed.** The fix moves the work but doesn't make it smaller, so the
batch should still take ~44 s. Criteria 1, 3 and 5 should be unchanged.
There's a risk on Render's fraction of one CPU: the health thread and the
model thread now compete for CPU, so batch times could get slightly worse.

**Run Log — After**

The same five criteria, the same script (`run_criteria.py --label after`),
three runs on three separate boots, after deploying the fix (`423182a`).
Results are in `results_capstone_after_trialN.json`, and server logs in
`logs/capstone_after_trialN_server.jsonl`.

| Criterion | Target | Run 1 | Run 2 | Run 3 | Verdict |
|---|---|---|---|---|---|
| 1. Bad input gets no label and leaves a record | 15/15 handled and 15/15 logged | 15/15 · 15/15 | 15/15 · 15/15 | 15/15 · 15/15 | **MET** |
| 2. Hosted answers match the trained model | 229/229 labels and 229/229 `truncated` | 229/229 · 229/229 | 229/229 · 229/229 | 229/229 · 229/229 | **MET** |
| 3. Warm single post | median server ≤ 250 ms and every round trip ≤ 1,000 ms | 173.4 ms · 397.7 ms | **266.4 ms** · 504.7 ms | 166.3 ms · 532.1 ms | **MISSED** |
| 4. 50-post batch of long posts | server `latency_ms` ≤ 30,000 ms | 43,731.1 ms | 44,479.5 ms | 45,422.3 ms | **MISSED** |
| 5. Memory headroom | `peak_rss_mib` ≤ 480 at startup and after the batch | 444.5 · 444.5 | 444.6 · 444.6 | 466.8 · **501.2** | **MISSED** |

Deviations, stated plainly:
- **Run 3 started 1 min 56 s after run 2 ended**, not the ≥ 5 minutes
  `criteria.md` sets. I restarted the service early. It was still a separate,
  fresh boot (33 s uptime), so no load carried over from run 2. But it breaks
  my own protocol, so I'm noting it, not hiding it. I didn't re-run it,
  because replacing a run with a bad result is exactly what the protocol is
  there to stop.
- **Run 1's boot had one extra 50-post batch** before the trial: my smoke
  test that `/health` answers mid-batch after the deploy (it did, in
  0.195 s). That batch is in run 1's log at 05:03:14. It can only have
  raised run 1's memory, and run 1's memory was the lowest of the three.

**Before and after, side by side**

| Criterion | Target | Before (runs 1 · 2 · 3) | Before | After (runs 1 · 2 · 3) | After |
|---|---|---|---|---|---|
| 1. Bad input | 15/15 · 15/15 | 15·15 / 15·15 / 15·15 | MET | 15·15 / 15·15 / 15·15 | MET |
| 2. Same answers | 229/229 | 229 / 200 / 150 | MISSED | 229 / 229 / 229 | **MET** |
| 3. Warm latency | median ≤ 250 ms | 162.2 / 239.5 / 148.4 ms | MET | 173.4 / 266.4 / 166.3 ms | **MISSED** |
| 4. 50-post batch | ≤ 30,000 ms | 43,651.5 / 44,114.2 / 43,763.6 | MISSED | 43,731.1 / 44,479.5 / 45,422.3 | MISSED |
| 5. Memory | ≤ 480 MiB | 473.2 / 440.6 / 470.0 (max) | MET | 444.5 / 444.6 / 501.2 (max) | **MISSED** |
| Restarts mid-trial | — | 0 / 1 / 1 | | 0 / 0 / 0 | |

**Real output, after.** Run 3's server log (`server.py`, `_finish()`): the
five C2 batches that used to kill the instance, then the C4 batch, with no
new `startup` in between:

```jsonl
{"ts": "2026-11-16T05:22:30.686+00:00", "run_id": null, "event": "startup", "model": "takemeter_onnx", "backend": "onnx", "runtime": "onnxruntime 1.32.0", "labels": ["analysis", "hot_take", "reaction"], "model_load_ms": 9301.1, "rss_mib": 467.6, "peak_rss_mib": 466.8, "python": "3.13.1"}
{"ts": "2026-11-16T05:23:36.688+00:00", "run_id": "4bcf1d97", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 16604.8, "n_posts": 50, "error": null, "rss_mib": 482.7, "peak_rss_mib": 499.4}
{"ts": "2026-11-16T05:23:52.883+00:00", "run_id": "9b5e2423", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 15894.9, "n_posts": 50, "error": null, "rss_mib": 498.2, "peak_rss_mib": 499.8}
{"ts": "2026-11-16T05:24:06.885+00:00", "run_id": "ef1721f9", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 13688.0, "n_posts": 50, "error": null, "rss_mib": 488.0, "peak_rss_mib": 499.8}
{"ts": "2026-11-16T05:24:24.783+00:00", "run_id": "19fac8b9", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 17575.6, "n_posts": 50, "error": null, "rss_mib": 495.3, "peak_rss_mib": 500.4}
{"ts": "2026-11-16T05:24:33.778+00:00", "run_id": "61bb6151", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 8644.3, "n_posts": 29, "error": null, "rss_mib": 495.3, "peak_rss_mib": 500.4}
{"ts": "2026-11-16T05:25:23.887+00:00", "run_id": "30b74b2c", "event": "request", "path": "/classify", "status_code": 200, "latency_ms": 45422.3, "n_posts": 50, "error": null, "rss_mib": 496.4, "peak_rss_mib": 501.2}
```

`run_criteria.py` console output for the three runs (`main()`):

```
trial 1 (after)  C2 same answers labels 229/229  truncated 229/229  (8 over 128)
trial 2 (after)  C2 same answers labels 229/229  truncated 229/229  (8 over 128)
trial 3 (after)  C2 same answers labels 229/229  truncated 229/229  (8 over 128)
trial 2 (after)  C3 warm latency median server 266.4 ms  max round trip 504.7 ms
trial 3 (after)  C5 memory       startup peak 466.8 MiB  after batch 501.2 MiB
```

**Did It Help**

**It fixed what it was aimed at, and it cost memory I didn't have to
spare.**

- **Criterion 2: fixed.** 229/229 on all three runs, against 229 / 200 / 150
  before, with **zero restarts in three runs** against two in three before.
  The same five back-to-back batches that triggered Render's 5-second
  health-check failure now go through, because `/health` answers in under
  0.2 s during a batch instead of not at all. That was the prediction, and it
  held.
- **Criterion 4: unchanged, as predicted.** 43.7–45.4 s against 43.7–44.1 s.
  The fix moved the work, it didn't make it smaller. Run 3's 45.4 s is the
  slowest batch I've recorded, which fits the risk I wrote down: the health
  thread and the model thread now share a fraction of one CPU.
- **Criterion 5: broken by the fix. This is the trade.** On the two
  low-memory boots, settled memory went from ~371–374 MiB before to
  ~396–401 MiB after. On a high-memory boot (startup ~467 MiB in both cases),
  the first 50-post batch added **5 MiB before the fix** (before run 1:
  471.1 peak) and **33 MiB after** (after run 3: 499.4 peak), ending at
  **501.2 MiB, 11 MiB from Render's kill line**. Running the model in a
  worker thread costs about 25–33 MiB. My best explanation is that glibc's
  allocator gives each new thread its own memory pool, so the worker's
  allocations don't reuse what the main thread freed. That's an inference I
  haven't verified on Render. What I *have* measured is the size of the jump
  and that the fix is the only difference between the two runs. So the fix
  turned the crash I could see (health-check timeout) into one I'm closer to
  (out of memory on an unlucky boot).
- **Criterion 3: missed on one run, and I don't think the fix caused it.**
  Run 2's ten posts were all slow (218.8–303.0 ms, median 266.4). But
  `latency_ms` is timed inside the worker thread around the model call
  alone, so the thread hand-off isn't in it, and runs 1 and 3 were normal
  (173.4 and 166.3). Before the fix, run 2 had already come within 10.5 ms of
  the target (239.5). I read it as Render's shared CPU, and the target
  had too little margin for that noise. But the rule is that it has to hold
  on all three runs, and it didn't, so it's **MISSED**.

**Net:** three MET before (1, 3, 5) and two MET after (1, 2). By count it
went down. By what matters, I'd still keep the fix: a
service that restarts itself under the load my API advertises is worse than
one that's 11 MiB from a limit. But criterion 5 is now the most urgent
problem, and the fix created it. The next change would be `MALLOC_ARENA_MAX=2`
in `render.yaml`, which caps glibc's per-thread pools, measured the same
way. That's a second improvement, and this milestone allows one.
