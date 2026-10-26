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
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

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

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



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
