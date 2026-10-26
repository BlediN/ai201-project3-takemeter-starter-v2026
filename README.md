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
| Overall accuracy | 0.314 |  |  |
| Macro F1 | 0.159 |  |  |
| F1 — `analysis` | 0.478 |  |  |
| F1 — `hot_take` | 0.000 |  |  |
| F1 — `reaction` | 0.000 |  |  |

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
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



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

**Agreement rate:** ___ / 30 = ___%

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

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



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
