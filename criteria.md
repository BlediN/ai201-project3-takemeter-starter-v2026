# Acceptance criteria — TakeMeter

Five criteria that say what "working" means for this classifier, written in
unit 5 **before** anything was trained.

**All five are yours this time.** None are given. You've had two projects of
practice.

An acceptance criterion names a number. *"The model is accurate"* is an
opinion. *"Every label has an F1 of at least 0.60 on the held-out set"* is a
criterion.

Under each, write a sentence or two on **why that number**. A reason that says
something about your data or your taxonomy earns credit — *"I picked 0.60 F1
for `reaction` because it's my smallest label and I only have about 50
examples of it"*. A reason that could be attached to any project does not.

> Missing your own targets next unit costs you nothing. Setting a target so
> easy you can't miss it does.

---

## Pick numbers you can defend

Not numbers that sound impressive. Three labels means a coin-flip guesser gets
about 33%, so a target of 0.40 is barely a target. Your number should sit
somewhere you'd honestly call useful.

**Cover at least three of these five areas.** They're here as prompts, not as a
form to fill in — a criterion that fits none of them is fine if it names a
number.

| Area | A question it could answer |
|---|---|
| Overall accuracy | How often does it need to be right to be worth using? |
| Per-label performance | Is one label allowed to be much worse than the others? |
| Balance | How lopsided can your label counts get before it's a problem? |
| Consistency | If someone else labelled the same posts, how often should you agree? |
| Confidence | Should a confident prediction be right more often than an unsure one? |

Two things worth knowing before you pick numbers, because both will affect
whether you hit them:

- **Your smallest label will have the jumpiest score.** If a label has 50
  examples, about 8 land in the test split. An F1 computed on 8 examples moves
  a lot between seeds. A target for that label should be looser than one for
  your biggest label, and saying so is a good reason.
- **Unit 6 tests across three seeds, and the target has to hold across all
  three.** A target of 0.65 against results of 0.71, 0.62, 0.68 is a **miss**.
  Pick with that in mind — it is stricter than it first sounds.

---

## 1. Overall accuracy

Overall accuracy is at least 0.65 on the held-out set, on every seed.

**Why this target:** With three labels, guessing gets about 0.33. Guessing
the biggest label every time gets about 0.40–0.45, because I expect `reaction`
to be the largest share of r/nba comments. 0.65 is clearly above both. The
test split is about 30 posts, so 0.65 means roughly 20 right. That's the
point where I'd trust it to sort a comment section for me.

---

## 2. Per-label performance

`reaction` and `hot_take` each have an F1 of at least 0.60 on the held-out
set, and `analysis` has at least 0.45, on every seed.

**Why this target:** `analysis` gets the looser target for two reasons. It
will be my smallest label (I expect about 40–50 examples, so 6–8 in the test
split), and its F1 will jump between seeds. Also, my second decision rule says
a live-game stat is still `reaction`, so the model can't just learn "has a
number = `analysis`". I expect it to confuse those two. `reaction` and
`hot_take` will each have more examples and clearer word cues ("this
quarter", "INSANE"), so they get 0.60.

---

## 3. Balance

No label is more than 50% of `labels.csv`, and none is less than 20%.

**Why this target:** The project's ceiling is 70%. That's too loose for
r/nba, where a single game thread can hand me 100 `reaction` comments in
minutes and drown out everything else. 20% of 200 is 40 posts, which puts
about 6 in the test split. Below that, a label's F1 is decided by one or two
posts and means nothing.

---

## 4. Consistency

My labels agree with the staff labels on at least 75% of the 30 staff posts,
as measured by `agreement.py`.

**Why this target:** 75% of 30 allows up to 7 disagreements. I'm not asking
for 90%: the staff definitions use the same three label names but draw
`reaction` differently (the writer's own experience, not live game action).
Some disagreement will come from carrying my rules over to theirs, not from
careless labeling.

---

## 5. Confidence

Accuracy on the model's most confident third of test posts is at least 0.20
higher than on its least confident third, on every seed.

**Why this target:** If confidence doesn't separate right from wrong, I
can't use it to flag posts for a human to check. Each third is only about 10
posts, so accuracy moves in steps of 0.10. A 0.20 gap means two more correct
posts in the confident third, which is the smallest gap I wouldn't write off
as noise. I expect the least confident third to be mostly
`analysis`/`hot_take` boundary posts, where the rule turns on whether a fact
is checkable.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     UNIT 6 — read this before you change anything above.

     If a criterion turns out to be BROKEN rather than merely unmet, you can
     revise it, and that earns credit. But never delete or edit the original
     line. Add the revision underneath, like this:

         ## 2. Every label performs acceptably

         The model performs well on all labels.

         **Why this target:** ...

         > **Revised in unit 6:** Every label has an F1 of at least 0.60 on
         > the held-out set.
         >
         > **Why revised:** "performs well" gave me nothing to check. I
         > couldn't produce a verdict from it at all.

     That's a revision because the criterion couldn't be MEASURED.

     Lowering a target because you missed it is not a revision, and it costs
     you the point:

         ✗ "Overall accuracy of at least 0.65" → "at least 0.55", because
            0.65 turned out to be optimistic for 200 examples.

     A number you missed stays where it is, gets diagnosed, and gets a fix
     attempted. That's where the points are.
     ───────────────────────────────────────────────────────────────────────── -->
