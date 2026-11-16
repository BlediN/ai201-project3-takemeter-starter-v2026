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

---
---

# Capstone criteria: hardened, hosted TakeMeter

Written on 2026-11-16, before any capstone test run. The five criteria above
are project 3's and are unchanged. These five are about the extension:
`classify.py`'s failure handling, the logging, and the hosted service at
https://takemeter.onrender.com (`server.py`, ONNX backend).

**What I already knew when I wrote these.** Milestone 3 measured a few
things while getting the service running: warm latency (means 161 and
172 ms over two sets of 10), one 50-post batch of long posts (42.9 s), peak
memory on two boots (471 and 439 MiB), and one cold start (32.6 s). Those
numbers set where the targets sit. None of the five tests below has been run
yet.

**What counts as a trial.** Each criterion runs 3 times. For 1–4 a trial is a
separate run of the whole test against the hosted URL, with the instance
already awake (a warm-up request first), and at least 5 minutes between
trials. For 5, each trial is a **separate boot** of the instance, because
memory changed between boots in Milestone 3. A criterion is **met** only if
it holds on all 3 trials.

---

## 1. Bad input never gets a label, and always leaves a record

Send this fixed set of 15 bad inputs to the hosted `/classify`. In every
trial, all 15 must:
- come back without a label,
- produce no `5xx` and no timeout, and
- leave a log record in `/logs` (a `request` record with the same
  `request_id`, plus a `post` record for each post-level case).

Request-level cases (expect `400` or `413`):
1. body not JSON
2. JSON with neither `text` nor `posts`
3. `posts` not a list
4. `posts` an empty list
5. 51 posts
6. a body over 1 MB

Post-level cases (expect `200` with `status: "rejected"` and `label: null`):

7. `""`
8. whitespace only (`"   \n\t"`)
9. emoji only (`"🔥🔥🔥"`, sent as UTF-8 from a file, not typed into the
   Windows shell, which mangled it in Milestone 3)
10. a link only
11. a bare score (`"23-9"`)
12. 10,001 characters of real words
13. a number (`42`)
14. `null`
15. zero-width spaces only (`"\u200b\u200b"`)

**Target: 15/15 handled correctly, and 15/15 logged, on every trial.**

**Why this target:** This is the bad path, and it's ordinary code, not the
model, so it should be all-or-nothing. A garbage post that comes back as
`hot_take` at 0.40 confidence looks the same as a real prediction to whoever
reads it. Given the model's 0.60 accuracy, that's one more silent wrong
label on top of the ones I can't avoid. One forced label, one `500`, or one
failure that leaves no record is a bug I can name, so the target is 100%.
Anything looser would excuse exactly the failures shape B is meant to remove.
Case 15 is in because Python's `strip()` doesn't treat a zero-width space as
whitespace, so I don't know which check catches it. Case 6 is in because
Render's proxy sits in front of my 1 MB limit, and I haven't seen what it does
with a large body.

---

## 2. The hosted model gives the same answers as the trained model

Send all 229 posts in `labels.csv` to the hosted `/classify` in 5 batches (at
most 50 each). Compare each result with the local PyTorch model in
`takemeter_model/` (the notebook's model) and with an independent token count
from the `transformers` tokenizer.

**Target, every trial: the same label on 229/229 posts, and the `truncated`
flag correct on 229/229.** "Correct" means `true` exactly when the
independent count is over 128. 8 of the 229 posts are over 128 tokens.

**Why this target:** The hosted service runs a different runtime (ONNX, not
PyTorch), a different tokenizer library (`tokenizers`, not `transformers`)
and my own hand-written truncation step. Each of those could change an answer
without any error. Locally the two agreed on 37/37 posts, but that's 37 posts,
on Windows. If even one label differs, the 0.600 accuracy in my project 3
README isn't the accuracy of the thing at the URL. So the target is exact,
not "close". The `truncated` flag is the reason long posts are no longer
silently cut, and a flag that's wrong some of the time is worse than none. 229
posts cover all 8 of my over-limit posts. A test-split-only check would cover
just 1–2 of them.

---

## 3. A warm single post comes back in under a second

Send the same single post (the 39-token Jokic post from `measure_cost.py`) 10
times to the awake instance.

**Target, every trial: median server `latency_ms` ≤ 250 ms, and every one of
the 10 round trips ≤ 1,000 ms.**

**Why this target:** The page at `/` is someone pasting one comment and
waiting. Under about a second it feels instant, and past that people click
again, which doubles the load on a fraction of one CPU. 250 ms on the server
leaves room for what I saw in Milestone 3 (medians of 153 and 164 ms, single
requests up to 245 ms) without accepting a real slowdown. A median of 200 ms
would make me fail on noise from Render's shared CPU, and 500 ms would hide a
real regression. I check every round trip against 1,000 ms, not the mean,
because the slowest request is what a person notices. Cold starts are left out
on purpose (~30 s, already known). This is about warm requests only.

---

## 4. The largest batch the API allows finishes in 30 seconds

Send one request with 50 posts (the API's limit), each one over 128 tokens
(the same 222-token post, so every one fills the model's full input).

**Target, every trial: status `200`, all 50 labelled, and the server's
`latency_ms` for the request ≤ 30,000 ms.**

**Why this target:** The API says 50 posts per request, so a 50-post batch of
long posts is a request I've promised to handle. In Milestone 3 one took
**42.9 s**, about 860 ms per post. 30 s is the default timeout in a lot of
HTTP clients and scripts, so a request slower than that gets abandoned
partway, and whoever sent it has no labels and doesn't know why. A looser
target would describe what the system already does, not what a caller needs.
A stricter one (10 s, say) would mean rethinking the free tier, not fixing
`server.py`. I expect this one to miss before the fix.

---

## 5. Memory stays at least 32 MiB under Render's 512 MiB limit

On each of 3 separate boots, read `peak_rss_mib` from the `startup` record,
then send the 50-long-post batch from criterion 4 and read `peak_rss_mib`
from its `request` record.

**Target, every boot: `peak_rss_mib` ≤ 480 MiB, both at startup and after the
batch.**

**Why this target:** Render kills the service if it goes over 512 MiB, and on
the free tier that means the URL is down until the next deploy or wake-up. I
measured 471 MiB on one boot and 439 MiB on another. The 32 MiB swing between
boots is about the same size as the headroom. 480 MiB keeps one boot's swing
(32 MiB) between me and the limit. A boot 32 MiB worse than my 471 MiB one
would sit at 503 MiB, 9 MiB from being killed. 500 MiB would leave 12 MiB, less than the swing I've already
seen, so it would pass right up to the boot that crashes. 450 would only
test whether the first boot was a fluke. I expect this one to miss on some
boots.
