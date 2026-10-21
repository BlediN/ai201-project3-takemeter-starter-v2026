# Milestone 2 — Stress test and boundary check

## Stress test

Eight AI-written posts designed to sit on the label boundaries, classified
under the first draft of the definitions.

| # | Post | First draft | Clean? |
|---|---|---|---|
| 1 | Wemby is going to win DPOY unanimously. Nobody else is even close. | `hot_take` | yes |
| 2 | That's his third block this quarter. Wemby is a cheat code, this is unfair. | `reaction` or `analysis` | **no** |
| 3 | Brunson shot 41% from three last year and he's still the most underrated guard in the league, it's not even debatable. | `analysis` or `hot_take` | **no** |
| 4 | If the Lakers stay healthy they're a top-4 seed, Luka and LeBron together just makes too much sense. | `hot_take` | yes |
| 5 | Knicks are 0-12 from three in the second half and Thibs still won't change the lineup. Fire him. | `reaction` or `analysis` | **no** |
| 6 | The Athletic putting Denver 15th is the dumbest thing I've read all year. | `hot_take` | yes, but only because "live game" was in the definition |
| 7 | Down 6 with 40 seconds left and they call timeout instead of running. They've done this every close game since last March. | `reaction` or `analysis` | **no** |
| 8 | Jokic averaged a triple-double last season. Denver at 15 is ridiculous. | `analysis` | yes |

Four of eight were clean. The failures fell into two groups, plus one near
miss:

- **Live stats (2, 5, 7).** A number from the game being played is
  checkable, so the first `analysis` definition let these in. But they're
  reactions to what's on screen.
- **A fact that doesn't obviously support the claim (3).** "Does work in the
  argument" was a judgment call.
- **Near miss: reacting to news, not a game (6).** It sorted cleanly, but
  nothing said outright where news reactions go, so `hot_take` now names them.

## What changed

- `analysis` now requires a fact **from outside the current game**, given
  **as the reason** for the claim.
- `reaction` is limited to **the game being played right now**, and keeps the
  label even when it quotes a number from that game.
- `hot_take` explicitly includes **reactions to news** with no fact.
- Added a second decision rule for `reaction` vs `analysis`.

Under the new definitions: 1 `hot_take`, 2 `reaction`, 3 `analysis`,
4 `hot_take`, 5 `reaction`, 6 `hot_take`, 7 `reaction`, 8 `analysis`.

## Strongest case for each label on the borderline post

> We're basically 100% dependent on health. If AG and Braun stay healthy then
> our defense doesn't suffer and we're a top team. If not then we're probably
> in the 5-7 range in the West

**Case for `analysis`:** It names two specific players and a specific
mechanism (their health drives the defense). It gives a conditional
prediction with a concrete range ("5-7 in the West"). It's reasoned, not
just loud.

**Case for `hot_take`:** Nothing in it can be looked up. "Our defense doesn't
suffer" and "top team" are assertions, and the seed range is a guess. Take
away the confident framing and there's no evidence left, only an opinion
about how the season will go.

Both cases are strong, which is why this boundary needs the decision rule. The
rule sends it to `hot_take`: reasoning without a checkable fact isn't
`analysis`.
