# Knotwork: certified visual reasoning data

*A one-page summary. Technical detail, measurements and limits are in [`README.md`](README.md).*

## The problem

Vision-language models are being evaluated for physical, visual work: tracing a patch cable across a
server rack, reading an analog gauge, following a harness from a socket to an instrument. The data
used to test and train these skills has three weaknesses.

- **Labels can be silently wrong.** When the program that draws an image also writes its answer, any
  drawing bug becomes a wrong label, and nobody checks.
- **Scores hide shortcuts.** A model can trace cables by matching colours, or answer a question about
  a dial from the wording alone, and still look competent.
- **Test results do not become training signal.** A benchmark score says *that* a model failed, not
  *where*, and it produces nothing a training run can use.

## What Knotwork does

Knotwork generates fresh visual tasks on demand and attaches a proof to each one.

1. **Every answer is checked against the pixels.** A small, isolated verification kernel recomputes
   each answer from the rendered image. It does not use the generator's own record of the scene. A
   task is released only when both answers agree. In testing, the kernel caught all 56 deliberately
   injected rendering bugs, and it caught one real one during development.
2. **Shortcuts are measured, not assumed away.** Each task can come with controlled variants: the
   same scene recoloured, decluttered, made one step easier, or written out as text. Comparing a
   model's answers across them shows whether it saw the scene or guessed.
3. **Failures become training data.** Every certified task yields a step-by-step reasoning trace in
   which each step points at the exact image region it reads. Failed answers become preference pairs.
   A built-in scoring function rewards correct, grounded answers and flags correct answers given
   without evidence.

## What it has shown

In a live audit of three current models (1,069 questions, $1.12 in total API cost):

- Overall accuracy was 20–25 %.
- On identical scenes, one model traced 59 % of cables correctly when each cable had its own colour
  and 3 % when all cables were the same colour. The colour, not the path, was doing the work.
- In a matched test of true and false claims about a dial, no model accepted true claims more often
  than false ones by a detectable margin.

## How it is used

- **Evaluation teams** run multi-model campaigns with confidence intervals, repeat-consistency
  scores and full audit exports, offline or against live models with a spend cap.
- **Training teams** export a ready post-training pack: reasoning traces for supervised fine-tuning,
  preference pairs, reinforcement-learning data and the matching scoring function. Training and
  evaluation items never share a seed.
- **Release engineering** pins any failure into a regression suite and blocks a model release when
  accuracy on it drops (`knot gate`).
- **New domains** plug in as separate packages and are verified by the same kernel, without changes
  to it.

## Cost and deployment

The system runs on a laptop with no API keys and no network access. Generating and certifying one
task takes a few seconds on one CPU core. That works out to roughly $48–$74 per million certified
tasks at spot-cloud CPU prices. The same container image runs air-gapped on-premises or as a capped
public sandbox.

## Current limits

Scenes are flat 2-D renders. Real camera perspective, lighting and clutter are planned but not
built. One statistical shortcut check still fails by a small margin and is recorded as an open item.
The live audit covered three of the four domains. See [`BLOCKERS.md`](BLOCKERS.md).

## Next steps

Camera-geometry rendering, multi-turn interactive inspection tasks, before/after change detection,
and a validation study on photographs of real equipment.

---

Swarit Sharma and Pankaj David (Team Bilota) · Apache-2.0 · [github.com/swaritbkp/vsfactory](https://github.com/swaritbkp/vsfactory)
