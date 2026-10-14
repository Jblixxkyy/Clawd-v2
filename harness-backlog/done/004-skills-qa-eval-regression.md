# 004 — Skills Q&A eval regression test

What: add the Oct 11 skills-Q&A scenario to the eval harness (`python3 -m src.eval run`): ask "what skills do you have?" and grade whether the model answers from the injected `## Available Skills` inventory instead of inventing generic skills from priors. The 30B model failed this 3x (verified the inventory bytes WERE injected — model limitation, not a registration bug). This is a real model-behavior regression test for prompt changes.

Tests: new eval task passes on a model that reads the inventory; documents the known 30B failure mode.
