# Remask policy development

This branch adds rollback as a separate operation from the upstream sampler.
The upstream `remasking` argument is retained for compatibility even though it
actually selects which masked positions to *unmask*.

## Stage 1: threshold heuristic

At each model evaluation, the sampler first scores every committed token using
the probability of that exact token under the current context. A token is
eligible for rollback when it:

- belongs to the active generation block;
- has survived at least `remask_min_age` later evaluations;
- has been remasked fewer than `max_remasks_per_token` times; and
- has current-token confidence below `remask_threshold`.

Unmask and remask decisions are applied simultaneously to disjoint candidate
sets. Therefore, a token changed back to MASK cannot be recommitted from the
same model forward pass. The sampler allows `remask_max_extra_steps` additional
evaluations per block and terminates after a complete sequence passes a settle
check without another rollback.

Run the matching offline smoke cases on the server:

```bash
bash scripts/smoke_fastdllm.sh
bash scripts/smoke_threshold_remask.sh
```

Both use the same unmask threshold (default `0.7`). The second additionally
uses a remask threshold (default `0.5`). Override them in `.env.local`:

```bash
UNMASK_THRESHOLD=0.7
REMASK_THRESHOLD=0.5
```

Each result prints generated text, NFE, and total remask actions. The first
meaningful benchmark will sweep remask thresholds on GSM8K while holding the
unmask rule, prompts, seeds, and generation limits fixed.

## Stage 2: learned policy

After the heuristic establishes that rollback works, its threshold decision
will be replaced by an independent Bernoulli KEEP/REMASK policy. The initial
feature set will contain current confidence, commit confidence, confidence
change, token age, remask count, and normalized timestep. LLaDA remains frozen.
