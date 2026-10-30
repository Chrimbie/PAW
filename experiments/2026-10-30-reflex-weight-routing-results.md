# PAW / ReflexStore weight-routing experiments — 2026-10-30

## Status and scope
Experimental notes from interactive Python runs, **not independently reproduced or CI-verified**. Tests used the supplied `reflex_optimized.py` (`ReflexStore` in append/index mode), NumPy neural networks, synthetic data, and temporary on-disk stores. These results do **not** establish production-ready PAW, semantic question understanding, neuron-level saturation offloading, or general catastrophic-forgetting prevention. Original runnable test harnesses and raw logs were not retained as files.

## Sequence of observed results

| Experiment | Key observations | Caveats |
|---|---|---|
| LRU cache, 12 synthetic configurations, 2,400 queries | 1,072 hits (44.67%); 100% agreement with original configuration predictions; active state unchanged. Direct `get`: 2.324 µs; cached lookup: 10.67 µs. | Timing was **not apples-to-apples** (cache path included decoding/management). IDs were supplied, not learned. |
| Learned selector, 12 well-separated synthetic contexts, 128 KiB/configuration | Selector 100%; cache hit 30.92%; payload integrity 100%; cached loop 135.19 ms vs direct 69.44 ms. | Easy synthetic contexts, no actual trained weights, unequal work in timing loops. |
| Two trained 2→12→1 networks, sequential conflicting tasks, 800 examples/task | Task A before B 99.25%; after B 1.75%; B 99.88%; selector 99.19%; reequipped A 98.25%, B 99.38%; cache 1,572 hits/28 misses. | Evaluation reused training examples; routing contexts well separated. Full-network snapshots, not individual neurons. |
| Harder overlapping contexts, separate held-out test sets | A before B 100%; sequential A 0.50%, B 99.67%; selector 72.92%; reequipped A 73.83%, B 71.67%. **Oracle routing:** A 100%, B 99.67%. | Overlapping task contexts create irreducible ambiguity when the input does not identify the intended task. |
| Question hash + token similarity, tiny arithmetic text examples | Exact 4/4; paraphrases 6/6; naive baseline 5/10; ambiguous **0/2 rejected**. | Handcrafted toy strings and placeholder stored payloads; not trained networks. |
| Question-linked routing + trained addition/subtraction *binary classifiers*, held-out numeric inputs | A initial 99.67%; sequential A 48.33%, B 99.67%; routing 100%; end-to-end 99.67% over 600 held-out numeric examples; cache 598 hits/2 misses; ambiguous 3/3 rejected. | **Operation cue vocabulary was manually hard-coded** (add/plus/sum vs subtract/minus/difference). Not learned question-to-weight associations. Networks classified signs of sums/differences; they did **not** calculate arithmetic answers. |

## Architecture actually demonstrated
1. Train a small NumPy network on Task A.
2. Serialize complete network parameters and save to ReflexStore.
3. Reuse the same active network for Task B; save its full parameters.
4. Route an incoming question to a configuration via a simple selector or handcrafted operation cues.
5. Retrieve the complete configuration from ReflexStore or a RAM LRU cache and perform inference without modifying active learning weights.

**Key finding:** stored full-network configurations can recover performance after sequential training overwrites the active model, **provided routing is correct**. Under overlapping contexts, routing—not persistence—was the main observed failure. Explicit question cues improved routing in a tiny controlled test; they were not automatically learned.

## Next reproducible experiments
- Save standalone harness, fixed seeds, exact environment/dependencies, and raw per-example predictions.
- Compare learned question-to-weight association against explicit cue routing, including paraphrases, out-of-domain inputs, and ambiguous questions.
- Evaluate on disjoint train/validation/test sets, multiple seeds, and tasks with overlapping inputs and meaningful context.
- Compare cache vs direct ReflexStore under identical decoding and access patterns, with warm/cold cache separation.
- Test actual **per-neuron** incoming/outgoing weights and bias swapping, neuron alignment, and activation/gradient-based saturation criteria.

Source: interactive experiment results in ChatGPT conversation, October 30, 2026. No claim of independent reproduction.
