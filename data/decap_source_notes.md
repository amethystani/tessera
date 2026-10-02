# Cross-paper replication attempt for the coverage/disposition decomposition

Log of every paper checked as a candidate for extending the decomposition in
`decap_decomposition.py` beyond DeCAP. Recorded because the negative results
matter: they are the reason the manuscript reports 32 pairs from one
paper rather than a broader sweep, and they are themselves evidence about
reporting practice.

The decomposition needs exactly two published numbers per method-model pair,
on the **ambiguous** split:

1. accuracy on ambiguous items (which, because the gold answer on ambiguous
   BBQ items IS the UNKNOWN option, equals the abstention rate `p`), and
2. the ambiguous bias score, defined by BBQ as `s_AMB = (1 - accuracy) * s_DIS`.

A paper that omits either, or that reports a differently-normalised quantity
under the name "bias score", cannot be decomposed.

## Confirmed: the identity itself

**Parrish et al. (2022), BBQ: A Hand-Built Bias Benchmark for Question
Answering**, Findings of ACL 2022, arXiv:2110.08193.
Checked 2026-09-17 via ar5iv HTML. The paper states the definition verbatim as
`s_Amb = (1 - accuracy) x s_Dis`, motivated as reflecting "that a biased answer
is more harmful if it happens more often." This is the authoritative source for
the identity the decomposition uses.

**Not usable as a data source.** The paper reports ambiguous-context accuracy
and bias score only in Figure 3 and Figure 5, without labelled numerical
values. Extracting per-model ambiguous pairs would require re-running the
authors' models, not reading their tables.

## Usable

**Bae, Choi & Lee (2025), DeCAP**, NAACL 2025, arXiv:2503.19426.
Table 11(a) reports ambiguous-split accuracy and bias score for 5 methods
(Base, Self-Debiasing, Def-1, Def-2, DeCAP) x 8 models. Four interventions
against their own Base row gives 32 decomposable pairs. Transcribed to
`decap_bbq_ambiguous.csv`. This is the dataset used in the manuscript.

## Rejected: metric definition incompatible

**Wu et al. (2025), Does Reasoning Introduce Bias? A Study of Social Bias
Evaluation and Mitigation in LLM Reasoning**, arXiv:2502.15361.
Table 1(a) reports BBQ ambiguous accuracy and a "bias score" per model:

| Model | Accuracy | Bias score | implied b_cond |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.82 | 0.56 | 3.11 |
| Qwen2.5-32B-Instruct | 0.93 | 0.34 | 4.86 |
| Marco-o1 | 0.87 | 0.55 | 4.23 |
| DeepSeek-R1-Distill-Llama-8B | 0.90 | 0.51 | 5.10 |
| DeepSeek-R1-Distill-Qwen-32B | 0.95 | 0.51 | 10.20 |

**Rejected.** Recovering `b_cond = s / (1 - accuracy)` yields 3.1 to 10.2 for
every model. A bias score among committed answers is bounded in [-1, 1] by
construction, so the reported quantity cannot be BBQ's `s_AMB` under BBQ's own
definition, and the identity does not apply. We do not know which variant
definition is in use; the point is only that it is not the one the
decomposition requires, so these numbers are excluded rather than
force-fitted.

**Not checked further:** Yang et al. (2025), "Rethinking Prompt-based
Debiasing in Large Language Models" (arXiv:2503.09219) evaluates on BBQ and
argues prompt-based debiasing shows "false prosperity" partly due to "flawed
evaluation metrics", which is the same concern as this work, but the
abstract-level content available did not expose a per-method ambiguous
accuracy/bias-score table, and the PDF did not parse to extractable tables.
Listed here as an open lead rather than a checked negative.

## Conclusion recorded in the manuscript

Of the papers checked, one reports both required quantities in extractable
form, one defines its ambiguous bias score incompatibly with BBQ's own
definition, and the benchmark's originating paper reports the relevant split
only in unlabelled figures. That is why the decomposition covers
one paper's harness, and it supports the manuscript's recommendation that
abstention rate and committed-answer bias be reported alongside any headline
bias score.

## Rejected: metric definition incompatible (third case)

**Kamoi et al. / Intent-Aware Self-Correction for Mitigating Social Biases in
Large Language Models**, arXiv:2503.06011. Checked 2026-09-17 by reading the
PDF directly.

Table 1 reports, for three models (GPT-3.5, GPT-4o-mini, LLaMA-3 70B Instruct)
across twelve reasoning/correction methods, an **Accuracy** column and a
**Diff-bias** column on BBQ. The qualitative pattern is the one this
paper is about, e.g. GPT-3.5 moves from accuracy 0.477 / diff-bias 0.221
(No-CoT) to accuracy 0.938 / diff-bias 0.028 (cross-model correction with a
debiasing prompt), a large accuracy gain accompanying a large bias-score drop.

**Rejected on two independent grounds:**
1. "Diff-bias" is a custom metric (the paper defines it as the difference
   between biased and counter-biased answer rates, where "a higher diff-bias
   score indicates a greater alignment of biases to social stereotypes"), not
   BBQ's `s_AMB`. The identity `s_AMB = (1 - acc) * b_cond` does not apply to
   it.
2. Table 1 aggregates over BBQ's nine categories without separating the
   ambiguous split, so even under a compatible metric the abstention rate
   (which is only equal to accuracy on the ambiguous split) could not be
   recovered.

This is the third paper checked whose reported quantities cannot be decomposed,
after Wu et al. (incompatible bias-score normalisation) and BBQ's own paper
(ambiguous split reported only in unlabelled figures). Recorded because the
pattern, that the numbers needed to audit this confound are usually not the
numbers that get printed, is itself the argument for the reporting
recommendation in the paper's conclusion.

## Adjacent work, cited rather than decomposed

**Manduru & Domeniconi (2026), Beyond Bias Scores: Unmasking Vacuous
Neutrality in Small Language Models**, EACL 2026 SRW, arXiv:2506.08487.
Names the same failure mode ("vacuous neutrality": models posting low bias
scores while failing downstream reasoning checks) for 0.5-5B models. Cited in
related work. Not used as a decomposition source, since its framework reports
staged evaluation outcomes rather than the paired (ambiguous accuracy,
ambiguous s_AMB) quantities the identity needs.
