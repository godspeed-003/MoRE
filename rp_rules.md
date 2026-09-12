# RP Rules — NeurIPS Workshop Paper Writing Rules

## 0. Purpose

This file defines the rules Claude must follow when writing, editing, restructuring, or extending the research paper in LaTeX.

The paper should use the uploaded/reference paper as a **structural and stylistic reference**, but it must be a **new paper with its own scientific story, claims, experiments, and wording**.

The most important constraint is:

> **The main paper content must fit within a hard maximum of 8 pages in the final NeurIPS workshop LaTeX rendering.**

Do not solve the page-limit problem by shrinking fonts, margins, or spacing. Compression must come from better writing, structure, figure/table design, and removal of redundancy.

---

# 1. Primary Objectives

Claude must optimize the paper for:

1. Scientific clarity.
2. A coherent and defensible central claim.
3. Strong experimental evidence.
4. Reproducibility.
5. Concise NeurIPS-style presentation.
6. Efficient use of the 8-page main-content budget.
7. Correct LaTeX compilation.
8. Internal consistency between abstract, introduction, methodology, experiments, discussion, and conclusion.

The paper must read like a **research paper**, not a project report, thesis chapter, implementation diary, or documentation file.

---

# 2. Reference-Paper Rule

The supplied/reference paper is a **model for organization, level of technical detail, figure/table integration, and academic tone**.

Claude may learn from its:

- section hierarchy,
- scientific narrative,
- experiment presentation,
- table organization,
- figure placement,
- mathematical notation style,
- LaTeX conventions,
- discussion structure,
- limitation reporting,
- citation style.

Claude must NOT:

- copy sentences,
- copy paragraphs,
- imitate distinctive wording,
- invent claims merely because they appeared in the reference,
- transfer numerical results,
- transfer experimental conclusions,
- assume that the reference paper's methodology is identical to the new paper.

The new paper's scientific content always takes priority over resemblance to the reference paper.

---

# 3. Hard Page-Budget Rule

## 3.1 Maximum

The final rendered manuscript must have:

> **≤ 8 pages of main paper content.**

Treat 8 pages as a hard upper bound, not a target to exceed slightly.

## 3.2 Safety Margin

Claude should normally target approximately:

> **7.0–7.7 pages**

for the main content so that small LaTeX changes do not unexpectedly push the paper over 8 pages.

Never deliberately target exactly 8.0 pages.

## 3.3 References

References may be outside the 8-page main-content budget **only if the applicable NeurIPS workshop template/CFP explicitly permits this**.

Never assume that references are excluded. Check the applicable submission rules when this matters.

If the workshop rules are unknown, write the paper so that the **core scientific content itself fits comfortably within 8 pages**, while keeping references compact but readable.

## 3.4 What Counts as Main Content

Unless the workshop explicitly states otherwise, treat the following as main-content material:

- title and author block,
- abstract,
- introduction,
- related work,
- methodology,
- mathematical formulation,
- system/architecture description,
- experiments,
- tables,
- figures,
- ablations,
- analysis,
- discussion,
- limitations,
- conclusion.

Do not attempt to classify oversized figures, tables, captions, appendices, or supplementary material as "not part of the paper" merely to bypass the page limit.

---

# 4. Page-Budget Allocation

Use the following approximate budget as a planning constraint.

| Section | Approx. Budget |
|---|---:|
| Abstract | 0.20–0.30 page |
| Introduction | 0.9–1.2 pages |
| Related Work | 0.5–0.8 pages |
| Methodology | 1.7–2.1 pages |
| Experiments / Setup | 0.7–1.0 pages |
| Results | 1.2–1.6 pages |
| Discussion / Limitations | 0.5–0.8 pages |
| Conclusion | 0.2–0.3 page |
| Figures/Tables | integrated into the above budget |

These are planning ranges, not rigid requirements.

If the paper is too long:

1. remove redundancy,
2. merge overlapping subsections,
3. move secondary analysis to supplementary material if permitted,
4. consolidate tables,
5. combine related figures,
6. shorten captions,
7. replace repeated prose with precise equations/tables,
8. remove low-value discussion.

Do **not** reduce readability by using tiny fonts or abnormal spacing.

---

# 5. Required Scientific Story

Every paper must have one clear central story.

Claude must be able to answer:

> What is the problem?
>
> Why does the problem matter?
>
> What is the proposed contribution?
>
> What exactly was evaluated?
>
> What did the experiments demonstrate?
>
> What did they NOT demonstrate?

The paper should follow a logical progression:

**Problem → Gap → Hypothesis/Idea → Method → Experimental Test → Evidence → Interpretation → Limitations → Conclusion**

Do not introduce a claim in the Discussion that was not supported by the experiments.

Do not introduce a method detail in Results that was never defined in Methodology.

Do not make the Conclusion stronger than the evidence.

---

# 6. Claim Discipline

This is one of the highest-priority rules.

Claude must distinguish between:

- observed result,
- measured association,
- interpretation,
- hypothesis,
- causal claim,
- general claim.

Do not convert a correlation into causation.

Do not use words such as:

- "proves",
- "demonstrates that X causes Y",
- "solves",
- "eliminates",
- "guarantees",
- "is the dominant bottleneck",
- "fundamentally resolves",

unless the experimental design genuinely supports those statements.

Prefer precise formulations such as:

- "our results suggest..."
- "under the evaluated setting..."
- "we observe..."
- "the results are consistent with..."
- "this indicates..."
- "we find evidence that..."
- "within this benchmark/configuration..."

---

# 7. Internal-Consistency Rule

The paper must never contradict itself.

Before finalizing, Claude must cross-check:

### Abstract ↔ Results
Every quantitative statement in the abstract must match the final tables/results exactly.

### Introduction ↔ Method
Every claimed contribution must actually be implemented.

### Method ↔ Experiments
Every experimental condition must be defined clearly enough to understand what was compared.

### Results ↔ Discussion
Interpretations must follow from the measured evidence.

### Discussion ↔ Conclusion
The conclusion must not exaggerate what the discussion qualifies.

### Tables ↔ Text
Numbers, baselines, labels, seeds, metrics, and statistical claims must match.

### Captions ↔ Figures
Captions must accurately describe what is visible and measured.

---

# 8. Numerical Integrity

Claude must NEVER fabricate:

- results,
- means,
- standard deviations,
- confidence intervals,
- p-values,
- sample counts,
- training steps,
- model sizes,
- latency measurements,
- retrieval rates,
- ablation results,
- benchmark scores.

If a number is not available, use an explicit placeholder such as:

`[RESULT NEEDED]`

rather than inventing a plausible value.

If the source material contains conflicting numbers:

1. identify the conflict,
2. do not silently choose one,
3. flag it for resolution,
4. use the verified value only after resolution.

All reported means/standard deviations must correspond to the stated number of seeds.

---

# 9. Experimental Claim Rule

For every major claim, Claude should ask:

> "Which experiment directly supports this sentence?"

If there is no direct supporting experiment, weaken or remove the claim.

A strong paper should separate:

### What was tested
Example:
- static retrieval,
- dynamic retrieval,
- LoRA adaptation,
- gold retrieval,
- retrieval coverage,
- latency.

### What was observed
Example:
- F1 increased,
- retrieval coverage increased,
- latency increased.

### What is inferred
Example:
- evidence utilization may remain a limitation.

### What is NOT established
Example:
- retrieval is not necessarily the sole bottleneck,
- dynamic retrieval may not generalize beyond the evaluated benchmark.

---

# 10. Avoid Contradictory Bottleneck Claims

Never make a statement such as:

> "Retrieval is the hard bottleneck"

if the same paper shows that gold retrieval provides substantially more available evidence but generation still performs poorly.

Instead, reason explicitly from the experiment.

For example:

- If retrieval coverage is low and gold retrieval dramatically improves generation, evidence availability may be a major limitation.
- If gold retrieval is high but F1 remains low, retrieval availability alone is insufficient to explain failure.
- The appropriate interpretation may instead involve evidence utilization, multi-hop synthesis, generation quality, representation mismatch, or another experimentally supported factor.

The wording must follow the evidence.

---

# 11. Contribution Rules

The Introduction should contain a concise contribution list.

Each contribution must be:

1. genuinely new or meaningfully useful,
2. technically specific,
3. supported by the paper,
4. distinguishable from prior work.

Avoid vague contributions such as:

- "We improve AI."
- "We propose a novel framework."
- "We achieve better performance."

Prefer:

- "We introduce X, a mid-denoising retrieval mechanism that..."
- "We evaluate X against Y under..."
- "We provide an analysis showing..."
- "We characterize the failure mode of..."

Do not claim novelty merely because a combination of known components has been implemented.

---

# 12. Related Work Rules

Related Work should establish the research gap, not become a literature catalogue.

For each important prior work, explain:

1. what it does,
2. what limitation matters for this paper,
3. how the present work differs.

Avoid long lists of papers with no synthesis.

Use citations where claims about prior work are made.

Never fabricate citations or bibliographic details.

---

# 13. Methodology Rules

Methodology must be concise but reproducible.

At minimum, clearly specify where applicable:

- model/backbone,
- dataset,
- retrieval corpus/index,
- retriever,
- embedding model,
- retrieval depth/top-k,
- query construction,
- generation procedure,
- denoising procedure,
- adaptation method,
- LoRA configuration,
- training data,
- evaluation data,
- random seeds,
- evaluation metrics,
- important hyperparameters.

Do not waste page space explaining standard concepts at textbook length.

Explain only what is necessary to understand:

- what is new,
- what differs from baselines,
- how the experiment was conducted.

---

# 14. Distinguish Original Algorithms from Implementations

If the paper implements a method inspired by a prior algorithm, Claude must use precise terminology.

Examples:

- "SARDI-inspired implementation"
- "inspired by SARDI"
- "our adaptation of the SARDI retrieval idea"

Do NOT call an implementation "SARDI" if it is not a faithful reproduction.

Explicitly state important deviations.

This prevents reviewers from interpreting an approximate implementation as an exact reproduction.

---

# 15. Dataset and Leakage Rules

Every evaluation setup must clearly distinguish:

- training/adaptation data,
- validation data,
- test/evaluation data.

If a subset is frozen, state that clearly.

If leakage was checked, state the procedure and result.

Never claim "zero overlap" unless it was actually verified.

If the evaluation uses a small subset, do not generalize results to the full benchmark without justification.

If the evaluation is only 100 questions, say so where scientifically relevant.

---

# 16. Statistical Reporting

Whenever multiple seeds are used:

- report mean,
- report standard deviation where appropriate,
- state number of seeds,
- keep the metric definition consistent.

Example:

`8.86 ± 1.64% (3 seeds)`

Do not imply statistical significance from a mean difference alone.

If significance testing is used, report the test and its assumptions appropriately.

If no significance test was performed, do not use "statistically significant."

---

# 17. Small-Sample Caution

When evaluation uses a small benchmark subset:

- explicitly state the sample size,
- avoid broad generalizations,
- emphasize that results are preliminary if appropriate,
- distinguish empirical observation from general capability.

Do not describe a result on a small frozen subset as establishing universal behavior.

---

# 18. Figures

Figures should communicate information that is difficult to communicate efficiently with prose.

Prioritize:

1. overall architecture,
2. experimental workflow,
3. key quantitative result,
4. important diagnostic analysis.

Avoid decorative figures.

Every figure must have:

- a useful caption,
- readable labels,
- consistent terminology,
- sufficient resolution,
- a clear purpose.

The figure should be understandable from the caption plus nearby text.

Do not create one figure for information that could be represented more efficiently as a table.

---

# 19. Tables

Tables should be compact and information-dense.

Prefer:

- one main results table,
- one focused diagnostic/ablation table,
- one setup table only when necessary.

Avoid multiple tables repeating the same metrics.

Every table should answer a reviewer question.

Examples:

- "Does the proposed method improve performance?"
- "Does LoRA matter?"
- "Does retrieval availability explain the result?"
- "What is the latency trade-off?"

Do not repeat every number in prose if the table already communicates it.

---

# 20. Captions

Captions must be concise but scientifically useful.

A good caption should explain:

- what is shown,
- what conditions are compared,
- what the reader should notice.

Do not write a paragraph-length caption merely to move methodology out of the main text.

---

# 21. Equations

Use equations when they clarify the method.

Every nontrivial variable must be defined.

Notation must remain consistent throughout the paper.

Do not introduce multiple symbols for the same concept.

Avoid equations that merely restate obvious implementation details.

---

# 22. LaTeX Rules

Use the official/appropriate NeurIPS template supplied for the target submission.

Claude must preserve:

- the required document class,
- official package conventions,
- anonymization requirements,
- bibliography format,
- submission formatting requirements.

Do not modify:

- margins,
- font sizes,
- line spacing,
- page geometry,

to artificially fit the paper into 8 pages.

Use standard LaTeX constructs such as:

- `figure`,
- `table`,
- `figure*`,
- `table*`,
- `\begin{minipage}`,
- `booktabs`,
- `subcaption`,

only when appropriate and compatible with the template.

---

# 23. LaTeX Compilation Rule

All generated LaTeX must be syntactically valid.

Before considering a section finished, check for:

- unmatched `{}`,
- unmatched `\begin{}` / `\end{}`,
- broken math mode,
- malformed citations,
- undefined labels,
- duplicate labels,
- missing figures,
- invalid file paths,
- accidental Markdown syntax,
- malformed table rows,
- broken special characters.

Do not leave pseudo-LaTeX or explanatory comments in places where actual compilable LaTeX is required.

---

# 24. Cross-Reference Rules

Use labels and references consistently.

For example:

- `\label{fig:architecture}`
- `\ref{fig:architecture}`
- `\label{tab:main-results}`
- `\ref{tab:main-results}`
- `\label{sec:method}`
- `\ref{sec:method}`

Never hard-code figure/table numbers in prose.

Avoid duplicate labels.

---

# 25. Writing Style

Use a professional, technical, concise research style.

Prefer:

> "We evaluate..."

over:

> "In this paper, we are trying to..."

Prefer:

> "We observe a 2.1-point improvement..."

over:

> "This gives a very significant improvement..."

Avoid:

- marketing language,
- excessive adjectives,
- rhetorical filler,
- conversational language,
- unsupported superlatives,
- repeated explanations,
- vague claims.

Use domain-appropriate terminology.

---

# 26. Abstract Rules

The abstract should fit comfortably within approximately 150–200 words unless the venue specifies otherwise.

It should contain:

1. problem,
2. approach,
3. experimental setting,
4. main result,
5. important interpretation/limitation.

Do not put every implementation detail in the abstract.

Do not claim more than the experiments establish.

Every numerical result in the abstract must exactly match the final results.

---

# 27. Introduction Rules

The Introduction should establish the motivation quickly.

Recommended structure:

### Paragraph 1 — Problem
What problem exists?

### Paragraph 2 — Why existing approaches struggle
What is the technical gap?

### Paragraph 3 — Our approach
What do we do differently?

### Paragraph 4 — Experimental finding
What did we actually observe?

### Contributions
Use a concise list.

Do not spend most of the Introduction explaining background that belongs in Related Work.

---

# 28. Results Rules

Results should be evidence-first.

Recommended ordering:

1. main quantitative result,
2. baseline comparison,
3. ablation,
4. diagnostic analysis,
5. efficiency/latency,
6. interpretation.

For every result:

- identify the metric,
- identify the comparison,
- report uncertainty when available,
- explain the implication,
- avoid repeating the table verbatim.

---

# 29. Discussion Rules

Discussion is for interpretation, not for introducing new experimental results.

It should answer:

- Why might the result have occurred?
- What does the result imply?
- Which hypothesis is supported?
- Which hypothesis is not supported?
- What remains unresolved?

If an explanation is speculative, label it as such.

---

# 30. Limitations

Limitations should be explicit and credible.

Possible limitations include:

- small evaluation subset,
- benchmark-specific behavior,
- limited seeds,
- latency overhead,
- retrieval-index assumptions,
- approximate reproduction of prior methods,
- weak absolute performance,
- limited generalization evidence.

Do not hide weaknesses.

A strong limitation section increases reviewer trust when it is precise.

---

# 31. Efficiency / Latency Claims

If the method introduces additional retrieval or generation steps, report the computational/latency consequence when available.

Do not claim that an iterative method is "efficient" merely because it is parameter-efficient.

Distinguish:

- parameter efficiency,
- memory efficiency,
- retrieval efficiency,
- wall-clock latency,
- throughput,
- training cost,
- inference cost.

If latency is measured only within one experimental condition, do not make cross-condition claims that the experiment did not establish.

---

# 32. Main-Text Compression Strategy

When the paper exceeds 8 pages, Claude must NOT immediately delete important scientific information.

Apply this order:

### Step 1
Remove repeated explanations.

### Step 2
Merge overlapping subsections.

### Step 3
Shorten background/related work.

### Step 4
Consolidate tables.

### Step 5
Combine related figures.

### Step 6
Move secondary analysis to appendix/supplementary material if the venue permits.

### Step 7
Compress captions.

### Step 8
Rewrite long sentences.

### Step 9
Remove low-value implementation details.

### Step 10
Only then reconsider whether any nonessential scientific material should be removed.

Never sacrifice reproducibility or the central evidence merely to save space.

---

# 33. Section Hierarchy for an 8-Page Paper

A strong default structure is:

```text
\section{Introduction}
\section{Related Work}
\section{Methodology}
    \subsection{Problem Formulation}
    \subsection{Proposed Method}
    \subsection{Implementation Details}
\section{Experiments}
    \subsection{Experimental Setup}
    \subsection{Main Results}
    \subsection{Analysis / Ablations}
\section{Discussion and Limitations}
\section{Conclusion}
```

Do not create many tiny subsections.

A subsection should exist only if it helps navigation or scientific clarity.

---

# 34. What Must Be in the Main Paper

Prioritize the following:

- central problem,
- core method,
- experimental setup,
- main result,
- most important ablation,
- most important diagnostic analysis,
- limitations.

Secondary implementation details may be shortened or moved to supplementary material if allowed.

---

# 35. What Should Usually Be Removed or Minimized

Unless essential:

- generic explanations of transformers,
- generic explanations of LoRA,
- generic explanations of RAG,
- long descriptions of standard datasets,
- repeated model descriptions,
- repeated metric definitions,
- implementation trivia,
- redundant result tables,
- repetitive prose around figures,
- long literature summaries.

The paper should spend its limited pages on **what is scientifically new and what the experiments show**.

---

# 36. Reviewer-Oriented Self-Check

Before finalizing, Claude must mentally evaluate the paper from a skeptical reviewer perspective.

Ask:

### Novelty
What is actually new?

### Validity
Do the experiments support the claims?

### Baselines
Are the comparisons fair?

### Reproducibility
Could another researcher reproduce the setup?

### Statistics
Are the conclusions justified by the number of seeds/examples?

### Scope
Are the claims limited to what was tested?

### Contradictions
Does any section undermine another section?

### Page limit
Does the main content fit within 8 pages?

### Clarity
Can the central idea be understood quickly?

### Weaknesses
Are limitations honestly reported?

---

# 37. Mandatory Final Consistency Audit

Before delivering a final LaTeX manuscript, Claude must perform a final audit covering:

- [ ] Main content ≤ 8 pages.
- [ ] No artificial margin/font manipulation.
- [ ] Abstract matches final results.
- [ ] Introduction claims match actual contributions.
- [ ] Every contribution is supported.
- [ ] All numerical values are verified.
- [ ] No fabricated results.
- [ ] No fabricated citations.
- [ ] Dataset splits are clear.
- [ ] Leakage claims are verified.
- [ ] Number of seeds is stated.
- [ ] Metrics are defined.
- [ ] Baselines are clearly described.
- [ ] Proposed method is reproducible.
- [ ] Prior methods are not misrepresented.
- [ ] Figures are referenced in the text.
- [ ] Tables are referenced in the text.
- [ ] Captions match figures/tables.
- [ ] Labels are unique.
- [ ] LaTeX environments are balanced.
- [ ] No undefined variables.
- [ ] No contradictory claims.
- [ ] Discussion does not overstate results.
- [ ] Limitations are explicit.
- [ ] Conclusion is evidence-consistent.
- [ ] Bibliography/citations compile correctly.
- [ ] Paper follows the target NeurIPS workshop's current formatting rules.

---

# 38. Important Instruction to Claude

When asked to "write the paper," do not immediately generate large amounts of prose.

First determine:

1. the central claim,
2. the contributions,
3. the experimental evidence available,
4. the required sections,
5. the page budget,
6. the figures/tables that deserve main-paper space,
7. the claims that must be weakened or qualified.

Then write the LaTeX around that evidence.

If information is missing, use a clearly marked placeholder or ask for the missing information rather than inventing it.

**Scientific correctness takes priority over completing the prose.**

**The 8-page limit takes priority over unnecessary detail.**

**Evidence takes priority over narrative.**

**Precision takes priority over persuasive wording.**
