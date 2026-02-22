# Evaluation Metrics for Multi-turn Code Generation

To better evaluate an LLM's ability to handle ambiguous requirements and ask clarification questions for code generation tasks, we propose two new metrics: **Turn-discounted Key Question Rate (TKQR)** and **Optimal Round Adherence (ORA)**. These metrics explicitly model interaction efficiency, since early identification of key missing information is more valuable and extra turns impose a measurable dialogue cost. We introduce TKQR and ORA in the following sections.

---

## Evaluation Metrics

### 1. Turn-discounted Key Question Rate (TKQR)

We define **Turn-discounted Key Question Rate (TKQR)** by adapting normalized DCG to reward asking key clarification questions **early**. The metric favors dialogues in which the model covers annotated key questions in the first few turns rather than delaying them or spending turns on non-key questions.

#### Notation and Definitions

- Let **n** be the total number of dialogue turns before the model stops asking questions.
- Let **K** be the number of annotated key questions for the task.
- We define a **hit sequence** $H = (h_1, \dots, h_n)$, where $h_i \in \{0, 1\}$:
  - $h_i = 1$ if the model asks a **previously uncovered** key question at turn $i$;
  - $h_i = 0$ otherwise.

#### Discounted Cumulative Gain (DCG)

We compute a discounted gain that favors early hits:

$$
\mathrm{DCG}_n = \sum_{i=1}^{n} \frac{h_i}{\log_2(i+1)}.
$$

#### Ideal DCG (IDCG)

To make scores comparable across tasks with different $K$, we normalize by the ideal case. The ideal case asks key questions as early as possible, using at most $\min(n, K)$ turns:

$$
\mathrm{IDCG}_n = \sum_{i=1}^{\min(n,K)} \frac{1}{\log_2(i+1)}.
$$

#### TKQR Definition

TKQR is the normalized ratio:

$$
\mathrm{TKQR} = \frac{\mathrm{DCG}_n}{\mathrm{IDCG}_n}.
$$

- **Range:** $[0, 1]$.
- **Interpretation:** TKQR increases when the model covers key questions earlier; it decreases when the model delays key questions or spends turns on non-key questions.

---

### 2. Optimal Round Adherence (ORA)

**Optimal Round Adherence (ORA)** measures whether a model uses a **near-optimal** number of clarification rounds. We treat the number of question-asking rounds as a **cost**, because extra rounds increase interaction overhead.

#### Notation and Definitions

- Let **n** be the number of rounds in which the model asks clarification questions.
- Let  $\mathcal{Q}$ be the set of annotated key questions for the task.
- We define the **optimal interaction round count** as $K = |\mathcal{Q}| + 1$. The extra round accounts for the final step that stops questioning and transitions to code generation.

#### ORA Formula

ORA assigns the highest score when $n = K$, and it decreases as $n$ moves away from $K$. We use a Gaussian-shaped penalty:

$$
\mathrm{ORA}(n, K, \sigma) = \exp\left( -\frac{(n - K)^2}{2\sigma^2} \right).
$$

This form is bounded in $(0, 1]$ and smoothly decays with the deviation $|n - K|$.

#### Setting the Hyperparameter $\sigma$

We set $\sigma$ so that $\mathrm{ORA} = 0.5$ when $|n - K| = 0.5K$. This yields:

$$
\sigma = \frac{0.5K}{\sqrt{2\ln 2}} \approx 0.425K.
$$

#### Relationship to TKQR

- **TKQR** checks whether key questions appear **early** in the dialogue.
- **ORA** checks whether the model **stops at a reasonable time** (i.e., uses a near-optimal number of rounds).

Together, TKQR and ORA evaluate both *which* key questions are asked and *when*, and whether the number of clarification rounds is appropriate.

---

## Summary

| Metric | Focus | Range |
|--------|--------|--------|
| **TKQR** | Whether key questions are covered early | $[0, 1]$ |
| **ORA**  | Whether the number of clarification rounds is near-optimal | $(0, 1]$ |

The two metrics jointly assess an LLM's ability to ask clarification questions under ambiguous requirements and its **interaction efficiency**.
