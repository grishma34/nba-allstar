# KNOWLEDGE_GAPS.md

Required by the assignment specification:

> "If there are components you used but do not fully understand, explicitly state
> this and explain: Why the component is necessary. How you verified its correctness.
> What attempts you made to understand the underlying mechanism.
>
> Note: Documenting your thought process and verification attempts will protect you
> from penalties if questioned during the presentation."

Add an entry the moment something is unclear. This section is an asset, not an
admission — the A3 criteria list "I don't know, here's how I'd check" as a GOOD
response worth credit, and a confident vague answer as worth zero.

---

## Template

```
## [Component]
**What it does:**
**Why it's necessary:**
**What I don't fully understand:**
**How I verified it works:**
**What I did to understand it:**
```

---

## Example entry (delete once you have real ones)

## Epsilon clipping in log loss
**What it does:** Clips predicted probabilities into [1e-15, 1-1e-15] before taking
the log.
**Why it's necessary:** log(0) is negative infinity. If the model ever outputs
exactly 0 or 1, the loss becomes undefined and training breaks.
**What I don't fully understand:** Why 1e-15 specifically rather than some other
small value, and whether the choice measurably affects the fitted parameters.
**How I verified it works:** Unit test - passed y_pred containing exact 0 and 1 and
confirmed the loss returns a finite number rather than inf or nan.
**What I did to understand it:** Read the numpy float64 precision limits; tested
1e-10 and 1e-20 and compared the resulting loss values and final weights.
