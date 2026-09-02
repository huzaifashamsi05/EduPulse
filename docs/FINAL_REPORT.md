# EduPulse — Final Report

**Machine Learning Capstone 2026**
**Author:** Muhammad Huzaifa (solo)
**Repository:** https://github.com/huzaifashamsi05/EduPulse

---

## 1. Problem & Motivation

Academic advisors at most universities are responsible for far more students than
they can individually track in depth. By the time a student's disengagement
becomes visible through obvious signs — missed classes, failing grades, withdrawal
— the window for effective intervention has often already narrowed. EduPulse asks
a specific, bounded question: can a student's enrollment and first/second-semester
academic record be used to flag likely outcomes early enough that a human advisor
can act on it, presented in a way that is honest about the model's confidence and
limitations rather than dressed up as certainty?

This is deliberately framed as **decision support**, not automation. The product
and this report both treat that distinction as non-negotiable — see Section 7.

## 2. Data

The UCI "Predict Students' Dropout and Academic Success" dataset (4,424 records,
36 features, three-class target: Dropout / Enrolled / Graduate) was used as
specified in the project brief. The dataset's own documentation confirms no
missing values and a moderate class imbalance (Graduate ~50%, Dropout ~32%,
Enrolled ~18%), both of which I independently verified after loading it (see
`ml/data/SOURCE.md` for the exact verification steps and a note on how the data
was retrieved in a network-restricted development environment).

**Leakage consideration:** the 36 features include second-semester academic
performance (units approved, grades, etc.). This is not treated as leakage,
because it reflects information that would genuinely be available to an advisor
partway through a student's first year — it is the intended prediction point, not
a look-ahead into the final outcome itself. This is explicitly the framing the
UCI dataset was designed around.

## 3. Methodology Summary

- **Split:** 70/15/15 train/validation/test, stratified on the target class, with
  the test set held out untouched until final model selection.
- **Preprocessing:** one serialized `ColumnTransformer` — StandardScaler for
  numeric features, OneHotEncoder for genuinely nominal categorical features
  (Course, Nationality, parental occupation/education codes, etc.), passthrough
  for already-binary fields. This transformer is bundled inside the same
  `Pipeline` object as the trained estimator and serialized as one artifact, so
  the FastAPI backend never re-implements or risks diverging from the training
  preprocessing logic.
- **Models compared:** Logistic Regression, Random Forest, and XGBoost, each
  evaluated with 5-fold stratified cross-validation on the training split.
- **Selection metric:** macro-F1, not accuracy — reasoning detailed in Section 4.
- **Explainability:** SHAP, with the explainer type (`TreeExplainer` vs.
  `LinearExplainer`) chosen automatically based on which model won selection,
  since that is not fixed in advance (see Section 4's reproducibility note).

Full technical detail is in `README.md`; this report focuses on reasoning and
reflection rather than repeating that documentation.

## 4. Model Selection Reasoning

Accuracy alone is a poor selection criterion here because the classes are
imbalanced enough that a model could score well by mostly predicting the
majority "Graduate" class and still perform badly on "Enrolled" — arguably the
group where early intervention has the most value, since these are students
whose trajectory hasn't resolved yet. Macro-F1 was used instead because it
weights all three classes equally regardless of how many examples each has.

An unplanned but genuinely useful finding came out of this process: Logistic
Regression and XGBoost scored within roughly 0.0002 macro-F1 of each other
during cross-validation — a statistical tie. Testing on a second machine with
slightly different NumPy/SciPy versions actually flipped which model "won,"
purely due to floating-point differences in the underlying computation. Rather
than treat this as a bug to suppress, I treated it as a real finding: **when two
models are this close, "the best model" is not a fixed, portable fact** — it can
depend on measurement noise. This directly shaped an engineering decision: the
explainability module (`ml/src/explain.py`) originally assumed the winning model
would always be tree-based, which broke the first time the tie flipped toward
Logistic Regression during testing. I fixed this by making the SHAP explainer
selection type-aware at runtime rather than hardcoded, and by making
`GET /model/info` and the Model Insights page report whichever model is actually
loaded, rather than hardcoding a single number in documentation that could
silently become wrong on a different machine.

## 5. Results

On the held-out test set, the final model achieves roughly 74–78% accuracy and
approximately 0.70 macro-F1, against a majority-class baseline of 50% accuracy
and 0.22 macro-F1 — a substantial improvement over the naive baseline, but far
from perfect. Per-class performance is uneven: "Enrolled" is consistently the
weakest class (recall in the 55–65% range across model candidates), which
matches the intuition that these are the "not yet resolved" students whose
eventual trajectory the model has the least signal about. "Graduate" and
"Dropout" are both predicted meaningfully better. Exact current numbers are
always available live via `GET /model/info` and the Model Insights page rather
than restated here, for the reproducibility reason described in Section 4.

## 6. Engineering Reflection

The most important structural decision in this project was treating
`ml/src/features.py` as the single source of truth for feature types, human-
readable labels, and the mapping between clean API field names and the dataset's
raw column names. Every other module — training, the Pydantic request schema,
the explainability aggregation, the sample-roster generator — imports from this
file rather than redefining any of it. This is what the brief calls out directly
in section 6.2 as the most common integration failure mode ("preprocessing the
web form differently from the training notebook"), and structuring the project
around a single shared definition made that class of bug structurally
impossible rather than something to remember to avoid.

The frontend follows the same principle: a single `api/client.js` module is the
only place any component is allowed to call `fetch()`, and the Student Detail
and New Prediction pages call the live `/predict` and `/explain` endpoints on
every load and submit rather than displaying precomputed values — verified
directly by editing an input and confirming the returned Dropout probability
changed substantially (19% → 93% when second-semester approved units was
changed to 0).

## 7. Limitations & Responsible Use

- Test performance (~0.70 macro-F1) reflects a genuinely useful but imperfect
  signal — not a certainty, and not something that should be presented to a
  student as a verdict.
- The model is trained on one country's dataset from one historical period. Its
  accuracy is not guaranteed to transfer to a different institution's population.
- SHAP-based feature importance, when aggregated from one-hot encoded columns
  back to original features, can be inflated for high-cardinality categorical
  features (e.g., a 46-category occupation code has more encoded columns to
  accumulate weight across than a single binary field). This is documented in
  the app itself, not just in this report.
- Consistent with the brief's responsible-ML requirements, EduPulse is built and
  documented throughout — in the UI, the README, and here — as a decision-support
  signal for a human advisor, explicitly not a system authorized to automate
  academic consequences for any student.

## 8. What I Would Do With More Time

- Add `CalibratedClassifierCV` so that the displayed probabilities are closer to
  literal confidence levels rather than raw model scores.
- Expand cross-institution validation — the current evaluation only speaks to
  this one dataset's population.
- Add the optional what-if simulator and MLflow experiment tracking called out
  as P1/stretch items in the brief, once the required P0 path was fully solid.
