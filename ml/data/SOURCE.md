# Dataset Source

**Name:** Predict Students' Dropout and Academic Success
**Origin:** UCI Machine Learning Repository
**DOI:** 10.24432/C5MC89
**Records:** 4,424 | **Features:** 36 | **Target classes:** Dropout, Enrolled, Graduate (3-class, imbalanced)

## Retrieval note
The sandboxed environment used to build this project could not reach `archive.ics.uci.edu`
directly (network egress allowlist). The identical dataset (verified by row count, column
schema, and class distribution against the official UCI listing) was retrieved from a
public GitHub mirror used in a University of British Columbia Master of Data Science
course repository:

- Mirror: https://raw.githubusercontent.com/caesarw0/ml-dataset/main/students_dropout_prediction/data.csv
- Referenced from: https://github.com/UBC-MDS/dropout-predictions

## Verification performed
- Shape: (4424, 37) — 36 features + Target ✅
- Target value counts: Graduate 2209, Dropout 1421, Enrolled 794 ✅ (matches UCI-reported imbalance)
- Missing values: 0 ✅ (matches UCI dataset documentation)
- Column names match the official UCI feature list ✅

File saved as `student_dropout_raw.csv` (semicolon-delimited source converted to standard
comma-delimited CSV, column names whitespace-stripped).
