pip install -r requirements.txt# EduPulse — Local Setup (VS Code)

## 1. Open the project
Unzip `edupulse.zip`, then in VS Code: `File → Open Folder...` and select the
`edupulse/` folder. Install the **Python extension** (ms-python.python) if you
don't have it — VS Code will prompt you.

## 2. Create a virtual environment
Open a terminal inside VS Code (`` Ctrl+` ``) and run:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, select this environment as your interpreter:
`Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv`.

## 3. Run the ML pipeline yourself
This reproduces everything from scratch — good way to confirm you understand
the flow, not just trust that it ran once for me.

```bash
cd ml/src
python data.py        # prints the train/val/test split + class balance
python train.py        # trains & compares 3 models, saves the artifact
python explain.py      # runs SHAP on one example row
```

Watch for:
- `data.py` — should print identical class proportions across train/val/test
  (that's what "stratified" split means).
- `train.py` — prints CV scores for all 3 models, then the final test-set
  confusion matrix. Compare the numbers to what's in `ml/reports/model_comparison_report.json`.
- `explain.py` — prints the top 5 factors behind one prediction.

## 4. Project structure so far
```
edupulse/
├── ml/
│   ├── data/
│   │   ├── student_dropout_raw.csv   ← the dataset
│   │   └── SOURCE.md                  ← where it came from, how verified
│   ├── src/
│   │   ├── features.py    ← feature type definitions (numeric/categorical/binary)
│   │   ├── data.py        ← loading + train/val/test split
│   │   ├── preprocessing.py ← the ColumnTransformer (scaling + one-hot)
│   │   ├── train.py       ← trains & compares LogReg / RandomForest / XGBoost
│   │   └── explain.py     ← SHAP explainability
│   ├── artifacts/
│   │   └── model_pipeline.joblib  ← the trained, serialized model
│   └── reports/
│       └── model_comparison_report.json ← all metrics, CV scores, rationale
├── backend/   ← empty so far, building tomorrow
├── frontend/  ← empty so far, building tomorrow
└── requirements.txt
```

## 5. If something errors
Most likely cause: package version mismatch. Run `pip install -r requirements.txt --upgrade`
inside the activated venv. If SHAP fails to install (it sometimes needs build
tools), tell me the exact error and I'll adjust.
