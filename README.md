# Feature Extraction and Price Prediction for Mobile Phones

Predicts mobile phone prices from specs and ranks which features matter most.

## What it does
1. Loads your CSV/XLSX and explores it (shape, missing values, price distribution)
2. Cleans it: parses "4 GB", "5000 mAh", "48MP + 2MP", "Rs. 9,999", handles missing values and outliers, one-hot encodes brand/colour/processor
3. Extracts features: Spearman correlation, mutual information, Random Forest importance
4. Trains Linear Regression, Decision Tree, Random Forest, Gradient Boosting
5. Evaluates with MAE, RMSE, R2 and cross-validation
6. Confirms feature importance with permutation importance
7. Writes charts, a Markdown report and recommendations

## Machine requirements
| Item | Requirement |
|---|---|
| OS | Windows 10/11, macOS, or Linux |
| Python | 3.9 or newer (3.10 - 3.12 recommended) |
| RAM | 4 GB is enough (8 GB comfortable) |
| CPU / GPU | Any modern CPU, no GPU needed |
| Disk | About 500 MB for Python packages, plus a few MB for outputs |
| Internet | Only to install packages the first time |

Python packages (in `requirements.txt`): numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, openpyxl.

## Setup
```bash
# 1. (optional but recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. install packages
pip install -r requirements.txt
```

## Run
```bash
# with your own data
python price_prediction.py --data your_file.csv

# optional settings
python price_prediction.py --data your_file.csv --out results --test-size 0.25 --min-cat 5
python price_prediction.py --data your_file.csv --encode-model   # also one-hot the full Model column

# quick self-test with synthetic (fake) data
python make_sample_data.py
python price_prediction.py --data sample_mobiles.csv
```

Your data should have a column whose name contains "price", "prize" or "cost". Other columns are detected by name
(model, colour, memory, RAM, battery, rear camera, front camera, AI lens, mobile height, processor);
exact spelling and extra columns are fine.

## Outputs (in `outputs/`)
- `report.md` - full report with tables, charts and recommendations
- `feature_importance.csv`, `model_metrics.csv`, `cleaned_data.csv`
- `01_...png` to `07_...png` - visualizations
- `best_model.joblib` - trained model:
  ```python
  import joblib; bundle = joblib.load("outputs/best_model.joblib")
  bundle["model"].predict(new_df[bundle["columns"]])   # new_df encoded the same way as cleaned data
  ```

## Notes
- Prices are modelled on a log scale (they are right-skewed); reported errors are in your original currency.
- The full Model name is not one-hot encoded by default because it is nearly unique per row and causes overfitting; the brand is extracted from it instead. Use `--encode-model` to include it.
- If a Model column exists, the train/test split and cross-validation are grouped by model, so colour variants of the same phone never appear on both sides (this avoids inflated scores).
- Importance for categorical features (brand, processor, colour) is the sum over their one-hot columns, so they can look larger than a single numeric feature.
- Importance reflects association in your data, not causation.
- To convert the report to PDF/Word: open `report.md` in VS Code or Typora, or run `pandoc report.md -o report.pdf`.
