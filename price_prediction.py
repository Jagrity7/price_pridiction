#!/usr/bin/env python3
"""
Feature Extraction and Price Prediction for Mobile Phones
=========================================================

End-to-end pipeline:
  1. Data exploration
  2. Preprocessing (cleaning, unit parsing, outliers, encoding)
  3. Feature extraction (correlation, mutual information, tree importance)
  4. Model building (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
  5. Evaluation (MAE, RMSE, R2, cross-validation)
  6. Feature-importance analysis (impurity + permutation importance)
  7. Report + visualizations + recommendations

Usage:
    python price_prediction.py --data mobiles.csv
    python price_prediction.py --data mobiles.csv --out results --test-size 0.25

The script auto-detects columns by name (model, colour, memory, ram, battery,
rear camera, front camera, ai lens, mobile height, processor, price), so exact
header spelling does not matter much. Values such as "4 GB", "5000 mAh",
"48MP + 2MP" or "Rs. 9,999" are parsed automatically.
"""
import argparse
import re
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def norm(col):
    return re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()


def first_num(x):
    if pd.isna(x):
        return np.nan
    m = NUM_RE.search(str(x).replace(",", ""))
    return float(m.group()) if m else np.nan


def to_gb(x):
    """'128 GB' -> 128, '1 TB' -> 1024, '512 MB' -> 0.5. Bare numbers are GB."""
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb|kb)?", str(x).replace(",", "").lower())
    if not m:
        return np.nan
    unit = m.group(2) or "gb"
    return float(m.group(1)) * {"tb": 1024, "gb": 1, "mb": 1 / 1024, "kb": 1 / 1024**2}[unit]


def parse_price(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    # Take the first number after removing thousands separators, so that
    # "Rs. 12,345", "₹9,999" and "INR 1,24,999.00" are all read correctly.
    m = NUM_RE.search(str(x).replace(",", ""))
    return float(m.group()) if m else np.nan


def to_binary(x):
    """Yes/No style columns. Missing is treated as 'feature absent'."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1", "available", "present", "ai lens", "ai"}:
        return 1
    if s in {"no", "n", "false", "0", "none", "not available", "absent", "", "nan"}:
        return 0
    return 1 if "ai" in s or "yes" in s else 0


def camera_features(series, prefix):
    lists = series.map(
        lambda s: [v for v in (float(n) for n in NUM_RE.findall(str(s).replace(",", ""))) if v > 0]
        if pd.notna(s)
        else []
    )
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_main_mp"] = lists.map(lambda l: l[0] if l else np.nan)
    out[f"{prefix}_lens_count"] = lists.map(lambda l: float(len(l)) if l else np.nan)
    out[f"{prefix}_total_mp"] = lists.map(lambda l: float(sum(l)) if l else np.nan)
    return out


def group_rare(series, min_count):
    s = series.astype(str).str.strip().replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
    counts = s.value_counts()
    keep = counts[counts >= min_count].index
    return s.where(s.isin(keep), "Other")


def detect_role(col, series):
    n = norm(col)
    if "cam" in n and any(k in n for k in ("rear", "back", "primary")):
        return "rear_cam"
    if "cam" in n and any(k in n for k in ("front", "selfie")):
        return "front_cam"
    if re.search(r"\bram\b", n):
        return "ram"
    if re.search(r"memory|storage|\brom\b|internal", n):
        return "memory"
    if "batter" in n:
        return "numeric"
    if "height" in n:
        return "numeric"
    if re.search(r"\bai\b", n):
        return "binary"
    if re.search(r"model|name|brand", n):
        return "brand"
    if "colo" in n:
        return "categorical"
    if re.search(r"process|chipset|cpu", n):
        return "categorical"
    # Unknown column: numeric if most values contain numbers, else categorical
    parsed = series.map(first_num)
    if series.dtype != object or parsed.notna().mean() >= 0.8:
        return "numeric"
    return "categorical" if series.nunique() <= 60 else "drop"


# --------------------------------------------------------------------------
# Data loading / preprocessing
# --------------------------------------------------------------------------
def load_data(path):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_target(df):
    for c in df.columns:
        if re.search(r"pric|prize|cost", norm(c)):
            return c
    raise SystemExit("Could not find a price column (a column name containing 'price', 'prize' or 'cost').")


def build_features(df, target_col, min_cat, encode_model, log):
    """Returns clean_df, X, y, source_map (encoded column -> original feature)."""
    df = df.copy()
    y = df[target_col].map(parse_price)
    feats = pd.DataFrame(index=df.index)
    cats = {}
    source = {}

    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        role = detect_role(col, s)
        log(f"  column '{col}' -> {role}")
        if role == "rear_cam" or role == "front_cam":
            cf = camera_features(s, "rear" if role == "rear_cam" else "front")
            for c in cf.columns:
                feats[c] = cf[c]
                source[c] = col
        elif role == "ram":
            feats["ram_gb"] = s.map(to_gb)
            source["ram_gb"] = col
        elif role == "memory":
            feats["memory_gb"] = s.map(to_gb)
            source["memory_gb"] = col
        elif role == "numeric":
            name = norm(col).replace(" ", "_")
            feats[name] = s.map(first_num)
            source[name] = col
        elif role == "binary":
            name = norm(col).replace(" ", "_")
            feats[name] = s.map(to_binary)
            source[name] = col
        elif role == "brand":
            brand = s.astype(str).str.strip().str.split().str[0].str.title()
            cats["brand"] = (group_rare(brand, min_cat), col)
            if encode_model:
                cats[norm(col).replace(" ", "_")] = (group_rare(s, min_cat), col)
        elif role == "categorical":
            # processors etc.: strip stray whitespace, group rare values
            cats[norm(col).replace(" ", "_")] = (group_rare(s, min_cat), col)
        else:
            log(f"  (dropping '{col}': high-cardinality text)")

    clean = feats.copy()
    dummies = []
    for name, (series, orig) in cats.items():
        clean[name] = series
        d = pd.get_dummies(series, prefix=name, dtype=int)
        dummies.append(d)
        for c in d.columns:
            source[c] = name
    X = pd.concat([feats] + dummies, axis=1)

    ok = y.notna() & (y > 0)
    dropped = (~ok).sum()
    if dropped:
        log(f"  dropped {dropped} rows with missing/non-positive price")
    clean["price"] = y
    return clean[ok], X[ok], y[ok], source


# --------------------------------------------------------------------------
# Plot / report helpers
# --------------------------------------------------------------------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def md_table(df, floatfmt="{:,.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def aggregate(values, columns, source, normalize=True):
    s = pd.Series(np.clip(values, 0, None), index=columns)
    g = s.groupby(lambda c: source[c]).sum()
    return g / g.sum() * 100 if normalize and g.sum() > 0 else g


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Mobile phone price prediction")
    ap.add_argument("--data", required=True, help="path to CSV/XLSX dataset")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-cat", type=int, default=10,
                    help="categories rarer than this are grouped as 'Other'")
    ap.add_argument("--encode-model", action="store_true",
                    help="also one-hot encode the full Model column (not just the brand)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_lines = []
    log = print

    # 1. Exploration ---------------------------------------------------------
    print("[1/6] Loading and exploring data")
    raw = load_data(args.data)
    print(f"  shape: {raw.shape}")
    target_col = find_target(raw)
    n_dupes = int(raw.duplicated().sum())
    raw = raw.drop_duplicates()
    missing = raw.isna().sum()
    missing = missing[missing > 0]

    # 2. Preprocessing -------------------------------------------------------
    print("[2/6] Preprocessing")
    clean, X, y, source = build_features(raw, target_col, args.min_cat, args.encode_model, log)
    clean.to_csv(out / "cleaned_data.csv", index=False)
    n_rows = len(X)
    print(f"  {n_rows} rows, {X.shape[1]} encoded features")

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(y, bins=40, kde=True)
    plt.title("Price distribution")
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(y), bins=40, kde=True, color="darkorange")
    plt.title("log(1 + price) distribution")
    savefig(out / "01_price_distribution.png")

    # Colour variants of one phone share specs and price. Splitting by model keeps all
    # variants of a phone on the same side, so test phones are genuinely unseen.
    model_col = next((c for c in raw.columns if re.search(r"model|name", norm(c))), None)
    groups = raw.loc[X.index, model_col].astype(str).str.strip().str.lower() if model_col else None
    if groups is not None and groups.nunique() >= 10:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                        random_state=RANDOM_STATE).split(X, y, groups))
        X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
        y_train, y_test = y.iloc[tr], y.iloc[te]
        groups_train = groups.iloc[tr]
        split_note = f"grouped by `{model_col}` so every colour variant of a phone is on one side"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=RANDOM_STATE
        )
        groups_train = None
        split_note = "random split"
    print(f"  split: {split_note}; train={len(X_train)}, test={len(X_test)}")
    # Winsorize continuous features using training-set percentiles (no leakage)
    continuous = [c for c in X.columns if X[c].nunique() > 2]
    for c in continuous:
        lo, hi = X_train[c].quantile([0.01, 0.99])
        X_train[c] = X_train[c].clip(lo, hi)
        X_test[c] = X_test[c].clip(lo, hi)

    # 3. Feature extraction --------------------------------------------------
    print("[3/6] Feature extraction")
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)

    spearman = Xtr_imp.corrwith(y_train, method="spearman").dropna()
    top_corr = spearman.reindex(spearman.abs().sort_values(ascending=False).index).head(15)
    plt.figure(figsize=(8, 5))
    colors = ["seagreen" if v > 0 else "indianred" for v in top_corr.values[::-1]]
    plt.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors)
    plt.title("Top features by Spearman correlation with price")
    plt.xlabel("Spearman correlation")
    savefig(out / "02_correlation_with_price.png")

    num_cols = [c for c in X.columns if X[c].nunique() > 2]
    if len(num_cols) >= 2:
        plt.figure(figsize=(8, 6.5))
        corr = pd.concat([Xtr_imp[num_cols], y_train.rename("price")], axis=1).corr(method="spearman")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
        plt.title("Correlation heatmap (numeric features)")
        savefig(out / "03_correlation_heatmap.png")

    mi = mutual_info_regression(Xtr_imp, np.log1p(y_train), random_state=RANDOM_STATE)
    mi_group = aggregate(mi, X.columns, source)

    # 4. Model building ------------------------------------------------------
    print("[4/6] Training models")

    def wrap(model, scale=False):
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("sc", StandardScaler()))
        steps.append(("m", model))
        # Train on log(price): prices are right-skewed; predictions come back in original units
        return TransformedTargetRegressor(Pipeline(steps), func=np.log1p, inverse_func=np.expm1)

    models = {
        "Linear Regression": wrap(LinearRegression(), scale=True),
        "Decision Tree": wrap(DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, random_state=RANDOM_STATE)),
        "Random Forest": wrap(RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                    n_jobs=-1, random_state=RANDOM_STATE)),
        "Gradient Boosting": wrap(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                            max_depth=3, subsample=0.8,
                                                            random_state=RANDOM_STATE)),
    }
    if groups_train is not None:
        folds = GroupKFold(n_splits=min(5, groups_train.nunique()))
    else:
        folds = KFold(n_splits=min(5, max(2, len(X_train) // 20)), shuffle=True, random_state=RANDOM_STATE)

    # 5. Evaluation ----------------------------------------------------------
    print("[5/6] Evaluating")
    rows, preds = [], {}
    for name, model in models.items():
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=folds, groups=groups_train,
                                   scoring="neg_root_mean_squared_error").mean()
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, p),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
            "R2": r2_score(y_test, p),
            "CV RMSE (train)": cv_rmse,
        })
        print(f"  {name:<18} MAE={rows[-1]['MAE']:,.0f}  RMSE={rows[-1]['RMSE']:,.0f}  R2={rows[-1]['R2']:.3f}")
    metrics = pd.DataFrame(rows).sort_values("CV RMSE (train)").reset_index(drop=True)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    best_name = metrics.loc[0, "Model"]
    best = models[best_name]

        
    # Save actual vs predicted prices for demonstration
    demo = pd.DataFrame({
        "Actual_Price": y_test,
        "Predicted_Price": preds[best_name],
    })

    demo["Absolute_Error"] = (
        demo["Actual_Price"] - demo["Predicted_Price"]
    ).abs()

    demo.to_csv(out / "prediction_demo.csv", index=False)

    print("\nSample price predictions:")
    print(demo.head(10).round(2).to_string(index=False))

    joblib.dump({"model": best, "columns": list(X.columns)}, out / "best_model.joblib")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    m2 = metrics.set_index("Model")[["MAE", "RMSE"]]
    m2.plot.bar(ax=ax[0], rot=20)
    ax[0].set_title("Test error by model (lower is better)")
    ax[1].scatter(y_test, preds[best_name], alpha=0.5, s=18)
    lim = [min(y_test.min(), preds[best_name].min()), max(y_test.max(), preds[best_name].max())]
    ax[1].plot(lim, lim, "r--", lw=1)
    ax[1].set_xlabel("Actual price")
    ax[1].set_ylabel("Predicted price")
    ax[1].set_title(f"Actual vs predicted - {best_name}")
    savefig(out / "04_model_performance.png")

    # 6. Feature importance --------------------------------------------------
    print("[6/6] Feature importance")
    rf = models["Random Forest"].regressor_.named_steps["m"]
    rf_group = aggregate(rf.feature_importances_, X.columns, source)
    perm = permutation_importance(best, X_test, y_test, n_repeats=15, random_state=RANDOM_STATE,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
    perm_group = aggregate(perm.importances_mean, X.columns, source)

    imp = pd.DataFrame({
        "mutual_info_%": mi_group,
        "random_forest_%": rf_group,
        f"permutation_{best_name}_%": perm_group,
    }).fillna(0)
    imp["average_%"] = imp.mean(axis=1)
    imp = imp.sort_values("average_%", ascending=False)
    imp.index.name = "feature"
    imp.to_csv(out / "feature_importance.csv")

    top = imp.head(12)
    plt.figure(figsize=(9, 5.5))
    top.drop(columns="average_%").iloc[::-1].plot.barh(ax=plt.gca(), width=0.8)
    plt.title("Feature importance by method (% share, grouped per original feature)")
    plt.xlabel("Importance (%)")
    savefig(out / "05_feature_importance.png")

    # Price vs key drivers
    drivers = [c for c in ("ram_gb", "memory_gb") if c in X.columns]
    if drivers:
        fig, axes = plt.subplots(1, len(drivers), figsize=(6 * len(drivers), 4.2))
        axes = np.atleast_1d(axes)
        for a, c in zip(axes, drivers):
            sns.boxplot(x=clean[c], y=clean["price"], ax=a)
            a.set_title(f"Price by {c}")
        savefig(out / "06_price_vs_ram_memory.png")
    if "brand" in clean:
        top_brands = clean["brand"].value_counts().head(8).index
        plt.figure(figsize=(9, 4.5))
        sns.boxplot(data=clean[clean["brand"].isin(top_brands)], x="brand", y="price")
        plt.xticks(rotation=25)
        plt.title("Price by brand (top 8 by count)")
        savefig(out / "07_price_by_brand.png")

    # ------------------------------------------------------------------ report
    def direction(group):
        cols = [c for c, s in source.items() if s == group and c in spearman.index]
        cat = group in clean.columns and not pd.api.types.is_numeric_dtype(clean[group])
        if cat:
            med = clean.groupby(group)["price"].median().sort_values(ascending=False)
            return (f"varies by category - highest median price: **{med.index[0]}** ({med.iloc[0]:,.0f}); "
                    f"lowest: **{med.index[-1]}** ({med.iloc[-1]:,.0f})")
        if cols:
            r = spearman[cols].loc[spearman[cols].abs().idxmax()]
            return f"{'higher' if r > 0 else 'lower'} values go with higher prices (Spearman {r:+.2f})"
        return "see charts"

    top5 = list(imp.index[:5])
    R = report_lines.append
    R("# Mobile Phone Price Prediction - Report\n")
    R("## 1. Data exploration")
    R(f"- Rows after removing {n_dupes} duplicate(s) and invalid prices: **{n_rows}**")
    R(f"- Original columns: {', '.join(raw.columns)}")
    R(f"- Target column: `{target_col}`  |  min {y.min():,.0f}, median {y.median():,.0f}, "
      f"mean {y.mean():,.0f}, max {y.max():,.0f}")
    R(f"- Columns with missing values: {', '.join(f'{k} ({v})' for k, v in missing.items()) or 'none'}")
    R("- Price is right-skewed, so models are trained on log(price) and predictions are converted back.\n")
    R("![Price distribution](01_price_distribution.png)\n")
    R("## 2. Preprocessing")
    R("- Text values were converted to numbers (GB/TB -> GB, mAh, inches, MP). Camera strings such as "
      "`48MP + 2MP` became main MP, lens count and total MP.")
    R("- AI lens became 0/1. Brand was extracted from the model name; brand, colour and processor were "
      f"one-hot encoded (categories with fewer than {args.min_cat} phones grouped as `Other`).")
    R("- Missing numeric values were median-imputed inside the model pipeline. Continuous features were "
      "clipped to the 1st-99th percentile of the training set to limit outliers.")
    R(f"- Train/test split: {int((1 - args.test_size) * 100)}% / {int(args.test_size * 100)}% ({split_note}), "
      f"random_state={RANDOM_STATE}. Cross-validation uses the same grouping.\n")
    R("## 3. Feature extraction")
    R("Three views were combined: Spearman correlation, mutual information, and tree-based importance.\n")
    R("![Correlation](02_correlation_with_price.png)\n")
    if (out / "03_correlation_heatmap.png").exists():
        R("![Heatmap](03_correlation_heatmap.png)\n")
    R("## 4. Models and evaluation")
    R("Metrics are on the held-out test set, in the same currency units as your price column.\n")
    R(md_table(metrics, "{:,.3f}") + "\n")
    R(f"Best model by cross-validated RMSE: **{best_name}**.\n")
    R("![Performance](04_model_performance.png)\n")
    R("## 5. Feature importance")
    R(md_table(imp.reset_index().head(12), "{:,.1f}") + "\n")
    R("![Importance](05_feature_importance.png)\n")
    R("## 6. Recommendations")
    for i, f in enumerate(top5, 1):
        R(f"{i}. **{f}** - {direction(f)}.")
    R("\n- Use the top features as the backbone of price tiers: set price bands by those specs first, "
      "then adjust for brand positioning.")
    R("- Features near the bottom of the ranking add little pricing power; avoid charging a premium for them "
      "and avoid over-emphasising them in marketing.")
    R("- Use the saved model (`best_model.joblib`) to sanity-check new listings: a big gap between predicted "
      "and planned price flags a potential under- or over-pricing.")
    R("- Caveat: importance shows association, not causation, and depends on the dataset size and coverage.")
    (out / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nDone. Best model: {best_name}")
    print(f"Top features: {', '.join(top5)}")
    print(f"Everything is in: {out.resolve()}")


if __name__ == "__main__":
    main()
