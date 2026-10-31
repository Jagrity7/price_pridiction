#!/usr/bin/env python3
"""Interactive single-phone price prediction using the existing training script."""
import importlib.util
from pathlib import Path

import joblib
import pandas as pd

# Keep this file in the same folder as price_prediction.py and Processed_Flipdata.xlsx.
TRAIN_SCRIPT = Path(__file__).with_name("price_prediction.py")
DATA_FILE = Path(__file__).with_name("Processed_Flipdata.xlsx")
MODEL_FILE = Path(__file__).with_name("outputs") / "best_model.joblib"


def load_training_module():
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_SCRIPT.name}. Rename your existing "
            "'price_prediction(1).py' to 'price_prediction.py' and keep it here."
        )
    spec = importlib.util.spec_from_file_location("price_prediction_module", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else (default if default is not None else "")


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. First run:\n"
            "python price_prediction.py --data Processed_Flipdata.xlsx"
        )

    module = load_training_module()
    raw = module.load_data(DATA_FILE)
    target_col = module.find_target(raw)
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    model_columns = bundle["columns"]

    print("\n=== Mobile Phone Price Predictor ===")
    print("Enter the phone specifications below.")
    print("Press Enter to use the example value shown in brackets.\n")

    phone = {
        "Model": ask("Phone model / brand", "Infinix SMART 7"),
        "Colour": ask("Colour", "Night Black"),
        "Memory": ask("Storage (e.g. 64 GB, 128 GB)", "64 GB"),
        "RAM": ask("RAM (e.g. 4 GB, 8 GB)", "4 GB"),
        "Battery_": ask("Battery capacity in mAh", "6000"),
        "Rear Camera": ask("Rear camera (e.g. 50MP + 2MP)", "13MP"),
        "Front Camera": ask("Front camera (e.g. 16MP)", "5MP"),
        "AI Lens": ask("AI Lens? (1 for yes, 0 for no)", "1"),
        "Mobile Height": ask("Mobile height (use the same units as the dataset)", "16.76"),
        "Processor_": ask("Processor / chipset", "Unisoc Spreadtrum SC9863A1"),
    }

    # Build a row with the same source columns as the dataset.
    # A placeholder target is needed only because the existing feature-builder
    # filters rows with missing target values; the target is NOT model input.
    phone[target_col] = float(raw[target_col].map(module.parse_price).median())
    for col in raw.columns:
        if col not in phone:
            phone[col] = raw[col].iloc[0]

    phone_df = pd.DataFrame([phone], columns=raw.columns)
    phone_df.index = [max(raw.index.max() + 1, 1_000_000)]
    combined = pd.concat([raw, phone_df], ignore_index=False)

    _, all_X, _, _ = module.build_features(
        combined, target_col, min_cat=8, encode_model=False, log=lambda *_: None
    )
    new_X = all_X.loc[phone_df.index].reindex(columns=model_columns, fill_value=0)

    predicted_price = float(model.predict(new_X)[0])
    print("\n----------------------------------------")
    print(f"Phone: {phone['Model']}")
    print(f"Predicted price: ₹{predicted_price:,.0f}")
    print("----------------------------------------")
    print("This is a model estimate based on the dataset—not a live market price.")
    print("Accuracy depends on whether the dataset contains comparable phones/specifications.")


if __name__ == "__main__":
    main()
