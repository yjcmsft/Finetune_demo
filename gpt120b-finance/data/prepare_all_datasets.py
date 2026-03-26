#!/usr/bin/env python
"""
Download and convert five fraud-detection datasets into unified JSONL
for fine-tuning GPT-OSS 120B and 20B models.

Datasets
--------
1. ULB Credit Card Fraud (2013)           – Kaggle: mlg-ulb/creditcardfraud
2. IEEE-CIS Fraud Detection (2019)        – Kaggle: ieee-fraud-detection
3. Fraudulent E-Commerce Transactions     – Kaggle: shriyashjagtap/fraudulent-e-commerce-transactions
4. PaySim Mobile Money Transfers           – Kaggle: ealaxi/paysim1
5. Sparkov Credit Card Transactions        – already in repo (fraudTrain.csv / fraudTest.csv)

Prerequisites
-------------
    pip install kaggle pandas scikit-learn
    export KAGGLE_USERNAME=<your-username>
    export KAGGLE_KEY=<your-api-key>

Usage
-----
    python prepare_all_datasets.py          # download + convert all
    python prepare_all_datasets.py --skip-download   # convert only (CSVs already present)
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
OUT_TRAIN = DATA_DIR / "training.jsonl"
OUT_VAL = DATA_DIR / "validation.jsonl"
SYSTEM_MSG = (
    "You are a fraud detection assistant. Analyze the transaction details "
    "and determine whether the transaction is fraudulent."
)
SEED = 42
VAL_RATIO = 0.10  # 10 % held-out for validation


# ---------------------------------------------------------------------------
# Kaggle helpers
# ---------------------------------------------------------------------------
def _kaggle_download(dataset_slug: str, dest: Path, competition: bool = False):
    """Download a Kaggle dataset/competition zip into *dest* and unzip."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle"]
    if competition:
        cmd += ["competitions", "download", "-c", dataset_slug, "-p", str(dest)]
    else:
        cmd += ["datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"]
    print(f"  -> {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # competitions come as a zip; unzip manually
    if competition:
        import zipfile
        for zf in dest.glob("*.zip"):
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dest)
            zf.unlink()


def download_all():
    """Download all five datasets into raw/<name>/."""
    print("\n=== Downloading datasets ===\n")

    # 1. ULB Credit Card Fraud
    _kaggle_download("mlg-ulb/creditcardfraud", RAW_DIR / "ulb_creditcard")

    # 2. IEEE-CIS Fraud Detection
    _kaggle_download("ieee-fraud-detection", RAW_DIR / "ieee_cis", competition=True)

    # 3. Fraudulent E-Commerce Transactions
    _kaggle_download(
        "shriyashjagtap/fraudulent-e-commerce-transactions",
        RAW_DIR / "ecommerce_fraud",
    )

    # 4. PaySim Mobile Money
    _kaggle_download("ealaxi/paysim1", RAW_DIR / "paysim")

    # 5. Sparkov – already in repo
    print("  -> Sparkov: using existing fraudTrain.csv / fraudTest.csv")


# ---------------------------------------------------------------------------
# Per-dataset converters  ->  list[dict]  each dict = one JSONL record
# ---------------------------------------------------------------------------

def _make_record(prompt_text: str, is_fraud: bool) -> dict:
    """Return a single {"prompt": ..., "completion": ...} dict."""
    if is_fraud:
        completion = (
            "Yes, this transaction appears to be fraudulent. "
            "The transaction pattern shows characteristics commonly associated with fraud."
        )
    else:
        completion = (
            "No, this transaction appears to be legitimate. "
            "The transaction pattern is consistent with normal purchasing behavior."
        )
    return {"prompt": prompt_text, "completion": completion}


# ---- 1. ULB Credit Card Fraud -------------------------------------------
def convert_ulb() -> list[dict]:
    csv_path = RAW_DIR / "ulb_creditcard" / "creditcard.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found"); return []
    print("  Converting ULB Credit Card Fraud …")
    df = pd.read_csv(csv_path)
    records = []
    for _, r in df.iterrows():
        prompt = (
            "Analyze this credit-card transaction for potential fraud:\n"
            f"- Time (seconds from first txn): {r['Time']}\n"
            f"- Amount: ${r['Amount']:.2f}\n"
            f"- PCA features V1–V28: "
            + ", ".join(f"V{i}={r[f'V{i}']:.4f}" for i in range(1, 29))
            + "\n\nIs this transaction fraudulent?"
        )
        records.append(_make_record(prompt, bool(r["Class"])))
    return records


# ---- 2. IEEE-CIS Fraud Detection ----------------------------------------
def convert_ieee_cis() -> list[dict]:
    txn_path = RAW_DIR / "ieee_cis" / "train_transaction.csv"
    if not txn_path.exists():
        print(f"  [SKIP] {txn_path} not found"); return []
    print("  Converting IEEE-CIS Fraud Detection …")
    df = pd.read_csv(txn_path, usecols=[
        "TransactionDT", "TransactionAmt", "ProductCD",
        "card4", "card6", "addr1", "addr2",
        "P_emaildomain", "R_emaildomain", "isFraud",
    ])
    df = df.fillna("N/A")
    records = []
    for _, r in df.iterrows():
        prompt = (
            "Analyze this e-commerce transaction for potential fraud:\n"
            f"- Transaction time delta: {r['TransactionDT']}\n"
            f"- Amount: ${r['TransactionAmt']:.2f}\n"
            f"- Product code: {r['ProductCD']}\n"
            f"- Card network: {r['card4']}\n"
            f"- Card type: {r['card6']}\n"
            f"- Billing address: {r['addr1']}, {r['addr2']}\n"
            f"- Purchaser email domain: {r['P_emaildomain']}\n"
            f"- Recipient email domain: {r['R_emaildomain']}\n"
            "\nIs this transaction fraudulent?"
        )
        records.append(_make_record(prompt, bool(r["isFraud"])))
    return records


# ---- 3. Fraudulent E-Commerce Transactions (Synthetic) ------------------
def convert_ecommerce() -> list[dict]:
    # The dataset may have different file names; try common ones
    candidates = [
        RAW_DIR / "ecommerce_fraud" / "Fraudulent_E-Commerce_Transaction_Data.csv",
        RAW_DIR / "ecommerce_fraud" / "fraudulent_ecommerce.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c; break
    # fallback: pick first CSV in directory
    if csv_path is None:
        csvs = list((RAW_DIR / "ecommerce_fraud").glob("*.csv"))
        if csvs:
            csv_path = csvs[0]
    if csv_path is None:
        print("  [SKIP] E-Commerce fraud CSV not found"); return []

    print(f"  Converting E-Commerce Fraud ({csv_path.name}) …")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Determine fraud label column
    fraud_col = None
    for candidate_col in ["Is Fraudulent", "is_fraud", "isFraud", "fraud", "label", "Class"]:
        if candidate_col in df.columns:
            fraud_col = candidate_col; break
    if fraud_col is None:
        print(f"  [SKIP] Cannot find fraud label column in {df.columns.tolist()}")
        return []

    records = []
    for _, r in df.iterrows():
        parts = ["Analyze this e-commerce transaction for potential fraud:"]
        for col in df.columns:
            if col == fraud_col:
                continue
            parts.append(f"- {col}: {r[col]}")
        parts.append("\nIs this transaction fraudulent?")
        prompt = "\n".join(parts)
        label = r[fraud_col]
        is_fraud = str(label).strip().lower() in ("1", "true", "yes")
        records.append(_make_record(prompt, is_fraud))
    return records


# ---- 4. PaySim Mobile Money Transfers -----------------------------------
def convert_paysim() -> list[dict]:
    csv_path = RAW_DIR / "paysim" / "PS_20174392719_1491204016305_log.csv"
    if not csv_path.exists():
        # try alternate name
        csvs = list((RAW_DIR / "paysim").glob("*.csv"))
        csv_path = csvs[0] if csvs else None
    if csv_path is None or not csv_path.exists():
        print("  [SKIP] PaySim CSV not found"); return []
    print("  Converting PaySim …")
    df = pd.read_csv(csv_path)
    records = []
    for _, r in df.iterrows():
        prompt = (
            "Analyze this mobile-money transfer for potential fraud:\n"
            f"- Step (hour): {r['step']}\n"
            f"- Type: {r['type']}\n"
            f"- Amount: ${r['amount']:.2f}\n"
            f"- Origin account balance before: ${r['oldbalanceOrg']:.2f}\n"
            f"- Origin account balance after: ${r['newbalanceOrig']:.2f}\n"
            f"- Destination balance before: ${r['oldbalanceDest']:.2f}\n"
            f"- Destination balance after: ${r['newbalanceDest']:.2f}\n"
            "\nIs this transaction fraudulent?"
        )
        records.append(_make_record(prompt, bool(r["isFraud"])))
    return records


# ---- 5. Sparkov Credit Card Transactions (already in repo) --------------
def convert_sparkov() -> list[dict]:
    train_csv = DATA_DIR / "fraudTrain.csv"
    test_csv = DATA_DIR / "fraudTest.csv"
    records = []
    for csv_path in [train_csv, test_csv]:
        if not csv_path.exists():
            continue
        print(f"  Converting Sparkov ({csv_path.name}) …")
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            merchant = str(r.get("merchant", "")).replace("fraud_", "")
            prompt = (
                "Analyze this credit-card transaction for potential fraud:\n"
                f"- Merchant: {merchant}\n"
                f"- Category: {r.get('category', '')}\n"
                f"- Amount: ${r.get('amt', 0):.2f}\n"
                f"- Location: {r.get('city', '')}, {r.get('state', '')}\n"
                f"- Cardholder occupation: {r.get('job', '')}\n"
                "\nIs this transaction fraudulent?"
            )
            records.append(_make_record(prompt, bool(r.get("is_fraud", 0))))
    return records


# ---------------------------------------------------------------------------
# Assemble & write JSONL
# ---------------------------------------------------------------------------
def write_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fraud-detection datasets")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (CSVs must already be in raw/)")
    parser.add_argument("--max-per-dataset", type=int, default=None,
                        help="Cap samples per dataset (useful for quick tests)")
    args = parser.parse_args()

    if not args.skip_download:
        download_all()

    print("\n=== Converting datasets to JSONL ===\n")
    all_records: list[dict] = []
    converters = [
        ("ULB Credit Card", convert_ulb),
        ("IEEE-CIS", convert_ieee_cis),
        ("E-Commerce Fraud", convert_ecommerce),
        ("PaySim", convert_paysim),
        ("Sparkov", convert_sparkov),
    ]

    for name, fn in converters:
        try:
            recs = fn()
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            recs = []
        if args.max_per_dataset and len(recs) > args.max_per_dataset:
            random.seed(SEED)
            recs = random.sample(recs, args.max_per_dataset)
        print(f"  {name}: {len(recs):,} records")
        all_records.extend(recs)

    random.seed(SEED)
    random.shuffle(all_records)

    train_recs, val_recs = train_test_split(
        all_records, test_size=VAL_RATIO, random_state=SEED
    )

    print(f"\n  Total: {len(all_records):,}  |  Train: {len(train_recs):,}  |  Val: {len(val_recs):,}\n")

    write_jsonl(train_recs, OUT_TRAIN)
    write_jsonl(val_recs, OUT_VAL)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
