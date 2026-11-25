from pathlib import Path

import pandas as pd

COLUMN_MAP = {
    "invoice": "invoice",
    "invoicedate": "invoice_date",
    "customer id": "customer_id",
    "customerid": "customer_id",
    "quantity": "quantity",
    "price": "price",
    "stockcode": "stock_code",
    "description": "description",
    "country": "country",
}

REQUIRED_COLUMNS = ["invoice", "invoice_date", "customer_id", "quantity", "price"]


def normalise_columns(raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        col: COLUMN_MAP[col.strip().lower()]
        for col in raw.columns
        if col.strip().lower() in COLUMN_MAP
    }
    df = raw.rename(columns=rename)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_transactions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
    elif suffix == ".csv":
        raw = pd.read_csv(path, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop unidentified customers, cancellations (invoice starts with 'C'), and returns or zero-price lines."""
    df = normalise_columns(raw).dropna(subset=["customer_id"]).copy()
    df["invoice"] = df["invoice"].astype(str)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype("int64")

    is_cancellation = df["invoice"].str.startswith("C")
    is_valid_line = (df["quantity"] > 0) & (df["price"] > 0)
    df = df[~is_cancellation & is_valid_line].copy()

    df["revenue"] = df["quantity"] * df["price"]
    return df[
        ["customer_id", "invoice", "invoice_date", "quantity", "price", "revenue"]
    ].reset_index(drop=True)
