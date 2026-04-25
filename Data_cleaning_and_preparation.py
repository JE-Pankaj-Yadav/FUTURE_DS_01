
# =============================================================================
# DATA CLEANING & PREPARATION ROADMAP
# Dataset: Invoice/Sales Transaction Data
# Columns: InvoiceNo, StockCode, Description, InvoiceDate,
#          UnitPrice, Quantity, CustomerID, Country, Total Sales
# =============================================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =============================================================================
# STEP 0: LOAD THE DATA
# =============================================================================

# Load from CSV (adjust path as needed)
df = pd.read_csv('D:/INTERNSHIP (DATA ANALYSTIC)/Future Interns/TASK 1/online_retail.csv', encoding="ISO-8859-1")

# Preview
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

# =============================================================================
# STEP 1: INITIAL DATA AUDIT
# =============================================================================

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- BASIC STATISTICS ---")
print(df.describe(include="all"))

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

print("\n--- DUPLICATE ROWS ---")
print(f"Total duplicate rows: {df.duplicated().sum()}")

# =============================================================================
# STEP 2: RENAME COLUMNS (Standardize naming convention)
# =============================================================================

# Strip whitespace from column names and convert to snake_case
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename 'total_sales_(please_provide_us_with_a_complete_roadmap_...)' or similar
# Adjust the exact name to match your actual column after loading
df.rename(columns={
    "invoiceno":   "invoice_no",
    "stockcode":   "stock_code",
    "description": "description",
    "invoicedate": "invoice_date",
    "unitprice":   "unit_price",
    "quantity":    "quantity",
    "customerid":  "customer_id",
    "country":     "country",
    # Rename whatever "total sales" resolved to:
    "total_sales": "total_sales"
}, inplace=True)

print("\nRenamed columns:", df.columns.tolist())

# =============================================================================
# STEP 3: FIX DATA TYPES
# =============================================================================

# 3a. Parse InvoiceDate as datetime
df["invoice_date"] = pd.to_datetime(df["invoice_date"], infer_datetime_format=True, errors="coerce")
print(f"\nDate parse failures: {df['invoice_date'].isna().sum()}")

# 3b. Ensure numeric columns are numeric
df["unit_price"]  = pd.to_numeric(df["unit_price"],  errors="coerce")
df["quantity"]    = pd.to_numeric(df["quantity"],     errors="coerce")
df["total_sales"] = pd.to_numeric(df["total_sales"],  errors="coerce")

# 3c. CustomerID should be treated as a string ID (not a float)
df["customer_id"] = df["customer_id"].astype("Int64")  # nullable integer
# OR, if you prefer string IDs:
# df["customer_id"] = df["customer_id"].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan)

# 3d. Strip whitespace from string columns
for col in ["invoice_no", "stock_code", "description", "country"]:
    df[col] = df[col].astype(str).str.strip()

print("\nData types after fixing:")
print(df.dtypes)

# =============================================================================
# STEP 4: HANDLE MISSING VALUES
# =============================================================================

# 4a. CustomerID – missing means guest/anonymous transaction
#     Flag them rather than drop, so we don't lose transaction data
df["is_guest"] = df["customer_id"].isna()
print(f"\nGuest (no CustomerID) transactions: {df['is_guest'].sum()}")

# 4b. Description – try to fill from other rows with the same StockCode
desc_map = (
    df.dropna(subset=["description"])
    .groupby("stock_code")["description"]
    .agg(lambda x: x.mode()[0] if len(x) > 0 else np.nan)
)
df["description"] = df["description"].fillna(df["stock_code"].map(desc_map))

# 4c. Remaining nulls in Description → fill with placeholder
df["description"] = df["description"].fillna("Unknown")

# 4d. UnitPrice / Quantity / TotalSales nulls – fill with NaN for now
#     (will be validated and dropped in Step 6)
print("\nMissing values after filling:")
print(df.isnull().sum())

# =============================================================================
# STEP 5: REMOVE DUPLICATES
# =============================================================================

before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"\nDuplicates removed: {before - after} rows")

# =============================================================================
# STEP 6: FILTER OUT INVALID / ANOMALOUS RECORDS
# =============================================================================

# 6a. Remove cancellations (InvoiceNo starting with 'C')
#     These are credit notes / returns – separate them if needed
cancellations = df[df["invoice_no"].str.startswith("C")].copy()
df = df[~df["invoice_no"].str.startswith("C")]
print(f"\nCancellation rows separated: {len(cancellations)}")

# 6b. Remove rows where Quantity <= 0 (bad data, not a return/cancellation)
invalid_qty = df[df["quantity"] <= 0]
print(f"Rows with Quantity <= 0 (non-cancellations): {len(invalid_qty)}")
df = df[df["quantity"] > 0]

# 6c. Remove rows where UnitPrice < 0
invalid_price = df[df["unit_price"] < 0]
print(f"Rows with UnitPrice < 0: {len(invalid_price)}")
df = df[df["unit_price"] >= 0]

# 6d. Remove test/internal stock codes (non-alphanumeric patterns like 'POST', 'DOT', 'BANK', etc.)
test_codes = ["POST", "DOT", "M", "BANK", "PADS", "D", "C2", "CRUK"]
df = df[~df["stock_code"].isin(test_codes)]
print(f"\nRows after removing test/internal stock codes: {len(df)}")

# =============================================================================
# STEP 7: VALIDATE AND RECALCULATE TOTAL SALES
# =============================================================================

# Recalculate expected total
df["expected_total"] = df["unit_price"] * df["quantity"]

# Find rows where TotalSales doesn't match (with tolerance for floating point)
tolerance = 0.01
mismatch = df[abs(df["total_sales"] - df["expected_total"]) > tolerance]
print(f"\nTotalSales mismatches: {len(mismatch)}")

# Overwrite TotalSales with the recalculated value (more reliable)
df["total_sales"] = df["expected_total"]
df.drop(columns=["expected_total"], inplace=True)

# =============================================================================
# STEP 8: HANDLE OUTLIERS
# =============================================================================

# 8a. Check distributions
print("\n--- Quantity stats ---")
print(df["quantity"].describe())
print("\n--- UnitPrice stats ---")
print(df["unit_price"].describe())

# 8b. IQR-based outlier detection for UnitPrice
Q1 = df["unit_price"].quantile(0.25)
Q3 = df["unit_price"].quantile(0.75)
IQR = Q3 - Q1
upper_price = Q3 + 3 * IQR   # 3×IQR = extreme outliers
lower_price = max(0, Q1 - 3 * IQR)

outlier_price = df[(df["unit_price"] > upper_price)]
print(f"\nExtreme UnitPrice outliers (>{upper_price:.2f}): {len(outlier_price)}")

# Option A: Cap (Winsorize) – keeps rows, limits extreme values
df["unit_price_capped"] = df["unit_price"].clip(upper=upper_price)

# Option B: Remove if they appear to be data entry errors
# df = df[df["unit_price"] <= upper_price]

# 8c. Same for Quantity
Q1_q = df["quantity"].quantile(0.25)
Q3_q = df["quantity"].quantile(0.75)
IQR_q = Q3_q - Q1_q
upper_qty = Q3_q + 3 * IQR_q
df["quantity_capped"] = df["quantity"].clip(upper=upper_qty)

# =============================================================================
# STEP 9: CLEAN STRING / CATEGORICAL COLUMNS
# =============================================================================

# 9a. Normalize Description text
df["description"] = (
    df["description"]
    .str.upper()           # Standardize to upper-case
    .str.strip()           # Remove leading/trailing spaces
    .str.replace(r"\s+", " ", regex=True)  # Collapse multiple spaces
)

# 9b. Standardize Country names
df["country"] = df["country"].str.strip().str.title()

# 9c. Check country value counts for anomalies
print("\nTop countries:")
print(df["country"].value_counts().head(10))

# 9d. Fix known misspellings (add as many as needed)
country_corrections = {
    "Eire":              "Ireland",
    "Rsa":               "South Africa",
    "Usa":               "United States",
    "Channel Islands":   "United Kingdom",
}
df["country"] = df["country"].replace(country_corrections)

# =============================================================================
# STEP 10: FEATURE ENGINEERING (Derived Columns)
# =============================================================================

# 10a. Date components – useful for time-series analysis
df["year"]       = df["invoice_date"].dt.year
df["month"]      = df["invoice_date"].dt.month
df["day"]        = df["invoice_date"].dt.day
df["day_of_week"] = df["invoice_date"].dt.dayofweek   # 0=Monday
df["hour"]       = df["invoice_date"].dt.hour
df["quarter"]    = df["invoice_date"].dt.quarter
df["week"]       = df["invoice_date"].dt.isocalendar().week.astype(int)

# 10b. Is the transaction from the UK (home market)?
df["is_uk"] = df["country"].str.strip().str.lower() == "united kingdom"

# 10c. Revenue tier per transaction
df["revenue_tier"] = pd.cut(
    df["total_sales"],
    bins=[0, 10, 50, 200, np.inf],
    labels=["Low", "Medium", "High", "Premium"]
)

# 10d. Average unit price per stock code (useful for imputation/comparison)
df["avg_price_per_stock"] = df.groupby("stock_code")["unit_price"].transform("mean")

# =============================================================================
# STEP 11: FINAL QUALITY CHECK
# =============================================================================

print("\n=== FINAL DATASET SUMMARY ===")
print(f"Shape: {df.shape}")
print(f"\nNull values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nDate range: {df['invoice_date'].min()} → {df['invoice_date'].max()}")
print(f"Unique customers: {df['customer_id'].nunique()}")
print(f"Unique stock codes: {df['stock_code'].nunique()}")
print(f"Unique countries: {df['country'].nunique()}")
print(f"\nSample cleaned data:")
print(df.head(3))

# =============================================================================
# STEP 12: VISUALISE DATA QUALITY (Optional but recommended)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 12a. Missing values heatmap
sns.heatmap(df.isnull(), cbar=False, ax=axes[0], yticklabels=False, cmap="viridis")
axes[0].set_title("Missing Values")

# 12b. UnitPrice distribution after capping
axes[1].hist(df["unit_price_capped"], bins=50, color="steelblue", edgecolor="white")
axes[1].set_title("UnitPrice Distribution (Capped)")
axes[1].set_xlabel("Unit Price")

# 12c. Transactions per month
monthly = df.groupby(["year", "month"]).size().reset_index(name="count")
monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
axes[2].bar(monthly["period"], monthly["count"], color="coral")
axes[2].set_title("Transactions per Month")
axes[2].set_xlabel("Month")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("data_quality_overview.png", dpi=150)
plt.show()
print("\nPlot saved → data_quality_overview.png")

# =============================================================================
# STEP 13: SAVE CLEANED DATASET
# =============================================================================

# output_path = "cleaned_invoice_data.csv"
# df.to_csv(output_path, index=False)
# print(f"\n✅ Cleaned dataset saved to: {output_path}")
# print(f"   Final shape: {df.shape}")


# Business Questions
# 1. Which products generate the most revenue?
product_revenue = df.groupby("description")["total_sales"].sum()
# groupby("description") → groups all rows by product name
# ["total_sales"].sum() → adds up all sales for each product

product_revenue = product_revenue.sort_values(ascending=False)
# sort_values() → arranges products from highest to lowest revenue
# ascending=False → highest revenue comes first

print(product_revenue.head(10))
# head(10) → shows only the top 10 products

# Visualize it
product_revenue.head(10).plot(kind="bar", color="steelblue", figsize=(12, 5))
plt.title("Top 10 Products by Revenue")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2.How do sales change over time?
# First make sure invoice_date is datetime
df["invoice_date"] = pd.to_datetime(df["invoice_date"])

# Extract month and year
df["month_year"] = df["invoice_date"].dt.to_period("M")
# dt.to_period("M") → converts date to month period like "2011-01"

monthly_sales = df.groupby("month_year")["total_sales"].sum()
# Groups all rows by month and sums total sales for each month

print(monthly_sales)

# Visualize it
monthly_sales.plot(kind="line", color="coral", figsize=(12, 5), marker="o")
# kind="line" → line chart is best for showing trends over time
# marker="o" → adds a dot at each data point
plt.title("Sales Over Time (Monthly)")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.Which categories or regions are most profitable?
region_sales = df.groupby("country")["total_sales"].sum()
# Groups all rows by country
# Sums total sales for each country

region_sales = region_sales.sort_values(ascending=False)
# Sorts from highest to lowest

print(region_sales.head(10))
# Shows top 10 countries by revenue

# Visualize it
region_sales.head(10).plot(kind="bar", color="green", figsize=(12, 5))
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.Where should the business focus to grow faster?
# Step 1 — Find low performing products
low_products = product_revenue[product_revenue < product_revenue.quantile(0.25)]
# quantile(0.25) → bottom 25% of products by revenue
# These are products that are not selling well

print("Low performing products:")
print(low_products)

# Step 2 — Find best performing country
best_country = region_sales.idxmax()
# idxmax() → returns the name of the country with highest sales
print(f"\nBest performing country: {best_country}")

# Step 3 — Find best month for sales
best_month = monthly_sales.idxmax()
# idxmax() → returns the month with highest sales
print(f"Best month for sales: {best_month}")

# Step 4 — Find best performing customers
top_customers = df.groupby("customer_id")["total_sales"].sum().sort_values(ascending=False)
print("\nTop 10 Customers:")
print(top_customers.head(10))
# These are your most valuable customers — focus on retaining them

# Where should the business focus to grow faster?
# The business should focus on high-performing products, profitable regions, and peak sales periods to grow faster. It should increase marketing and availability in these areas while improving or removing low-performing products and regions. Additionally, targeting repeat customers and optimizing pricing can help increase overall growth and profitability.