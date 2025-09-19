import pandas as pd

# Recency, Frequency, Monetary (RFM) analysis -------------------------------
cleaned_df = pd.read_csv('data/cleaned_retail_data.csv')

cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], errors="coerce")
ref_date = cleaned_df["Date"].max()

# Add Recency (days since last purchase)
cleaned_df["Recency"] = (ref_date - cleaned_df["Date"]).dt.days

# Rename the existing columns for clarity
cleaned_df = cleaned_df.rename(columns={
    "Total_Purchases": "Frequency",
    "Total_Amount": "Monetary"
})

print(cleaned_df.head())

cleaned_df.dropna()

cleaned_df["R_Score"] = pd.qcut(cleaned_df["Recency"], 5, labels=[1,2,3,4,5])
cleaned_df["F_Score"] = pd.qcut(cleaned_df["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
cleaned_df["M_Score"] = pd.qcut(cleaned_df["Monetary"].rank(method="first"), 5, labels=[5,4,3,2,1])

cleaned_df["RFM_Score"] = cleaned_df[["R_Score","F_Score","M_Score"]].astype(int).sum(axis=1)

cleaned_df.info()

cleaned_df.to_csv('data/rfm_retail_data.csv', index=False)