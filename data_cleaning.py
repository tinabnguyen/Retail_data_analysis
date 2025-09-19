import pandas as pd

df = pd.read_csv('data/retail_data.csv')

# data cleaning -------------------------------------------------------------

cleaned_df = df[[
    "Customer_ID", "Age", "Gender", "Date", 
    "Total_Purchases", "Amount", "Total_Amount"
]]

cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], errors="coerce")
cleaned_df = cleaned_df.dropna()

cleaned_df.to_csv('data/cleaned_retail_data.csv', index=False)

print(cleaned_df.info())