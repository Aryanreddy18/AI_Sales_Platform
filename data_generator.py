import pandas as pd
import numpy as np
from datetime import timedelta, date
import random

# Generate realistic sales data
np.random.seed(42)
num_records = 2000

categories = ['Electronics', 'Furniture', 'Clothing', 'Groceries']
regions = ['North', 'South', 'East', 'West']
customers = [f"Customer_{i}" for i in range(1, 501)]

data = {
    'OrderID': [f"ORD-{1000+i}" for i in range(num_records)],
    'Date': [date(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(num_records)],
    'CustomerName': np.random.choice(customers, num_records),
    'Category': np.random.choice(categories, num_records, p=[0.4, 0.2, 0.3, 0.1]),
    'Region': np.random.choice(regions, num_records),
    'Sales': np.round(np.random.uniform(50, 5000, num_records), 2),
    'Quantity': np.random.randint(1, 10, num_records)
}

# Calculate profit (Furniture has lower profit margins)
df = pd.DataFrame(data)
df['Profit'] = df.apply(lambda row: row['Sales'] * np.random.uniform(0.05, 0.3) if row['Category'] != 'Furniture' else row['Sales'] * np.random.uniform(-0.1, 0.1), axis=1)
df['Profit'] = np.round(df['Profit'], 2)

# Save to CSV
df.to_csv("sales_data.csv", index=False)
print("✅ sales_data.csv successfully generated!")