
#Price Prediction In Ahmedabad 2

# 1. Load and manipulate the CSV row by row

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
# Set dark theme for all charts at once
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'text.color':       '#ccccee',
})

# Load the CSV we built earlier
df = pd.read_csv('ahmedabad_housing_with_latlong.csv')

# 2. Feature engineering = creating new useful columns from existing data. We added 5 new columns.

# ── Zone column ──────────────────────────────────
west  = ['Satellite', 'Bodakdev', 'Vastrapur', 'Prahlad Nagar', 'SG Highway', 'Thaltej', 'Makarba', 'Memnagar', 'Ambawadi', 'Paldi', 'Usmanpura', 'Navrangpura', 'Naranpura', 'Shyamal', 'Anand Nagar', 'Jivraj Park', 'Maninagar', 'Ghuma', 'South Bopal', 'Chenpur', 'Sindhu Bhavan Road', 'Ambli', 'Sanand', 'Science City', 'Vaishno Devi Circle', 'SP Ring Road']
north = ['Chandkheda', 'Gota', 'Motera', 'Sabarmati', 'Kalol', 'Kudasan', 'Gandhinagar', 'Adalaj', 'Koba']
east  = ['Naroda', 'Nikol', 'Vastral', 'Rakhial', 'Bapunagar', 'Amraiwadi', 'Odhav', 'Ramol', 'C T M', 'Arjun Ashram', 'Danilimda']
south = ['Bareja', 'Kheda', 'Nadiad', 'Dholka', 'Sanand', 'Bavla', 'Sarkhej', 'Bhat', 'Shela', 'Changodar', 'Vasna', 'Shahwadi']

def get_zone(loc):
    if loc in west:  return 'West'
    if loc in north: return 'North'
    if loc in east:  return 'East'
    if loc in south: return 'South'
    return 'Central'

df['zone'] = df['locality'].apply(get_zone)

# ── Metro proximity (1/0 flag) ───────────────────
metro_areas = ['Vastrapur', 'Satellite', 'Chandkheda', 'Gota', 'Bodakdev', 'Thaltej', 'SG Highway', 'SP Ring Road', 'Vaishno Devi Circle', 'Science City']
df['metro_proximity'] = df['locality'].apply(
    lambda x: 1 if x in metro_areas else 0
)

# ── 2025 price adjustment (+10% YoY) ────────────
df['price_lakh_2025'] = (df['price_lakh'] * 1.10).round(2)
df['price_per_sqft_2025'] = (df['price_per_sqft'] * 1.10).round(2)

# ── Furnishing → numeric score ───────────────────
furn_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully Furnished': 2}
df['furnishing_score'] = df['furnishing'].map(furn_map)

# ── Floor: 'G' → 0, else int ────────────────────
df['floor_num'] = df['floor'].apply(
    lambda x: 0 if str(x).upper() == 'G' else int(x)
)


# 3. The dashboard is a 2×3 grid of subplots — all drawn in one figure using plt.subplots(2, 3).

# Create 2 rows × 3 columns of chart panels
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle('EDA Dashboard', fontsize=15)

# Panel [0,0] — Histogram by zone
ax = axes[0, 0]
for zone in ['West', 'North', 'East', 'South']:
    data = df[df['zone'] == zone]['price_lakh_2025']
    ax.hist(data, bins=14, alpha=0.75, label=zone)
ax.legend()

# Panel [0,1] — Bar chart avg price/sqft by zone
ax = axes[0, 1]
zone_avg = df.groupby('zone')['price_per_sqft_2025'].mean()
ax.bar(zone_avg.index, zone_avg.values)

# Panel [0,2] — Boxplot by BHK
ax = axes[0, 2]
bhk_data = [df[df['bhk']==b]['price_lakh_2025'].values
            for b in df['bhk'].unique()]
ax.boxplot(bhk_data, patch_artist=True)

# Panel [1,0] — Pie chart property types
ax = axes[1, 0]
counts = df['property_type'].value_counts()
ax.pie(counts, labels=counts.index, autopct='%1.1f%%')

# Panel [1,1] — Scatter (metro highlighted)
ax = axes[1, 1]
ax.scatter(df[df['metro_proximity']==0]['area_sqft'],
          df[df['metro_proximity']==0]['price_lakh_2025'],
          label='Non-Metro')
ax.scatter(df[df['metro_proximity']==1]['area_sqft'],
          df[df['metro_proximity']==1]['price_lakh_2025'],
          label='Metro Corridor', zorder=5)

# Panel [1,2] — Seaborn heatmap
ax = axes[1, 2]
pivot = df.pivot_table(values='price_lakh_2025',
                       index='zone', columns='furnishing',
                       aggfunc='mean')
sns.heatmap(pivot, ax=ax, cmap='RdPu', annot=True, fmt='.0f')

plt.tight_layout()
plt.savefig('eda_dashboard.png', dpi=150)
st.pyplot(fig)

# 4. Linear Regression predicts price from 9 numeric features. ML needs all inputs as numbers — we encode text columns first.

# Encode text → numbers for ML
le = LabelEncoder()
df['zone_enc'] = le.fit_transform(df['zone'])
df['type_enc'] = le.fit_transform(df['property_type'])

# Define features (X) and target (y)
features = ['bhk', 'area_sqft', 'furnishing_score',
            'floor_num', 'parking', 'age_years',
            'metro_proximity', 'zone_enc', 'type_enc']
X = df[features]
y = df['price_lakh_2025']

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on unseen test data
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred)   # 0.9454 → 94.5% variance explained
mae = mean_absolute_error(y_test, y_pred)  # ₹21.21 Lakh avg error

# Feature coefficients — what drives price most
coefs = pd.Series(model.coef_, index=features).sort_values()

# 5. Add model predictions + segments back to the dataframe, then export a Tableau-ready CSV.

# Add model predictions to every row
df['predicted_price_lakh'] = model.predict(X).round(2)
df['price_error_lakh']     = (
    df['price_lakh_2025'] - df['predicted_price_lakh']
).round(2)

# Segment column — for Tableau color coding
df['segment'] = pd.cut(
    df['price_lakh_2025'],
    bins=[0, 60, 120, 250, 900],
    labels=['Budget', 'Mid', 'Premium', 'Luxury']
)

# Export — only the columns Tableau needs
export_cols = [
    'locality', 'zone', 'latitude', 'longitude',
    'price_lakh_2025', 'price_per_sqft_2025',
    'metro_proximity', 'segment',
    'predicted_price_lakh', 'price_error_lakh'
]
df[export_cols].to_csv('ahmedabad_housing_final.csv', index=False)
