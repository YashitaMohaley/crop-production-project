# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:44:52 2026

@author: LENOVO
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

#LOAD DATA
df = pd.read_csv("C:/Users/LENOVO/Desktop/excel&powerbi/CropProject/crop_production.csv")

print("Data loaded ")
print("Dataset size:", df.shape)

print("\nFirst few rows:")
print(df.head())

print("\nColumns present in dataset:")
print(df.columns.tolist())



# overview
print("\nDataset info:")
print(df.info())

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

# Checking diversity in data
print("\nNo. of States:", df['State_Name'].nunique())
print("No. of Districts:", df['District_Name'].nunique())
print("No. of Crops:", df['Crop'].nunique())

print("\nSeasons available:", df['Season'].unique())
print("Year range:", df['Crop_Year'].min(), "to", df['Crop_Year'].max())

# Missing values check
print("\nMissing values count:")
print(df.isnull().sum())

print("\nMissing percentage:")
print((df.isnull().sum() / len(df)) * 100)

# Most frequent crops
print("\nTop crops:")
print(df['Crop'].value_counts().head(10))

# Season distribution
print("\nSeason distribution:")
print(df['Season'].value_counts())


# Fixing spacing issues in text columns
df['Season'] = df['Season'].str.strip()
df['State_Name'] = df['State_Name'].str.strip()
df['District_Name'] = df['District_Name'].str.strip()
df['Crop'] = df['Crop'].str.strip()

print("\nCleaned season values:", df['Season'].unique())

# Handling missing Production values
# Filling with median of respective crop (more meaningful)
df['Production'] = df.groupby('Crop')['Production'].transform(
    lambda x: x.fillna(x.median())
)

# If still anything is left, fill with overall median
df['Production'].fillna(df['Production'].median(), inplace=True)

# Removing invalid entries (Area = 0 doesn't make sense)
before_rows = len(df)
df = df[df['Area'] > 0]
after_rows = len(df)

print(f"\nRemoved {before_rows - after_rows} invalid rows (Area = 0)")

# Removing extreme outliers 
Q1 = df['Production'].quantile(0.25)
Q3 = df['Production'].quantile(0.75)
IQR = Q3 - Q1

df = df[df['Production'] <= Q3 + 3 * IQR]

print("\nAfter cleaning:")
print("Shape:", df.shape)
print("Remaining nulls:\n", df.isnull().sum())


 
# Creating a new useful feature (Yield)
df['Yield'] = df['Production'] / df['Area']

# Encoding categorical data
le_state = LabelEncoder()
le_dist = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df['State_Enc'] = le_state.fit_transform(df['State_Name'])
df['District_Enc'] = le_dist.fit_transform(df['District_Name'])
df['Season_Enc'] = le_season.fit_transform(df['Season'])
df['Crop_Enc'] = le_crop.fit_transform(df['Crop'])

print("\nEncoding done ✔")
print(df[['State_Name', 'State_Enc', 'Crop', 'Crop_Enc']].head())

# Selecting features and target
features = ['State_Enc', 'District_Enc', 'Crop_Year', 
            'Season_Enc', 'Crop_Enc', 'Area']

X = df[features]
y = df['Production']

# Splitting data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# Top crops by production
plt.figure(figsize=(12, 5))
top_crops = df.groupby('Crop')['Production'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_crops.values, y=top_crops.index)
plt.title("Top Crops by Production")
plt.xlabel("Production")
plt.tight_layout()
plt.show()

# Yearly trend
plt.figure(figsize=(12, 5))
yearly = df.groupby('Crop_Year')['Production'].sum()
plt.plot(yearly.index, yearly.values, marker='o')
plt.title("Production over Years")
plt.xlabel("Year")
plt.ylabel("Production")
plt.grid()
plt.tight_layout()
plt.show()

# Season-wise share
plt.figure(figsize=(7, 7))
season_prod = df.groupby('Season')['Production'].sum()
plt.pie(season_prod, labels=season_prod.index, autopct='%1.1f%%')
plt.title("Season-wise Production")
plt.show()

# Top states
plt.figure(figsize=(12, 5))
state_prod = df.groupby('State_Name')['Production'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=state_prod.values, y=state_prod.index)
plt.title("Top States by Production")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
cols = ['Crop_Year', 'Area', 'Production', 'Yield', 'Crop_Enc', 'Season_Enc']
sns.heatmap(df[cols].corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()


# Using Random Forest for prediction
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nModel training completed ")

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# Feature importance
importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# Actual vs predicted comparison
plt.figure(figsize=(8, 5))
plt.scatter(y_test[:200], y_pred[:200], alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()