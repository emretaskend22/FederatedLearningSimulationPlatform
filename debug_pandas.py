import pandas as pd
import numpy as np

data_path = 'src/data'
train_path = f'{data_path}/adult-training-data.csv'

print("Loading csv...")
df = pd.read_csv(train_path, sep=';')
print("Columns:", df.columns.tolist())

# Clean column names
df.columns = df.columns.str.strip()
print("Cleaned Columns:", df.columns.tolist())

# Drop rows with missing values ' ?'
df = df.replace(' ?', pd.NA).dropna()

categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                    'relationship', 'race', 'sex', 'native.country']
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                  'capital.loss', 'hours.per.week']

print("Dtypes before processing:\n", df.dtypes)

features = df.drop(columns=['income'])

# One-hot
features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
print("Dtypes after get_dummies:\n", features.dtypes.value_counts())

# Check if any object cols remain
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
print("Dtypes after Scaler:\n", features.dtypes.value_counts())
print("Sample dtypes:\n", features.dtypes.head(10))

obj_cols = features.select_dtypes(include=['object']).columns
print("Remaining object cols:", obj_cols.tolist())
if len(obj_cols) > 0:
    print("Sample of object cols:\n", features[obj_cols].head())

# Try conversion
try:
    arr = features.values.astype(np.float32)
    print("Conversion success")
except Exception as e:
    print("Conversion failed:", e)
