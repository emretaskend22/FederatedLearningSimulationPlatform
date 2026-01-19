import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class AdultDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.split = split
        self.transform = transform
        
        # Define paths
        train_path = os.path.join(data_path, 'adult-training-data.csv')
        test_path = os.path.join(data_path, 'adult-test-data.csv')
        
        # Load Data
        df = pd.read_csv(train_path if split == 'train' else test_path, sep=';')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Drop rows with missing values ' ?'
        df = df.replace(' ?', pd.NA).dropna()
        
        # Handle trailing dots in test set income/label
        if df['income'].dtype == object:
             df['income'] = df['income'].str.strip().str.rstrip('.')

        # Feature Engineering
        # Identify categorical and numerical columns
        categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                            'relationship', 'race', 'sex', 'native.country']
        numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                          'capital.loss', 'hours.per.week']
        
        # Processing (Simulated simple preprocessing for now - strictly should fit on train only)
        # Note: In a real FL scenario, everyone needs same encoder. 
        # Here we do a simplified version: assuming static encoding map or fitting per client in a real scenario
        # verifying consistency is tricky. ideally we preload fit encoders. 
        # For this simulation, we will use pandas get_dummies for simplicity and speed, 
        # aligning columns by reindexing against a known schema if needed. 
        # BUT to match torch expectation, let's use LabelEncoding for simplicity of numerical conversion first.
        
        self.labels = (df['income'] == '>50K').astype(int).values
        self.features = df.drop(columns=['income'])
        
        # Simple One-Hot Encoding
        self.features = pd.get_dummies(self.features, columns=categorical_cols, drop_first=True)
        
        # Normalize numerical
        scaler = StandardScaler()
        self.features[numerical_cols] = scaler.fit_transform(self.features[numerical_cols])
        
        # Ensure numeric type for all features (bool -> float)
        self.features = self.features.astype(float)

        self.X = torch.tensor(self.features.values, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(data_path):
    """
    Helper to return train/test datasets. 
    Notes: In a real distributed system, we must ensure feature columns align (same dummies).
    This implementation handles it locally for simulation.
    """
    train_ds = AdultDataset(data_path, split='train')
    test_ds = AdultDataset(data_path, split='test')
    
    # Align columns (test might miss some categories present in train)
    # Get union of columns
    # This is a bit hacky for a robust system but works for standard Adult dataset simulation
    # A better way is to fit the encoder on concatenation of both.
    
    # Re-implementing slightly to ensure alignment:
    
    # 1. Load Raw
    train_raw = pd.read_csv(os.path.join(data_path, 'adult-training-data.csv'), sep=';')
    test_raw = pd.read_csv(os.path.join(data_path, 'adult-test-data.csv'), sep=';')
    
    train_raw.columns = train_raw.columns.str.strip()
    test_raw.columns = test_raw.columns.str.strip()
    
    # Clean
    train_raw = train_raw.replace(' ?', pd.NA).dropna()
    test_raw = test_raw.replace(' ?', pd.NA).dropna()
    
    test_raw['income'] = test_raw['income'].str.strip().str.rstrip('.')
    train_raw['income'] = train_raw['income'].str.strip()

    # Concatenate for consistent encoding
    train_raw['is_train'] = 1
    test_raw['is_train'] = 0
    full_df = pd.concat([train_raw, test_raw], ignore_index=True)
    
    # Encoding
    categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                        'relationship', 'race', 'sex', 'native.country']
    full_df = pd.get_dummies(full_df, columns=categorical_cols, drop_first=True)
    
    # Normalize Numerical
    numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                      'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    full_df[numerical_cols] = scaler.fit_transform(full_df[numerical_cols])
    
    # Split back
    train_proc = full_df[full_df['is_train'] == 1].drop(columns=['is_train'])
    test_proc = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])
    
    # Extract X, y
    X_train = torch.tensor(train_proc.drop(columns=['income']).astype(float).values, dtype=torch.float32)
    y_train = torch.tensor((train_proc['income'] == '>50K').astype(int).values, dtype=torch.float32).unsqueeze(1)
    
    X_test = torch.tensor(test_proc.drop(columns=['income']).astype(float).values, dtype=torch.float32)
    y_test = torch.tensor((test_proc['income'] == '>50K').astype(int).values, dtype=torch.float32).unsqueeze(1)
    
    from torch.utils.data import TensorDataset
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
