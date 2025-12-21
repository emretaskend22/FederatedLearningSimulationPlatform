import sys
import os
sys.path.append(os.getcwd())
try:
    from src.data.datasets import load_data
    train, test = load_data('src/data')
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    print(f"Feature dim: {train[0][0].shape}")
except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
