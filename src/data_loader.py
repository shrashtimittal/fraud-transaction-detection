import pandas as pd
import os
import glob

def load_all_data(data_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all .pkl files from the given folder into one DataFrame.
    """
    files = glob.glob(os.path.join(data_dir, "*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")
    
    dfs = []
    for f in files:
        df = pd.read_pickle(f)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

if __name__ == "__main__":
    data_dir = os.path.join("data")   # points to E:\fraud_detection_project\data
    df = load_all_data(data_dir)
    print("Loaded shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
