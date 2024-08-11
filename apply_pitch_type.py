import pickle
import pandas as pd
import os

# Load data
cape = pd.read_csv("/Users/aidanbeilke/Desktop/Postgame_pdfs/csvs/combined_data.csv")
print("Initial DataFrame shape:", cape.shape)

# Load model
with open('models/knn_best_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)


pitch_type_mapping = {
    0: 'ChangeUp',  # Changeup
    1: 'Curveball',  # Curveball
    2: 'Cutter',  # Cutter
    3: 'Fastball',  # Four-seam Fastball
    4: 'Sinker',  # Splitter
    5: 'Slider',  # Sinker
    6: 'Splitter'   # Slider
}   

def get_pitch_type_var(df):

    pt_feats = ['RelSpeed', 'SpinRate', 'HorzBreak', 
                'InducedVertBreak', 'PitcherThrows']

    missing_columns = [col for col in pt_feats if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if 'PitcherThrows' in df.columns:
        df['PitcherThrows'] = df['PitcherThrows'].map({'Right': 0, 'Left': 1})
        if df['PitcherThrows'].isna().any():
            print("NaN values found in 'PitcherThrows' after mapping.")
            # Optionally handle or fill NaN values
            df['PitcherThrows'].fillna(-1, inplace=True)  # Example: Defaulting NaN to -1 or another placeholder
    else:
        raise ValueError("'PitcherThrows' column is missing.")

    # Check for NaN in features before prediction
    if df[pt_feats].isna().any().any():
        print("NaN values found in features. Details:")
        print(df[pt_feats].isna().sum())
        # Handle NaNs by dropping rows or filling with a value (e.g., median, mean)
        df = df.dropna(subset=pt_feats)

    # Prediction
    df['pitch_type_code'] = knn_model.predict(df[pt_feats])
    df['pitch_type'] = df['pitch_type_code'].map(pitch_type_mapping)

    return df

# Process data
cape_new = get_pitch_type_var(cape)
print("Processed DataFrame shape:", cape_new.shape)

# Save results
os.chdir("/Users/aidanbeilke/Desktop/Postgame_pdfs/csvs")
cape_new.to_csv("ml_ready.csv", index=False)
print("Data saved to ml_ready.csv")
