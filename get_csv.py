import os
import pandas as pd

# Change to the directory containing the CSV files
csv_dir = "/Users/aidanbeilke/Desktop/Cape_TM_Data/updated"
os.chdir(csv_dir)

# List all CSV files in the directory
csv_files = [f for f in os.listdir() if f.endswith('.csv')]
print("CSV files found:", csv_files)

# Read all CSV files into dataframes
dataframes = []
for f in csv_files:
    df = pd.read_csv(f, sep=",")
    dataframes.append(df)
    print(f"Read {f} with columns: {df.columns}")

# Combine all dataframes into a single dataframe
combined_data = pd.concat(dataframes, ignore_index=True)
print("Combined data shape:", combined_data.shape)

# Change the working directory to save the combined CSV
save_dir = "/Users/aidanbeilke/Desktop/Postgame_pdfs/csvs"
os.chdir(save_dir)

# Write the combined dataframe to a new CSV file without the index
combined_data.to_csv("combined_data.csv", index=False)
print("Combined data saved to combined_data.csv")
