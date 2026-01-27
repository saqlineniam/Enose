import pandas as pd
import glob
import os
from pathlib import Path


# Path to root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


# Folder locations
INPUT_FOLDER = ROOT_DIR / "data" / "raw"
OUTPUT_FOLDER = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_FOLDER / "processed_dataset.csv"

# Inhaling stage (flag 3)
TARGET_FLAG = 3 

# Defining sensor columns
SENSOR_COLS = [f"S{i}" for i in range(1, 33)]


# Preprocessing function
def preprocess_data():
    all_scenes_data = []

    # Getting all CSV files in the raw folder
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

    if not files:
        print("No files found!")
        return

    print(f"Found {len(files)} files. Processing S1-S32 for Flag {TARGET_FLAG}...")

    for filepath in files:
        filename = os.path.basename(filepath)

        try:
            # Skiping first 2 rows of metadata (date, time, other info)
            df = pd.read_csv(filepath, header=2)

            # Cleaning spaces in column names
            df.columns = [c.strip() for c in df.columns]

            # Checking if "flag" column exist
            if 'Flag' not in df.columns:
                print(f"Skipping {filename}: 'Flag' column missing.")
                continue
            
            # Checking if S1...S32 exist in this file
            available_sensors = [col for col in SENSOR_COLS if col in df.columns]
            
            if not available_sensors:
                print(f"Skipping {filename}: No Sensor columns (S1-S32) found.")
                continue

            # Filtering for "Flag" 3
            filtered_df = df[df['Flag'] == TARGET_FLAG]

            # Grouping by Class (Location) and taking MAX of SENSORS ONLY
            max_values = filtered_df.groupby('Class')[available_sensors].max()

            # Adding the Scene ID (File Name)
            max_values['Scene_ID'] = filename

            # Making Class a regular column again
            max_values.reset_index(inplace=True)

            all_scenes_data.append(max_values)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Combining and Saving
    if all_scenes_data:
        final_df = pd.concat(all_scenes_data, ignore_index=True)
        
        # Reordering columns: Class, Scene_ID, S1, S2... S32
        cols = ['Class', 'Scene_ID'] + [c for c in SENSOR_COLS if c in final_df.columns]
        final_df = final_df[cols]

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        save_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
        
        final_df.to_csv(save_path, index=False)
        print("-" * 30)
        print(f"SUCCESS! Processed data saved to: {save_path}")
        print(f"Extracted {len(final_df)} samples.")
    else:
        print("No valid data was extracted.")

if __name__ == "__main__":
    preprocess_data()