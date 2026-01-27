import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pykrige.ok import OrdinaryKriging
from pathlib import Path

# Path to root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Folder Locations
DATA_FILE = ROOT_DIR / "data" / "processed" / "processed_dataset.csv"
POSITIONS_FILE = ROOT_DIR / "data" / "positions.csv"
TRAY_IMAGE_FILE = ROOT_DIR / "data" / "geometry" / "tray_cropped.png"
OUTPUT_DIR = ROOT_DIR / "data" / "maps_2d"

# Grid Resolution
GRID_RES = 100


def run_2d_kriging_per_sensor():

    # 1. Load data
    df = pd.read_csv(DATA_FILE)
    pos_df = pd.read_csv(POSITIONS_FILE)

    # 2. Load tray image for dimensions
    if not os.path.exists(TRAY_IMAGE_FILE):
        raise FileNotFoundError("Tray image not found.")

    bg_img = mpimg.imread(TRAY_IMAGE_FILE)
    TRAY_H, TRAY_W = bg_img.shape[:2]

    # 3. Merge positions
    merged_df = pd.merge(df, pos_df, on="Class", how="inner")

    # Identify sensor columns
    sensor_cols = [c for c in merged_df.columns if c.startswith("S") and c[1:].isdigit()]

    # Process scene-wise
    for scene in merged_df["Scene_ID"].unique():
        print(f"\nProcessing Scene: {scene}")

        scene_folder = OUTPUT_DIR / scene.replace(".csv", "")
        os.makedirs(scene_folder, exist_ok=True)

        scene_data = merged_df[merged_df["Scene_ID"] == scene]

        x = scene_data["X_cm"].values
        y = scene_data["Y_cm"].values

        grid_x = np.linspace(0, TRAY_W, GRID_RES)
        grid_y = np.linspace(0, TRAY_H, GRID_RES)

        # ---- loop over sensors ----
        for sensor in sensor_cols:
            print(f"   -> Sensor {sensor}")

            values = scene_data[sensor].values

            try:
                ok = OrdinaryKriging(
                    x,
                    y,
                    values,
                    variogram_model="linear",
                    nlags=10,
                    exact_values=True,
                    verbose=False,
                    enable_plotting=False,
                )

                z_pred, z_var = ok.execute("grid", grid_x, grid_y)

                # ---------------- Prediction map ----------------
                fig, ax = plt.subplots(figsize=(8, 6))

                # Background tray
                ax.imshow(
                    bg_img,
                    extent=[0, TRAY_W, TRAY_H, 0],
                    origin="upper"
                )

                # Kriging overlay
                im = ax.imshow(
                    z_pred,
                    extent=[0, TRAY_W, TRAY_H, 0],
                    origin="upper",
                    cmap="jet",
                    alpha=0.75
                )

                plt.colorbar(im, label="Sensor Response")

                ax.scatter(
                    x,
                    y,
                    c="white",
                    edgecolors="black",
                    s=80
                )

                ax.set_title(f"{scene} | {sensor}")
                ax.set_xlabel("X (cm)")
                ax.set_ylabel("Y (cm)")

                plt.savefig(
                    scene_folder / f"Map_{sensor}.png",
                    dpi=120,
                    bbox_inches="tight"
                )
                plt.close()

                # ---------------- Uncertainty map ----------------
                fig, ax = plt.subplots(figsize=(8, 6))

                # Background tray
                ax.imshow(
                    bg_img,
                    extent=[0, TRAY_W, TRAY_H, 0],
                    origin="upper"
                )

                # Variance overlay
                im = ax.imshow(
                    z_var,
                    extent=[0, TRAY_W, TRAY_H, 0],
                    origin="upper",
                    cmap="viridis",
                    alpha=0.65
                )

                plt.colorbar(im, label="Kriging Variance")

                ax.scatter(
                    x,
                    y,
                    c="white",
                    edgecolors="black",
                    s=80
                )

                ax.set_title(f"{scene} | {sensor} – Uncertainty")
                ax.set_xlabel("X (cm)")
                ax.set_ylabel("Y (cm)")

                plt.savefig(
                    scene_folder / f"Map_{sensor}_Uncertainty.png",
                    dpi=120,
                    bbox_inches="tight"
                )
                plt.close()

            except Exception as e:
                print(f"      Error in {sensor}: {e}")


if __name__ == "__main__":
    run_2d_kriging_per_sensor()
