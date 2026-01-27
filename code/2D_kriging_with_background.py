import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from pykrige.ok import OrdinaryKriging
from pathlib import Path

# Path to root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Folder Locations
DATA_FILE = ROOT_DIR / "data" / "processed" / "processed_dataset.csv"
POSITIONS_FILE = ROOT_DIR / "data" / "positions.csv"
SCENE_IMAGES_FOLDER = ROOT_DIR / "data" / "Tray view"
OUTPUT_DIR = ROOT_DIR / "data" / "maps_2d"

# Grid Resolution
GRID_RES = 100


def run_2d_kriging_per_sensor():

    # 1. Load data
    df = pd.read_csv(DATA_FILE)
    pos_df = pd.read_csv(POSITIONS_FILE)

    # 2. Merge positions
    merged_df = pd.merge(df, pos_df, on="Class", how="inner")

    # Identify sensor columns
    sensor_cols = [c for c in merged_df.columns if c.startswith("S") and c[1:].isdigit()]

    # Process scene-wise
    for scene in merged_df["Scene_ID"].unique():
        print(f"\nProcessing Scene: {scene}")

        scene_folder = OUTPUT_DIR / scene.replace(".csv", "")
        os.makedirs(scene_folder, exist_ok=True)

        scene_data = merged_df[merged_df["Scene_ID"] == scene]

        # Extract scene number from scene name
        # Try to find scene number pattern
        scene_num = None
        if "_S" in scene:
            # Extract S1, S2, etc.
            parts = scene.split("_S")
            if len(parts) > 1:
                scene_num = "S" + parts[1].split("-")[0]
        
        # Load scene-specific background image
        if scene_num:
            scene_image_path = SCENE_IMAGES_FOLDER / f"T_5F_{scene_num}.png"
            if os.path.exists(scene_image_path):
                print(f"   Loading scene-specific image: {scene_image_path.name}")
                bg_img = mpimg.imread(scene_image_path)
            else:
                raise FileNotFoundError(f"Scene image not found: {scene_image_path}")
        else:
            raise ValueError(f"Could not extract scene number from: {scene}")
        
        TRAY_H, TRAY_W = bg_img.shape[:2]

        x = scene_data["X_cm"].values
        y = scene_data["Y_cm"].values

        grid_x = np.linspace(0, TRAY_W, GRID_RES)
        grid_y = np.linspace(0, TRAY_H, GRID_RES)

        # Storage for PDF generation
        successful_sensors = []
        z_pred_dict = {}
        z_var_dict = {}

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

                # Store for PDF generation
                successful_sensors.append(sensor)
                z_pred_dict[sensor] = z_pred
                z_var_dict[sensor] = z_var

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

        # ============== Generate PDF with 6x6 grids ==============
        if successful_sensors:
            print(f"\n   Generating PDF with {len(successful_sensors)} sensors...")
            
            pdf_path = scene_folder / f"{scene.replace('.csv', '')}_Kriging_Maps.pdf"
            
            with PdfPages(pdf_path) as pdf:
                # -------- PREDICTION MAPS (6x6 grid) --------
                n_sensors = len(successful_sensors)
                n_pages = int(np.ceil(n_sensors / 36))  # 36 = 6x6
                
                for page in range(n_pages):
                    fig = plt.figure(figsize=(18, 18))
                    fig.suptitle(f"{scene.replace('.csv', '')} - Prediction Maps (Page {page+1}/{n_pages})", 
                                fontsize=18, fontweight='bold', y=0.995)
                    
                    start_idx = page * 36
                    end_idx = min(start_idx + 36, n_sensors)
                    
                    for i, sensor in enumerate(successful_sensors[start_idx:end_idx]):
                        ax = plt.subplot(6, 6, i + 1)
                        
                        # Background
                        ax.imshow(bg_img, extent=[0, TRAY_W, TRAY_H, 0], origin="upper")
                        
                        # Kriging overlay
                        im = ax.imshow(z_pred_dict[sensor], extent=[0, TRAY_W, TRAY_H, 0], 
                                      origin="upper", cmap="jet", alpha=0.75)
                        
                        # Sample points
                        ax.scatter(x, y, c="white", edgecolors="black", s=15, linewidths=0.5)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)
                        
                        ax.set_title(sensor, fontsize=11, fontweight='bold')
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.99])
                    pdf.savefig(fig, dpi=150)
                    plt.close()
                
                # -------- UNCERTAINTY MAPS (6x6 grid) --------
                for page in range(n_pages):
                    fig = plt.figure(figsize=(18, 18))
                    fig.suptitle(f"{scene.replace('.csv', '')} - Uncertainty Maps (Page {page+1}/{n_pages})", 
                                fontsize=18, fontweight='bold', y=0.995)
                    
                    start_idx = page * 36
                    end_idx = min(start_idx + 36, n_sensors)
                    
                    for i, sensor in enumerate(successful_sensors[start_idx:end_idx]):
                        ax = plt.subplot(6, 6, i + 1)
                        
                        # Background
                        ax.imshow(bg_img, extent=[0, TRAY_W, TRAY_H, 0], origin="upper")
                        
                        # Variance overlay
                        im = ax.imshow(z_var_dict[sensor], extent=[0, TRAY_W, TRAY_H, 0], 
                                      origin="upper", cmap="viridis", alpha=0.65)
                        
                        # Sample points
                        ax.scatter(x, y, c="white", edgecolors="black", s=15, linewidths=0.5)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)
                        
                        ax.set_title(f"{sensor}", fontsize=11, fontweight='bold')
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.99])
                    pdf.savefig(fig, dpi=150)
                    plt.close()
            
            print(f"   PDF saved: {pdf_path}")


if __name__ == "__main__":
    run_2d_kriging_per_sensor()