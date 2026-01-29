import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import json
from pykrige.ok import OrdinaryKriging
import seaborn as sns
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT_DIR / "data" / "processed" / "processed_dataset.csv"
POSITIONS_FILE = ROOT_DIR / "data" / "positions.csv"
FOOD_REGIONS_FILE = ROOT_DIR / "data" / "food_regions_per_scene.json"
SCENE_IMAGES_FOLDER = ROOT_DIR / "data" / "Tray view"
OUTPUT_DIR = ROOT_DIR / "data" / "food_sensitivity"

GRID_RES = 100
FOOD_ITEMS = ["Orange", "Milk", "Bread", "Bacon", "Cabbage"]

def apply_drift_correction(df, sensor_cols):
    """
    Apply baseline drift correction using rolling quantile method.
    """
    print("\n" + "="*70)
    print("APPLYING DRIFT CORRECTION")
    print("="*70)
    
    df_sorted = df.sort_values('Scene_ID').copy()
    
    samples_per_scene = len(df) // df['Scene_ID'].nunique()
    k_window = max(samples_per_scene, 5)
    q_quantile = 0.10
    
    print(f"Window size: {k_window} samples")
    print(f"Quantile: {q_quantile} (10th percentile)")
    
    corrected_data = df_sorted.copy()
    baseline_values = pd.DataFrame(index=df_sorted.index)
    
    for col in sensor_cols:
        baseline = df_sorted[col].rolling(window=k_window, min_periods=1).quantile(q_quantile)
        corrected_data[col] = df_sorted[col] - baseline
        baseline_values[col] = baseline
    
    corrected_data = corrected_data.sort_index()
    baseline_values = baseline_values.sort_index()
    
    print("✓ Drift correction applied")
    
    return corrected_data, baseline_values


def plot_drift_comparison(original_df, corrected_df, baseline_values, sensor_cols, output_dir):
    """
    Plot before/after drift correction for example sensors.
    """
    print("\n" + "="*70)
    print("📊 CREATING DRIFT CORRECTION VISUALIZATIONS")
    print("="*70)
    
    example_sensors = sensor_cols[:4]
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    for idx, sensor in enumerate(example_sensors):
        ax_before = axes[idx, 0]
        ax_before.plot(original_df.index, original_df[sensor], 
                      color='blue', alpha=0.6, label='Raw Data')
        ax_before.plot(baseline_values.index, baseline_values[sensor], 
                      color='orange', linewidth=2, linestyle='--', 
                      label='Estimated Baseline (10th percentile)')
        ax_before.set_ylabel('Sensor Value', fontweight='bold')
        ax_before.set_title(f'{sensor} - Before Drift Correction', fontweight='bold')
        ax_before.legend(loc='best', fontsize=8)
        ax_before.grid(True, alpha=0.3)
        
        ax_after = axes[idx, 1]
        ax_after.plot(corrected_df.index, corrected_df[sensor], 
                     color='green', label='Drift-Corrected Signal')
        ax_after.axhline(0, color='black', linestyle=':', alpha=0.5)
        ax_after.set_ylabel('Corrected Value', fontweight='bold')
        ax_after.set_title(f'{sensor} - After Drift Correction', fontweight='bold')
        ax_after.legend(loc='best', fontsize=8)
        ax_after.grid(True, alpha=0.3)
        
        if idx == 3:
            ax_before.set_xlabel('Sample Index', fontweight='bold')
            ax_after.set_xlabel('Sample Index', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "drift_correction_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Drift correction comparison saved")


def analyze_with_and_without_drift_correction():
    """
    Run the full analysis twice: once with drift correction, once without.
    """
    
    df = pd.read_csv(DATA_FILE)
    pos_df = pd.read_csv(POSITIONS_FILE)
    merged_df = pd.merge(df, pos_df, on="Class", how="inner")
    
    with open(FOOD_REGIONS_FILE, 'r') as f:
        all_food_regions = json.load(f)
    
    sensor_cols = [c for c in merged_df.columns if c.startswith("S") and c[1:].isdigit()]
    unique_scenes = sorted(merged_df["Scene_ID"].unique())
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("PART 1: ANALYSIS WITH DRIFT CORRECTION")
    print("="*80)
    
    corrected_df, baseline_values = apply_drift_correction(merged_df, sensor_cols)
    plot_drift_comparison(merged_df, corrected_df, baseline_values, sensor_cols, OUTPUT_DIR)
    
    results_corrected = run_ranking_analysis(
        corrected_df, pos_df, all_food_regions, sensor_cols, unique_scenes,
        output_suffix="_with_drift_correction"
    )
    
    print("\n" + "="*80)
    print("PART 2: ANALYSIS WITHOUT DRIFT CORRECTION")
    print("="*80)
    
    results_original = run_ranking_analysis(
        merged_df, pos_df, all_food_regions, sensor_cols, unique_scenes,
        output_suffix="_no_drift_correction"
    )
    
    print("\n" + "="*80)
    print("📊 COMPARING RESULTS")
    print("="*80)
    
    compare_results(results_corrected, results_original)


def run_ranking_analysis(merged_df, pos_df, all_food_regions, sensor_cols, unique_scenes, output_suffix=""):
    """
    Run the ranking consistency analysis with all visualizations.
    """
    
    scene_specific_results = {scene: {sensor: {} for sensor in sensor_cols} for scene in unique_scenes}
    
    print("\nAnalyzing sensor-food sensitivity...")
    
    for scene in unique_scenes:
        if scene not in all_food_regions:
            continue
        
        scene_food_regions = all_food_regions[scene]
        scene_data = merged_df[merged_df["Scene_ID"] == scene].copy()
        
        scaler = StandardScaler()
        scene_data[sensor_cols] = scaler.fit_transform(scene_data[sensor_cols])
        
        scene_num = None
        if "_S" in scene:
            parts = scene.split("_S")
            if len(parts) > 1:
                scene_num = "S" + parts[1].split("-")[0]
        
        scene_image_path = SCENE_IMAGES_FOLDER / f"T_5F_{scene_num}.png"
        bg_img = mpimg.imread(scene_image_path)
        TRAY_H, TRAY_W = bg_img.shape[:2]
        
        x = scene_data["X_cm"].values
        y = scene_data["Y_cm"].values
        
        grid_x = np.linspace(0, TRAY_W, GRID_RES)
        grid_y = np.linspace(0, TRAY_H, GRID_RES)
        
        for sensor in sensor_cols:
            values = scene_data[sensor].values
            
            if len(np.unique(values)) == 1:
                continue
            
            try:
                ok = OrdinaryKriging(
                    x, y, values,
                    variogram_model="linear",
                    exact_values=True,
                    verbose=False,
                    enable_plotting=False
                )
                
                z_pred, z_var = ok.execute("grid", grid_x, grid_y)
                
                for food_name, region in scene_food_regions.items():
                    x_start_idx = int((region['x_start'] / TRAY_W) * GRID_RES)
                    x_end_idx = int((region['x_end'] / TRAY_W) * GRID_RES)
                    y_start_idx = int((region['y_start'] / TRAY_H) * GRID_RES)
                    y_end_idx = int((region['y_end'] / TRAY_H) * GRID_RES)
                    
                    food_region_values = z_pred[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
                    
                    if food_region_values.size > 0:
                        mean_response = np.mean(food_region_values)
                        scene_specific_results[scene][sensor][food_name] = mean_response
                
            except Exception as e:
                continue
    
    rank_stats_data = []
    
    for sensor in sensor_cols:
        sensor_ranks = {food: [] for food in FOOD_ITEMS}
        
        for scene in unique_scenes:
            if scene not in all_food_regions:
                continue
            
            sensor_data = scene_specific_results[scene][sensor]
            
            if len(sensor_data) == len(FOOD_ITEMS):
                sorted_foods = sorted(sensor_data.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (food, value) in enumerate(sorted_foods, 1):
                    sensor_ranks[food].append(rank)
        
        row = {'Sensor': sensor}
        
        for food in FOOD_ITEMS:
            if sensor_ranks[food]:
                avg_rank = np.mean(sensor_ranks[food])
                std_rank = np.std(sensor_ranks[food])
                row[f"{food}_Avg_Rank"] = round(avg_rank, 2)
                row[f"{food}_Std_Dev"] = round(std_rank, 4)
        
        all_stds = [np.std(sensor_ranks[food]) for food in FOOD_ITEMS if sensor_ranks[food]]
        row['Avg_Std_Dev'] = round(np.mean(all_stds), 4) if all_stds else None
        
        rank_stats_data.append(row)
    
    rank_stats_df = pd.DataFrame(rank_stats_data).sort_values('Avg_Std_Dev')
    
    # ============== VISUALIZATIONS ==============
    
    # 1. Consistency bar chart (top 20)
    plt.figure(figsize=(14, 8))
    top_20 = rank_stats_df.head(20)
    
    colors = ['green' if x < 0.5 else 'orange' if x < 1.0 else 'red' 
              for x in top_20['Avg_Std_Dev']]
    
    plt.barh(range(len(top_20)), top_20['Avg_Std_Dev'], color=colors)
    plt.yticks(range(len(top_20)), top_20['Sensor'])
    plt.xlabel('Average Rank Std Dev (lower = more consistent)', fontsize=12, fontweight='bold')
    plt.ylabel('Sensor', fontsize=12, fontweight='bold')
    plt.title(f'Ranking Consistency Across Scenes{output_suffix.replace("_", " ").title()}', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Very consistent (< 0.5)')
    plt.axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Consistent (< 1.0)')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ranking_consistency{output_suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of rank std deviations - ALL SENSORS
    print("Creating heatmap for ALL sensors...")
    heatmap_data = []
    all_sensors = rank_stats_df['Sensor'].tolist()  # ALL sensors, sorted by consistency
    
    for sensor in all_sensors:
        row_data = []
        sensor_row = rank_stats_df[rank_stats_df['Sensor'] == sensor].iloc[0]
        for food in FOOD_ITEMS:
            std_col = f"{food}_Std_Dev"
            if std_col in sensor_row:
                row_data.append(sensor_row[std_col])
            else:
                row_data.append(np.nan)
        heatmap_data.append(row_data)
    
    # Create larger figure to fit all sensors
    fig_height = max(12, len(all_sensors) * 0.4)  # Scale height based on number of sensors
    plt.figure(figsize=(10, fig_height))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r',
               xticklabels=FOOD_ITEMS, yticklabels=all_sensors,
               cbar_kws={'label': 'Rank Std Dev (0 = perfectly consistent)'},
               linewidths=0.5, linecolor='gray')
    
    plt.title(f'Ranking Consistency Heatmap{output_suffix.replace("_", " ").title()}\n(ALL Sensors - Sorted by Overall Consistency)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Food Item', fontsize=12, fontweight='bold')
    plt.ylabel('Sensor', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ranking_std_heatmap_all_sensors{output_suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Ranking stability for top 3 sensors
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for idx, (sensor_idx, row) in enumerate(rank_stats_df.head(3).iterrows()):
        ax = axes[idx]
        sensor = row['Sensor']
        
        scene_rankings = []
        scene_labels = []
        
        for scene in unique_scenes:
            if scene not in all_food_regions:
                continue
            
            sensor_data = scene_specific_results[scene][sensor]
            if len(sensor_data) == len(FOOD_ITEMS):
                sorted_foods = sorted(sensor_data.items(), key=lambda x: x[1], reverse=True)
                scene_num = scene.split("_S")[1].split("-")[0] if "_S" in scene else scene
                
                ranks = []
                for food in FOOD_ITEMS:
                    rank = next((i+1 for i, (f, _) in enumerate(sorted_foods) if f == food), None)
                    ranks.append(rank)
                
                scene_rankings.append(ranks)
                scene_labels.append(f"S{scene_num}")
        
        x = np.arange(len(FOOD_ITEMS))
        width = 0.12
        
        for i, (ranks, label) in enumerate(zip(scene_rankings, scene_labels)):
            offset = (i - len(scene_rankings)/2) * width
            ax.bar(x + offset, ranks, width, label=label, alpha=0.8)
        
        ax.set_ylabel('Rank (1=highest)', fontsize=10, fontweight='bold')
        ax.set_title(f'{sensor} (Avg Std: {row["Avg_Std_Dev"]:.4f})', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(FOOD_ITEMS)
        ax.set_ylim(0, 6)
        ax.invert_yaxis()
        ax.legend(ncol=7, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ranking_stability{output_suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save Excel
    excel_path = OUTPUT_DIR / f"food_ranking{output_suffix}.xlsx"
    rank_stats_df.to_excel(excel_path, index=False)
    
    print(f"✓ Results saved: {excel_path}")
    print(f"✓ Consistency bar chart saved (top 20)")
    print(f"✓ Heatmap saved (ALL {len(all_sensors)} sensors)")
    print(f"✓ Ranking stability chart saved")
    
    return rank_stats_df


def compare_results(results_corrected, results_original):
    """
    Compare ranking consistency with and without drift correction.
    """
    
    print("\nTop 10 Most Consistent Sensors:")
    print(f"{'Rank':<6} {'With Drift Corr.':<20} {'Std Dev':<12} {'Without Drift Corr.':<20} {'Std Dev'}")
    print("-" * 80)
    
    for i in range(10):
        sensor_corr = results_corrected.iloc[i]['Sensor']
        std_corr = results_corrected.iloc[i]['Avg_Std_Dev']
        
        sensor_orig = results_original.iloc[i]['Sensor']
        std_orig = results_original.iloc[i]['Avg_Std_Dev']
        
        print(f"{i+1:<6} {sensor_corr:<20} {std_corr:<12.4f} {sensor_orig:<20} {std_orig:<12.4f}")
    
    # Comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_20_corr = results_corrected.head(20)
    top_20_orig = results_original.head(20)
    
    x = np.arange(len(top_20_corr))
    width = 0.35
    
    ax.barh(x - width/2, top_20_corr['Avg_Std_Dev'], width, 
            label='With Drift Correction', color='green', alpha=0.7)
    ax.barh(x + width/2, top_20_orig['Avg_Std_Dev'], width,
            label='Without Drift Correction', color='blue', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_20_corr['Sensor'])
    ax.set_xlabel('Average Rank Std Dev (lower = better)', fontweight='bold')
    ax.set_ylabel('Sensor', fontweight='bold')
    ax.set_title('Impact of Drift Correction on Ranking Consistency', fontweight='bold', fontsize=14)
    ax.legend()
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "drift_correction_impact.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Comparison plot saved")
    
    avg_improvement = results_original['Avg_Std_Dev'].mean() - results_corrected['Avg_Std_Dev'].mean()
    print(f"\n💡 Average improvement in consistency: {avg_improvement:.4f}")
    if avg_improvement > 0:
        print("   ✓ Drift correction IMPROVED ranking consistency")
    else:
        print("   ⚠️ Drift correction did not improve consistency significantly")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  With drift correction:")
    print("    - food_ranking_with_drift_correction.xlsx")
    print("    - ranking_consistency_with_drift_correction.png")
    print("    - ranking_std_heatmap_all_sensors_with_drift_correction.png (ALL sensors)")
    print("    - ranking_stability_with_drift_correction.png")
    print("  Without drift correction:")
    print("    - food_ranking_no_drift_correction.xlsx")
    print("    - ranking_consistency_no_drift_correction.png")
    print("    - ranking_std_heatmap_all_sensors_no_drift_correction.png (ALL sensors)")
    print("    - ranking_stability_no_drift_correction.png")
    print("  Comparison:")
    print("    - drift_correction_comparison.png (before/after drift)")
    print("    - drift_correction_impact.png (consistency comparison)")


if __name__ == "__main__":
    analyze_with_and_without_drift_correction()