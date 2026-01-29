import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector
from pathlib import Path
import json
import pandas as pd

# Path to root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Folder Locations
SCENE_IMAGES_FOLDER = ROOT_DIR / "data" / "Tray view"
DATA_FILE = ROOT_DIR / "data" / "processed" / "processed_dataset.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "food_regions_per_scene.json"

# Define your food items
FOOD_ITEMS = ["Orange", "Milk", "Bread", "Bacon", "Cabbage"]

# Storage: {scene: {food: coords}}
all_scenes_food_regions = {}
current_coords = None

def onselect(eclick, erelease):
    """Callback for RectangleSelector."""
    global current_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    current_coords = {
        'x_start': min(x1, x2),
        'x_end': max(x1, x2),
        'y_start': min(y1, y2),
        'y_end': max(y1, y2)
    }
    print(f"Selected area: X[{current_coords['x_start']}:{current_coords['x_end']}], Y[{current_coords['y_start']}:{current_coords['y_end']}]")

def select_food_regions_for_all_scenes():
    """Let user select rectangular regions for each food item in EACH scene."""
    global current_coords
    
    # Load processed data to get unique scenes
    df = pd.read_csv(DATA_FILE)
    unique_scenes = sorted(df["Scene_ID"].unique())
    
    print("\n" + "="*70)
    print(f"Found {len(unique_scenes)} scenes. You'll select food regions for each.")
    print("="*70 + "\n")
    
    # For each scene
    for scene in unique_scenes:
        print(f"\n{'='*70}")
        print(f"SCENE: {scene}")
        print(f"{'='*70}")
        
        # Extract scene number (e.g., "S7" from "5F_S7-TR-012326160254.csv")
        scene_num = None
        if "_S" in scene:
            parts = scene.split("_S")
            if len(parts) > 1:
                scene_num = "S" + parts[1].split("-")[0]
        
        if not scene_num:
            print(f"⚠️  Could not extract scene number from {scene}. Skipping.")
            continue
        
        # Load the corresponding tray image
        scene_image_path = SCENE_IMAGES_FOLDER / f"T_5F_{scene_num}.png"
        
        if not scene_image_path.exists():
            print(f"⚠️  Image not found: {scene_image_path}. Skipping.")
            continue
        
        img = mpimg.imread(scene_image_path)
        print(f"Using image: {scene_image_path.name}\n")
        
        # Storage for this scene's food regions
        scene_food_regions = {}
        
        # For each food item
        for food in FOOD_ITEMS:
            current_coords = None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.imshow(img)
            ax.set_title(f"SCENE: {scene_num} | Draw rectangle around: {food.upper()}\n(Close window when done)", 
                        fontsize=14, fontweight='bold')
            
            rs = RectangleSelector(ax, onselect, useblit=True,
                                  button=[1], minspanx=5, minspany=5,
                                  spancoords='pixels', interactive=True)
            
            plt.show()
            
            if current_coords:
                scene_food_regions[food] = current_coords
                print(f"  ✓ {food} region saved for {scene_num}")
            else:
                print(f"  ✗ No region selected for {food} in {scene_num}")
        
        # Save this scene's regions
        all_scenes_food_regions[scene] = scene_food_regions
        print(f"\n✓ Completed {scene_num}: {len(scene_food_regions)}/{len(FOOD_ITEMS)} foods marked\n")
    
    # Save all regions to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_scenes_food_regions, f, indent=4)
    
    print("\n" + "="*70)
    print(f"✅ All food regions saved to: {OUTPUT_FILE}")
    print(f"Total scenes processed: {len(all_scenes_food_regions)}")
    print("="*70)

if __name__ == "__main__":
    select_food_regions_for_all_scenes()