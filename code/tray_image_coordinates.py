import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os

# ================= CONFIGURATION =================
IMAGE_PATH = '/home/saklain/Downloads/5F_S1.png'
OUTPUT_COORD_FILE = '/home/saklain/Desktop/Enose/data/positions.csv'
OUTPUT_IMAGE_FILE = '/home/saklain/Desktop/Enose/data/geometry/tray_cropped.png'

# HARDCODED COORDINATES FROM YOUR FILES
# Format: Class_ID : (Global_Y, Global_X) <--- PROFESSOR SAID Y FIRST
GLOBAL_MARKERS = {
    1: (297, 423),  # L1
    2: (306, 827),  # L2
    3: (429, 624),  # L3 (Center)
    4: (544, 418),  # L4
    5: (554, 827)   # L5
}
# =================================================

def crop_and_transform():
    # 1. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        print("Please put the full image '5F_S1.png' in 'data/geometry/'")
        return

    img = mpimg.imread(IMAGE_PATH)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    
    # 2. Plot the Global Points (Visual Check)
    print("Plotting original points for verification...")
    for class_id, (g_y, g_x) in GLOBAL_MARKERS.items():
        # Matplotlib uses (X, Y) for plotting, but data is (Y, X)
        ax.plot(g_x, g_y, 'rx', markersize=12, markeredgewidth=3)
        ax.text(g_x + 10, g_y, f"L{class_id}", color='yellow', fontsize=12, fontweight='bold')

    plt.title("STEP 1: Verify Points (Red X) -> Then Click Top-Left & Bottom-Right of Tray")
    print("-" * 50)
    print("INSTRUCTIONS:")
    print("1. Look at the image. Do the Red X's match the sensor locations?")
    print("2. Click the TOP-LEFT corner of the region you want to crop.")
    print("3. Click the BOTTOM-RIGHT corner of the region.")
    print("-" * 50)

    # 3. Get User Clicks for Cropping
    clicks = plt.ginput(2, timeout=0)
    
    if len(clicks) != 2:
        print("Error: You didn't click two points.")
        return

    (x1, y1), (x2, y2) = clicks
    
    # Calculate Crop Boundaries
    crop_x_start = min(x1, x2)
    crop_y_start = min(y1, y2)
    crop_x_end = max(x1, x2)
    crop_y_end = max(y1, y2)

    print(f"Crop defined: X[{crop_x_start:.0f}:{crop_x_end:.0f}], Y[{crop_y_start:.0f}:{crop_y_end:.0f}]")

    # 4. TRANSLATE COORDINATES
    final_coords = []
    
    for class_id, (g_y, g_x) in GLOBAL_MARKERS.items():
        # TRANSLATION MATH: New = Old - Crop_Start
        l_x = g_x - crop_x_start
        l_y = g_y - crop_y_start
        
        # Check if point is inside the crop
        if not (0 <= l_x <= (crop_x_end - crop_x_start)) or not (0 <= l_y <= (crop_y_end - crop_y_start)):
            print(f"WARNING: L{class_id} is OUTSIDE your crop box!")

        final_coords.append({
            'Class': class_id,
            'Label': f"L{class_id}",
            'X_cm': round(l_x, 2),  # Keeping name 'X_cm' for compatibility, though it's pixels
            'Y_cm': round(l_y, 2)
        })

    # 5. Save Data
    df_out = pd.DataFrame(final_coords)
    df_out.to_csv(OUTPUT_COORD_FILE, index=False)
    
    # 6. Save Cropped Image
    cropped_img = img[int(crop_y_start):int(crop_y_end), int(crop_x_start):int(crop_x_end)]
    plt.imsave(OUTPUT_IMAGE_FILE, cropped_img)

    plt.close()
    
    # 7. Verification Plot
    print(f"\nSUCCESS! Local coordinates saved to {OUTPUT_COORD_FILE}")
    print(df_out)
    
    fig, ax = plt.subplots()
    ax.imshow(cropped_img)
    ax.scatter(df_out['X_cm'], df_out['Y_cm'], c='cyan', marker='o', s=100, label='Sensors')
    for i, row in df_out.iterrows():
        ax.text(row['X_cm'], row['Y_cm'], row['Label'], color='white', fontweight='bold')
    plt.title("Verification: This is your final Kriging Area")
    plt.show()

if __name__ == "__main__":
    crop_and_transform()