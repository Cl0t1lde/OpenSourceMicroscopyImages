# Re-run the area detection function and then plot the histogram

from PIL import Image
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

# Define the function
def find_enclosed_areas_per_region(image_path, red_threshold=125):
    image = Image.open(image_path).convert('RGB')
    data = np.array(image)

    # Step 1: Identify red pixels (tolerance-based)
    red_mask = (data[:, :, 0] > 200) & (data[:, :, 1] < red_threshold) & (data[:, :, 2] < red_threshold)

    # Step 2: Invert red mask -> non-red regions (black or in-between areas)
    non_red_mask = ~red_mask

    # Step 3: Label connected non-red components
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(non_red_mask, structure=structure)

    # Step 4: Detect and exclude border-touching components
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))

    # Step 5: Measure area for each enclosed region
    region_areas = []
    for label_id in range(1, num_features + 1):
        if label_id not in border_labels:
            area = np.sum(labeled == label_id)
            region_areas.append(area)

    return region_areas

# Apply function to the image
image_path = "manual-trace-5-1.png"
areas = find_enclosed_areas_per_region(image_path)

# Plot the distribution of region areas
plt.figure(figsize=(10, 6))
plt.hist(areas, bins=30, color='lightcoral', edgecolor='black')
plt.title('Distribution of Enclosed Region Areas')
plt.xlabel('Area (pixels)')
plt.ylabel('Number of Regions')
plt.grid(True)
plt.tight_layout()
plt.show()