from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation



# Load your manual trace and ilastik segmentation map
manual_img = Image.open('bitmap_2_copy_3.png').convert('RGB')
ilastik_img = Image.open('image2_ilastik_3_trace.png').convert('L')  # or 'rgb_image.png' if that's your map


# Convert both to numpy arrays
manual_array = np.array(manual_img)
ilastik_array = np.array(ilastik_img)

print("Shape:", manual_array.shape)
print("Red channel unique values:", np.unique(manual_array[:,:,0]))
print("Green channel unique values:", np.unique(manual_array[:,:,1]))
print("Blue channel unique values:", np.unique(manual_array[:,:,2]))
print("Unique values in ilastik segmentation:", np.unique(ilastik_array))

# Create binary masks

manual_bin = (
    (manual_array[:, :, 0] == 255) & 
    (manual_array[:, :, 1] == 0) & 
    (manual_array[:, :, 2] == 0)
).astype(np.uint8)

# Thicken the manual trace (binary image) by 2 iterations
manual_bin_dilated = binary_dilation(manual_bin, iterations=2).astype(np.uint8)


# For ilastik: cell wall is label 0
# Ilastik label masks
all_3_mask = (ilastik_array == 0).astype(np.uint8)
cell_wall_mask = (ilastik_array == 1).astype(np.uint8)
cytoplasm_mask = (ilastik_array == 2).astype(np.uint8)
extracellular_mask = (ilastik_array == 3).astype(np.uint8)


# Visualize the result
plt.imshow(manual_bin, cmap='gray')
plt.title('Manual Trace Binary Mask')
plt.axis('off')
plt.show()

plt.imshow(manual_bin_dilated, cmap='gray')
plt.title('Dilated Manual Trace')
plt.axis('off')
plt.show()

# Compute new Dice coefficient with the dilated trace
intersection = np.sum(manual_bin_dilated * cell_wall_mask)
dice_dilated = 2. * intersection / (np.sum(manual_bin_dilated) + np.sum(cell_wall_mask))
print(f'Dice coefficient (after dilation): {dice_dilated:.4f}')



# For cell wall (label 0)
plt.imshow(cell_wall_mask, cmap='gray')
plt.title('Cell Wall Mask (label 1)')
plt.show()


plt.imshow(manual_array, cmap='gray', vmin=0, vmax=255)
plt.title('bitmap.png')
plt.axis('off')
plt.show()

# Normalize arrays to [0, 1] for display
manual_norm = np.array(manual_bin_dilated)  / 255.0
ilastik_norm = ilastik_array / 255.0

plt.figure(figsize=(8, 8))

# Show ilastik image in magenta
plt.imshow(ilastik_norm, cmap='magma', alpha=0.6, interpolation='none')

# Overlay manual image in green
plt.imshow(manual_norm, cmap='viridis_r', alpha=0.5, interpolation='none')

plt.title('Overlay: bitmap.png (yellow) and ilastik trace (magenta)')
plt.axis('off')
plt.show()