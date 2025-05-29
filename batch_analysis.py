from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
import glob

def analyze_image_pair(manual_path, ilastik_path, pair_name):
    """Analyze a single pair of manual trace and ilastik segmentation images."""
    
    # Load images
    manual_img = Image.open(manual_path).convert('RGB')
    ilastik_img = Image.open(ilastik_path).convert('L')
    
    # Convert to numpy arrays
    manual_array = np.array(manual_img)
    ilastik_array = np.array(ilastik_img)
    
    print(f"\n--- Analyzing {pair_name} ---")
    print(f"Manual trace: {manual_path}")
    print(f"Ilastik segmentation: {ilastik_path}")
    print("Shape:", manual_array.shape)
    print("Red channel unique values:", np.unique(manual_array[:,:,0]))
    print("Green channel unique values:", np.unique(manual_array[:,:,1]))
    print("Blue channel unique values:", np.unique(manual_array[:,:,2]))
    print("Unique values in ilastik segmentation:", np.unique(ilastik_array))
    
    # Create binary masks
    manual_bin = (
        (manual_array[:, :, 0] > 127) &
        (manual_array[:, :, 1] == 0) & 
        (manual_array[:, :, 2] == 0)
    ).astype(np.uint8)
    
    # Thicken the manual trace by 2 iterations
    manual_bin_dilated = binary_dilation(manual_bin, iterations=2).astype(np.uint8)
    
    # Create ilastik label masks
    all_3_mask = (ilastik_array == 0).astype(np.uint8)
    cell_wall_mask = (ilastik_array == 1).astype(np.uint8)
    cytoplasm_mask = (ilastik_array == 2).astype(np.uint8)
    extracellular_mask = (ilastik_array == 3).astype(np.uint8)
    
    # Calculate Dice coefficients
    intersection_original = np.sum(manual_bin * cell_wall_mask)
    dice_original = 2. * intersection_original / (np.sum(manual_bin) + np.sum(cell_wall_mask))
    
    intersection_dilated = np.sum(manual_bin_dilated * cell_wall_mask)
    dice_dilated = 2. * intersection_dilated / (np.sum(manual_bin_dilated) + np.sum(cell_wall_mask))
    
    print(f'Dice coefficient (original): {dice_original:.4f}')
    print(f'Dice coefficient (after dilation): {dice_dilated:.4f}')
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Analysis for {pair_name}', fontsize=16)
    
    # Manual trace binary mask
    axes[0, 0].imshow(manual_bin, cmap='gray')
    axes[0, 0].set_title('Manual Trace Binary Mask')
    axes[0, 0].axis('off')
    
    # Dilated manual trace
    axes[0, 1].imshow(manual_bin_dilated, cmap='gray')
    axes[0, 1].set_title('Dilated Manual Trace')
    axes[0, 1].axis('off')
    
    # Cell wall mask
    axes[0, 2].imshow(cell_wall_mask, cmap='gray')
    axes[0, 2].set_title('Cell Wall Mask (label 1)')
    axes[0, 2].axis('off')
    
    # Original manual trace image
    axes[1, 0].imshow(manual_array)
    axes[1, 0].set_title('Original Manual Trace')
    axes[1, 0].axis('off')
    
    # Ilastik segmentation
    axes[1, 1].imshow(ilastik_array, cmap='viridis')
    axes[1, 1].set_title('Ilastik Segmentation')
    axes[1, 1].axis('off')
    
    # Overlay
    manual_norm = np.array(manual_bin_dilated) / 255.0
    ilastik_norm = ilastik_array / 255.0
    
    axes[1, 2].imshow(ilastik_norm, cmap='magma', alpha=0.6, interpolation='none')
    axes[1, 2].imshow(manual_norm, cmap='viridis_r', alpha=0.5, interpolation='none')
    axes[1, 2].set_title('Overlay: Manual (yellow) & Ilastik (magenta)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return dice_original, dice_dilated

def main():
    """Main function to process all image pairs."""
    
    # Find all manual trace files
    manual_pattern = "manual-trace-*.png"
    manual_files = glob.glob(manual_pattern)
    
    results = []
    
    for manual_file in manual_files:
        try:
            # Extract the X-Y part from manual-trace-X-Y.png
            # Remove "manual-trace-" prefix and ".png" suffix
            identifier = manual_file.replace("manual-trace-", "").replace(".png", "")
            
            # Look for corresponding simple-segmentation file
            ilastik_file = f"simple-segmentation-{identifier}.png"
            
            if os.path.exists(ilastik_file):
                # Create pair name
                pair_name = f"Pair_{identifier}"
                
                # Analyze the pair
                dice_orig, dice_dil = analyze_image_pair(manual_file, ilastik_file, pair_name)
                
                results.append({
                    'pair': pair_name,
                    'manual_file': manual_file,
                    'ilastik_file': ilastik_file,
                    'dice_original': dice_orig,
                    'dice_dilated': dice_dil
                })
            else:
                print(f"No matching segmentation file found for {manual_file}")
                print(f"Expected: {ilastik_file}")
            
        except Exception as e:
            print(f"Error processing {manual_file}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL ANALYSES")
    print("="*60)
    
    for result in results:
        print(f"{result['pair']:15} | Dice Original: {result['dice_original']:.4f} | Dice Dilated: {result['dice_dilated']:.4f}")
    
    if results:
        avg_dice_orig = np.mean([r['dice_original'] for r in results])
        avg_dice_dil = np.mean([r['dice_dilated'] for r in results])
        
        print("-"*60)
        print(f"{'Average':15} | Dice Original: {avg_dice_orig:.4f} | Dice Dilated: {avg_dice_dil:.4f}")
        print(f"Improvement with dilation: {avg_dice_dil - avg_dice_orig:.4f}")

if __name__ == "__main__":
    main()