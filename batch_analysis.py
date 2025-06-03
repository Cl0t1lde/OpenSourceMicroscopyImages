from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
import glob

def find_optimal_dilation(manual_bin, connected_bin, max_iterations=10):
    """Find the optimal dilation factor that maximizes Dice score with manual trace."""
    best_dice = 0
    best_iterations = 0
    dice_scores = []
    
    for iterations in range(0, max_iterations + 1):
        if iterations == 0:
            connected_dilated = connected_bin
        else:
            connected_dilated = binary_dilation(connected_bin, iterations=iterations).astype(np.uint8)
        
        # Calculate Dice coefficient
        intersection = np.sum(manual_bin * connected_dilated)
        dice = 2. * intersection / (np.sum(manual_bin) + np.sum(connected_dilated))
        dice_scores.append(dice)
        
        if dice > best_dice:
            best_dice = dice
            best_iterations = iterations
    
    return best_iterations, best_dice, dice_scores

def analyze_image_trio(manual_path, ilastik_path, connected_path, pair_name):
    """Analyze a trio of manual trace, ilastik segmentation, and connected skeleton images."""
    
    # Load images
    manual_img = Image.open(manual_path).convert('RGB')
    ilastik_img = Image.open(ilastik_path).convert('L')
    connected_img = Image.open(connected_path).convert('L')
    
    # Convert to numpy arrays
    manual_array = np.array(manual_img)
    ilastik_array = np.array(ilastik_img)
    connected_array = np.array(connected_img)
    
    print(f"\n--- Analyzing {pair_name} ---")
    print(f"Manual trace: {manual_path}")
    print(f"Ilastik segmentation: {ilastik_path}")
    print(f"Connected skeleton: {connected_path}")
    print("Shape:", manual_array.shape)
    print("Red channel unique values:", np.unique(manual_array[:,:,0]))
    print("Green channel unique values:", np.unique(manual_array[:,:,1]))
    print("Blue channel unique values:", np.unique(manual_array[:,:,2]))
    print("Unique values in ilastik segmentation:", np.unique(ilastik_array))
    print("Unique values in connected skeleton:", np.unique(connected_array))
    
    # Create binary masks for pure red shades only
    manual_bin = (
        (manual_array[:, :, 0] > 10) &  # Red channel should be significant
        (manual_array[:, :, 1] < 127) &   # Green should be 0
        (manual_array[:, :, 2] < 127)     # Blue should be 0
    ).astype(np.uint8)
    
    # Decide on the number of dilation iterations for manual trace based on filename
    if "6" in os.path.basename(manual_path): #6 is in the filename of the zoomed pictures.
        manual_dilation_iterations = 5
    elif "7" in os.path.basename(manual_path): #7 is in the filename of the a bit less zoomed pictures.
        manual_dilation_iterations = 4
    else:
        manual_dilation_iterations = 2
    
    # Thicken the manual trace
    manual_bin_dilated = binary_dilation(manual_bin, iterations=manual_dilation_iterations).astype(np.uint8)
    
    # Create binary mask for connected skeleton
    connected_bin = (connected_array > 128).astype(np.uint8)
    
    # Find optimal dilation for connected skeleton
    print("Finding optimal dilation for connected skeleton...")
    optimal_iterations, optimal_dice, dice_progression = find_optimal_dilation(
        manual_bin_dilated, connected_bin, max_iterations=15
    )
    
    print(f"Optimal dilation: {optimal_iterations} iterations (Dice: {optimal_dice:.4f})")
    
    # Apply optimal dilation
    if optimal_iterations == 0:
        connected_bin_optimal = connected_bin
    else:
        connected_bin_optimal = binary_dilation(connected_bin, iterations=optimal_iterations).astype(np.uint8)
    
    # Also keep the fixed dilation of 2 for comparison
    connected_bin_dilated = binary_dilation(connected_bin, iterations=2).astype(np.uint8)
    
    # Create ilastik label masks
    all_3_mask = (ilastik_array == 0).astype(np.uint8)
    cell_wall_mask = (ilastik_array == 1).astype(np.uint8)
    cytoplasm_mask = (ilastik_array == 2).astype(np.uint8)
    extracellular_mask = (ilastik_array == 3).astype(np.uint8)
    
    # Calculate Dice coefficients between manual and ilastik
    intersection_original = np.sum(manual_bin * cell_wall_mask)
    dice_manual_ilastik_orig = 2. * intersection_original / (np.sum(manual_bin) + np.sum(cell_wall_mask))
    
    intersection_dilated = np.sum(manual_bin_dilated * cell_wall_mask)
    dice_manual_ilastik_dil = 2. * intersection_dilated / (np.sum(manual_bin_dilated) + np.sum(cell_wall_mask))
    
    # Calculate Dice coefficients between manual and connected skeleton (original)
    intersection_connected_orig = np.sum(manual_bin * connected_bin)
    dice_manual_connected_orig = 2. * intersection_connected_orig / (np.sum(manual_bin) + np.sum(connected_bin))
    
    # Calculate Dice coefficients between manual and connected skeleton (fixed dilation)
    intersection_connected_dil = np.sum(manual_bin_dilated * connected_bin_dilated)
    dice_manual_connected_dil = 2. * intersection_connected_dil / (np.sum(manual_bin_dilated) + np.sum(connected_bin_dilated))
    
    # Calculate Dice coefficients between manual and connected skeleton (optimal dilation)
    intersection_connected_opt = np.sum(manual_bin_dilated * connected_bin_optimal)
    dice_manual_connected_opt = 2. * intersection_connected_opt / (np.sum(manual_bin_dilated) + np.sum(connected_bin_optimal))
    
    # Calculate Dice coefficients between connected skeleton and ilastik
    intersection_conn_ilastik = np.sum(connected_bin_optimal * cell_wall_mask)
    dice_connected_ilastik = 2. * intersection_conn_ilastik / (np.sum(connected_bin_optimal) + np.sum(cell_wall_mask))
    
    print(f'Dice coefficient (Manual vs Ilastik - original): {dice_manual_ilastik_orig:.4f}')
    print(f'Dice coefficient (Manual vs Ilastik - dilated): {dice_manual_ilastik_dil:.4f}')
    print(f'Dice coefficient (Manual vs Connected - original): {dice_manual_connected_orig:.4f}')
    print(f'Dice coefficient (Manual vs Connected - fixed dilation=2): {dice_manual_connected_dil:.4f}')
    print(f'Dice coefficient (Manual vs Connected - optimal dilation={optimal_iterations}): {dice_manual_connected_opt:.4f}')
    print(f'Dice coefficient (Connected vs Ilastik): {dice_connected_ilastik:.4f}')
    print(f'Improvement from optimal vs fixed dilation: {dice_manual_connected_opt - dice_manual_connected_dil:+.4f}')
    
    # Create enhanced visualizations
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f'Analysis for {pair_name} (Optimal dilation: {optimal_iterations})', fontsize=16)
    
    # Row 1: Original images
    axes[0, 0].imshow(manual_array)
    axes[0, 0].set_title('Original Manual Trace')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ilastik_array, cmap='viridis')
    axes[0, 1].set_title('Ilastik Segmentation')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(connected_array, cmap='gray')
    axes[0, 2].set_title('Connected Skeleton')
    axes[0, 2].axis('off')
    
    # Dilation optimization curve
    axes[0, 3].plot(range(len(dice_progression)), dice_progression, 'b-o', markersize=4)
    axes[0, 3].axvline(x=optimal_iterations, color='red', linestyle='--', label=f'Optimal: {optimal_iterations}')
    axes[0, 3].axvline(x=2, color='orange', linestyle='--', label='Fixed: 2')
    axes[0, 3].set_xlabel('Dilation Iterations')
    axes[0, 3].set_ylabel('Dice Score')
    axes[0, 3].set_title('Dilation Optimization')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Binary masks comparison
    axes[1, 0].imshow(manual_bin_dilated, cmap='gray')
    axes[1, 0].set_title(f'Manual Trace (dil={manual_dilation_iterations})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(connected_bin_dilated, cmap='gray')
    axes[1, 1].set_title(f'Connected (fixed dil=2)\nDice: {dice_manual_connected_dil:.4f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(connected_bin_optimal, cmap='gray')
    axes[1, 2].set_title(f'Connected (optimal dil={optimal_iterations})\nDice: {dice_manual_connected_opt:.4f}')
    axes[1, 2].axis('off')
    
    # Comparison: Fixed vs Optimal
    axes[1, 3].imshow(connected_bin_dilated, cmap='Reds', alpha=0.7, label='Fixed dil=2')
    axes[1, 3].imshow(connected_bin_optimal, cmap='Blues', alpha=0.7, label=f'Optimal dil={optimal_iterations}')
    axes[1, 3].set_title('Fixed (Red) vs Optimal (Blue)')
    axes[1, 3].axis('off')
    
    # Row 3: Overlays with manual
    axes[2, 0].imshow(manual_bin_dilated, cmap='Greens', alpha=0.7)
    axes[2, 0].imshow(connected_bin_dilated, cmap='Reds', alpha=0.5)
    axes[2, 0].set_title(f'Manual + Fixed Dilation\nDice: {dice_manual_connected_dil:.4f}')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(manual_bin_dilated, cmap='Greens', alpha=0.7)
    axes[2, 1].imshow(connected_bin_optimal, cmap='Blues', alpha=0.5)
    axes[2, 1].set_title(f'Manual + Optimal Dilation\nDice: {dice_manual_connected_opt:.4f}')
    axes[2, 1].axis('off')
    
    # Difference visualization
    diff_fixed = np.abs(manual_bin_dilated.astype(float) - connected_bin_dilated.astype(float))
    diff_optimal = np.abs(manual_bin_dilated.astype(float) - connected_bin_optimal.astype(float))
    
    axes[2, 2].imshow(diff_fixed, cmap='Reds', alpha=0.8)
    axes[2, 2].set_title(f'Differences (Fixed)\nError pixels: {np.sum(diff_fixed):.0f}')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(diff_optimal, cmap='Blues', alpha=0.8)
    axes[2, 3].set_title(f'Differences (Optimal)\nError pixels: {np.sum(diff_optimal):.0f}')
    axes[2, 3].axis('off')
    
    # Row 4: Performance comparison
    methods = ['Ilastik', 'Connected\n(fixed=2)', f'Connected\n(opt={optimal_iterations})']
    dice_scores = [dice_manual_ilastik_dil, dice_manual_connected_dil, dice_manual_connected_opt]
    colors = ['orange', 'red', 'blue']
    
    bars = axes[3, 0].bar(methods, dice_scores, color=colors, alpha=0.7)
    axes[3, 0].set_ylabel('Dice Score')
    axes[3, 0].set_title('Performance Comparison')
    axes[3, 0].set_ylim(0, 1)
    for i, (bar, score) in enumerate(zip(bars, dice_scores)):
        axes[3, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{score:.3f}', ha='center', va='bottom')
    
    # Improvement summary
    improvement_text = f"""
DILATION OPTIMIZATION RESULTS:

Fixed dilation (2 iterations):
  Dice score: {dice_manual_connected_dil:.4f}

Optimal dilation ({optimal_iterations} iterations):
  Dice score: {dice_manual_connected_opt:.4f}
  
Improvement: {dice_manual_connected_opt - dice_manual_connected_dil:+.4f}

vs Ilastik improvement:
  Fixed: {dice_manual_connected_dil - dice_manual_ilastik_dil:+.4f}
  Optimal: {dice_manual_connected_opt - dice_manual_ilastik_dil:+.4f}
"""
    
    axes[3, 1].text(0.1, 0.5, improvement_text, transform=axes[3, 1].transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[3, 1].set_title('Optimization Summary')
    axes[3, 1].axis('off')
    
    # Hide unused subplots
    axes[3, 2].axis('off')
    axes[3, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'dice_manual_ilastik_orig': dice_manual_ilastik_orig,
        'dice_manual_ilastik_dil': dice_manual_ilastik_dil,
        'dice_manual_connected_orig': dice_manual_connected_orig,
        'dice_manual_connected_dil': dice_manual_connected_dil,
        'dice_manual_connected_opt': dice_manual_connected_opt,
        'dice_connected_ilastik': dice_connected_ilastik,
        'optimal_iterations': optimal_iterations,
        'optimal_dice': optimal_dice,
        'dice_progression': dice_progression
    }

def main():
    """Main function to process all image trios."""
    
    # Find all manual trace files
    manual_pattern = "manual_trace/manual-trace-*.png"
    manual_files = glob.glob(manual_pattern)
    
    results = []
    
    for manual_file in manual_files:
        try:
            # Extract the X-Y part from manual-trace-X-Y.png
            identifier = manual_file.replace("manual_trace/manual-trace-", "").replace(".png", "")
            
            # Look for corresponding files
            ilastik_file = f"simple_segmentation/simple-segmentation-{identifier}.png"
            connected_file = f"connected_skeletons/{identifier}_connected.png"
            
            if os.path.exists(ilastik_file) and os.path.exists(connected_file):
                # Create pair name
                pair_name = f"Pair_{identifier}"
                
                # Analyze the trio
                dice_results = analyze_image_trio(manual_file, ilastik_file, connected_file, pair_name)
                
                results.append({
                    'pair': pair_name,
                    'manual_file': manual_file,
                    'ilastik_file': ilastik_file,
                    'connected_file': connected_file,
                    **dice_results
                })
            else:
                missing_files = []
                if not os.path.exists(ilastik_file):
                    missing_files.append(f"Ilastik: {ilastik_file}")
                if not os.path.exists(connected_file):
                    missing_files.append(f"Connected: {connected_file}")
                print(f"Missing files for {manual_file}: {', '.join(missing_files)}")
            
        except Exception as e:
            print(f"Error processing {manual_file}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL ANALYSES")
    print("="*80)
    print(f"{'Pair':15} | {'M-I Orig':10} | {'M-I Dil':10} | {'M-C Orig':10} | {'M-C Dil':10} | {'C-I':10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['pair']:15} | "
              f"{result['dice_manual_ilastik_orig']:8.4f} | "
              f"{result['dice_manual_ilastik_dil']:8.4f} | "
              f"{result['dice_manual_connected_orig']:8.4f} | "
              f"{result['dice_manual_connected_dil']:8.4f} | "
              f"{result['dice_connected_ilastik']:8.4f}")
    
    if results:
        avg_m_i_orig = np.mean([r['dice_manual_ilastik_orig'] for r in results])
        avg_m_i_dil = np.mean([r['dice_manual_ilastik_dil'] for r in results])
        avg_m_c_orig = np.mean([r['dice_manual_connected_orig'] for r in results])
        avg_m_c_dil = np.mean([r['dice_manual_connected_dil'] for r in results])
        avg_c_i = np.mean([r['dice_connected_ilastik'] for r in results])
        
        print("-"*80)
        print(f"{'Average':15} | "
              f"{avg_m_i_orig:8.4f} | "
              f"{avg_m_i_dil:8.4f} | "
              f"{avg_m_c_orig:8.4f} | "
              f"{avg_m_c_dil:8.4f} | "
              f"{avg_c_i:8.4f}")
        
        print("\nLegend:")
        print("M-I Orig: Manual vs Ilastik (original thickness)")
        print("M-I Dil:  Manual vs Ilastik (dilated)")
        print("M-C Orig: Manual vs Connected (original thickness)")
        print("M-C Dil:  Manual vs Connected (dilated)")
        print("C-I:      Connected vs Ilastik")
        
        print(f"\nKey insights:")
        print(f"Manual vs Connected performance: {avg_m_c_dil:.4f}")
        print(f"Improvement over raw Ilastik: {avg_m_c_dil - avg_m_i_dil:.4f}")

if __name__ == "__main__":
    main()