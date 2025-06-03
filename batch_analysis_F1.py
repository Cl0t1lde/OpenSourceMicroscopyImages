from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
import glob

def calculate_metrics(manual_mask, predicted_mask):
    """Calculate comprehensive metrics including FP, FN, precision, recall, specificity."""
    # True Positives: pixels that are 1 in both masks
    TP = np.sum((manual_mask == 1) & (predicted_mask == 1))
    
    # False Positives: pixels that are 0 in manual but 1 in predicted (overcorrection)
    FP = np.sum((manual_mask == 0) & (predicted_mask == 1))
    
    # False Negatives: pixels that are 1 in manual but 0 in predicted (undercorrection)
    FN = np.sum((manual_mask == 1) & (predicted_mask == 0))
    
    # True Negatives: pixels that are 0 in both masks
    TN = np.sum((manual_mask == 0) & (predicted_mask == 0))
    
    # Calculate derived metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # How much of predicted is correct
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0     # How much of manual is captured
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # How well we avoid false positives
    
    # F1 score (harmonic mean of precision and recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dice coefficient
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    
    # Calculate rates as percentages of manual trace
    manual_pixels = np.sum(manual_mask == 1)
    fp_rate = (FP / manual_pixels * 100) if manual_pixels > 0 else 0  # Overcorrection rate
    fn_rate = (FN / manual_pixels * 100) if manual_pixels > 0 else 0  # Undercorrection rate
    
    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'f1_score': f1_score, 'dice': dice,
        'fp_rate': fp_rate, 'fn_rate': fn_rate,
        'manual_pixels': manual_pixels
    }

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
    
    # Create binary masks for pure red shades only
    manual_bin = (
        (manual_array[:, :, 0] > 10) &  # Red channel should be significant
        (manual_array[:, :, 1] < 127) &   # Green should be 0
        (manual_array[:, :, 2] < 127)     # Blue should be 0
    ).astype(np.uint8)
    
    # Thicken the manual trace by 2 iterations
    manual_bin_dilated = binary_dilation(manual_bin, iterations=2).astype(np.uint8)
    
    # Create binary mask for connected skeleton
    connected_bin = (connected_array > 128).astype(np.uint8)
    
    # Thicken the connected skeleton by 2 iterations to match manual trace thickness
    connected_bin_dilated = binary_dilation(connected_bin, iterations=4).astype(np.uint8)
    
    # Create ilastik label masks
    cell_wall_mask = (ilastik_array == 1).astype(np.uint8)
    
    # Calculate comprehensive metrics
    print("\nCalculating metrics...")
    
    # Manual vs Ilastik (original)
    metrics_mi_orig = calculate_metrics(manual_bin, cell_wall_mask)
    
    # Manual vs Ilastik (dilated)
    metrics_mi_dil = calculate_metrics(manual_bin_dilated, cell_wall_mask)
    
    # Manual vs Connected (original)
    metrics_mc_orig = calculate_metrics(manual_bin, connected_bin)
    
    # Manual vs Connected (dilated) - THIS IS THE KEY COMPARISON
    metrics_mc_dil = calculate_metrics(manual_bin_dilated, connected_bin_dilated)
    
    # Connected vs Ilastik
    metrics_ci = calculate_metrics(connected_bin_dilated, cell_wall_mask)
    
    # Print detailed results
    print(f"\nManual pixels: {metrics_mc_dil['manual_pixels']}")
    print(f"\nManual vs Connected (dilated) - KEY METRICS:")
    print(f"  True Positives:  {metrics_mc_dil['TP']:6d} pixels")
    print(f"  False Positives: {metrics_mc_dil['FP']:6d} pixels (OVERCORRECTION: {metrics_mc_dil['fp_rate']:5.1f}% of manual)")
    print(f"  False Negatives: {metrics_mc_dil['FN']:6d} pixels (UNDERCORRECTION: {metrics_mc_dil['fn_rate']:5.1f}% of manual)")
    print(f"  Precision: {metrics_mc_dil['precision']:.4f} (how accurate are our connections)")
    print(f"  Recall:    {metrics_mc_dil['recall']:.4f} (how complete are our connections)")
    print(f"  F1 Score:  {metrics_mc_dil['f1_score']:.4f}")
    print(f"  Dice:      {metrics_mc_dil['dice']:.4f}")
    
    # Create enhanced visualizations
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f'Analysis for {pair_name}', fontsize=16)
    
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
    
    # Overlay: Manual + Connected
    axes[0, 3].imshow(connected_array, cmap='gray', alpha=0.7)
    axes[0, 3].imshow(manual_array, alpha=0.5)
    axes[0, 3].set_title('Overlay: Manual + Connected')
    axes[0, 3].axis('off')
    
    # Row 2: Binary masks
    axes[1, 0].imshow(manual_bin_dilated, cmap='gray')
    axes[1, 0].set_title('Manual Trace (Dilated)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cell_wall_mask, cmap='gray')
    axes[1, 1].set_title('Ilastik Cell Wall Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(connected_bin_dilated, cmap='gray')
    axes[1, 2].set_title('Connected Skeleton (Dilated)')
    axes[1, 2].axis('off')
    
    # True Positives (correct connections)
    tp_mask = (manual_bin_dilated == 1) & (connected_bin_dilated == 1)
    axes[1, 3].imshow(tp_mask, cmap='Greens')
    axes[1, 3].set_title(f'True Positives\n{metrics_mc_dil["TP"]} pixels')
    axes[1, 3].axis('off')
    
    # Row 3: Error analysis
    # False Positives (overcorrection - connected where manual isn't)
    fp_mask = (manual_bin_dilated == 0) & (connected_bin_dilated == 1)
    axes[2, 0].imshow(fp_mask, cmap='Reds')
    axes[2, 0].set_title(f'False Positives (Overcorrection)\n{metrics_mc_dil["FP"]} pixels ({metrics_mc_dil["fp_rate"]:.1f}%)')
    axes[2, 0].axis('off')
    
    # False Negatives (undercorrection - manual where connected isn't)
    fn_mask = (manual_bin_dilated == 1) & (connected_bin_dilated == 0)
    axes[2, 1].imshow(fn_mask, cmap='Blues')
    axes[2, 1].set_title(f'False Negatives (Undercorrection)\n{metrics_mc_dil["FN"]} pixels ({metrics_mc_dil["fn_rate"]:.1f}%)')
    axes[2, 1].axis('off')
    
    # Combined error visualization
    error_vis = np.zeros((manual_bin_dilated.shape[0], manual_bin_dilated.shape[1], 3))
    error_vis[tp_mask, 1] = 1.0  # Green for correct
    error_vis[fp_mask, 0] = 1.0  # Red for overcorrection
    error_vis[fn_mask, 2] = 1.0  # Blue for undercorrection
    axes[2, 2].imshow(error_vis)
    axes[2, 2].set_title('Error Analysis\nGreen=Correct, Red=Over, Blue=Under')
    axes[2, 2].axis('off')
    
    # Metrics comparison bar chart
    categories = ['Precision', 'Recall', 'F1 Score', 'Dice']
    ilastik_values = [metrics_mi_dil['precision'], metrics_mi_dil['recall'], 
                     metrics_mi_dil['f1_score'], metrics_mi_dil['dice']]
    connected_values = [metrics_mc_dil['precision'], metrics_mc_dil['recall'], 
                       metrics_mc_dil['f1_score'], metrics_mc_dil['dice']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[2, 3].bar(x - width/2, ilastik_values, width, label='Ilastik', alpha=0.7, color='orange')
    axes[2, 3].bar(x + width/2, connected_values, width, label='Connected', alpha=0.7, color='blue')
    axes[2, 3].set_ylabel('Score')
    axes[2, 3].set_title('Performance Comparison')
    axes[2, 3].set_xticks(x)
    axes[2, 3].set_xticklabels(categories, rotation=45)
    axes[2, 3].legend()
    axes[2, 3].set_ylim(0, 1)
    
    # Row 4: Rate analysis
    # Overcorrection vs Undercorrection rates
    methods = ['Ilastik', 'Connected']
    fp_rates = [metrics_mi_dil['fp_rate'], metrics_mc_dil['fp_rate']]
    fn_rates = [metrics_mi_dil['fn_rate'], metrics_mc_dil['fn_rate']]
    
    x = np.arange(len(methods))
    axes[3, 0].bar(x, fp_rates, alpha=0.7, color='red', label='Overcorrection %')
    axes[3, 0].set_title('Overcorrection Rate\n(% of manual trace)')
    axes[3, 0].set_ylabel('% of Manual Pixels')
    axes[3, 0].set_xticks(x)
    axes[3, 0].set_xticklabels(methods)
    axes[3, 0].legend()
    
    axes[3, 1].bar(x, fn_rates, alpha=0.7, color='blue', label='Undercorrection %')
    axes[3, 1].set_title('Undercorrection Rate\n(% of manual trace)')
    axes[3, 1].set_ylabel('% of Manual Pixels')
    axes[3, 1].set_xticks(x)
    axes[3, 1].set_xticklabels(methods)
    axes[3, 1].legend()
    
    # Net error (FP + FN as percentage of manual)
    net_errors = [fp_rates[i] + fn_rates[i] for i in range(len(methods))]
    axes[3, 2].bar(x, net_errors, alpha=0.7, color='purple')
    axes[3, 2].set_title('Total Error Rate\n(Over + Under correction)')
    axes[3, 2].set_ylabel('% of Manual Pixels')
    axes[3, 2].set_xticks(x)
    axes[3, 2].set_xticklabels(methods)
    
    # Improvement summary
    improvement_text = f"""
    IMPROVEMENT ANALYSIS:
    
    Precision: {metrics_mc_dil['precision'] - metrics_mi_dil['precision']:+.3f}
    Recall: {metrics_mc_dil['recall'] - metrics_mi_dil['recall']:+.3f}
    F1 Score: {metrics_mc_dil['f1_score'] - metrics_mi_dil['f1_score']:+.3f}
    
    Overcorrection: {metrics_mc_dil['fp_rate'] - metrics_mi_dil['fp_rate']:+.1f}%
    Undercorrection: {metrics_mc_dil['fn_rate'] - metrics_mi_dil['fn_rate']:+.1f}%
    """
    
    axes[3, 3].text(0.1, 0.5, improvement_text, transform=axes[3, 3].transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[3, 3].set_title('Improvement Summary')
    axes[3, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'metrics_mi_orig': metrics_mi_orig,
        'metrics_mi_dil': metrics_mi_dil,
        'metrics_mc_orig': metrics_mc_orig,
        'metrics_mc_dil': metrics_mc_dil,
        'metrics_ci': metrics_ci
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
                metrics_results = analyze_image_trio(manual_file, ilastik_file, connected_file, pair_name)
                
                results.append({
                    'pair': pair_name,
                    'manual_file': manual_file,
                    'ilastik_file': ilastik_file,
                    'connected_file': connected_file,
                    **metrics_results
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
    
    # Print comprehensive summary
    print("\n" + "="*120)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*120)
    print(f"{'Pair':12} | {'Method':10} | {'Precision':9} | {'Recall':9} | {'F1':9} | {'Dice':9} | {'Over%':7} | {'Under%':7}")
    print("-"*120)
    
    for result in results:
        # Ilastik results
        mi = result['metrics_mi_dil']
        print(f"{result['pair']:12} | {'Ilastik':10} | "
              f"{mi['precision']:8.4f} | {mi['recall']:8.4f} | {mi['f1_score']:8.4f} | {mi['dice']:8.4f} | "
              f"{mi['fp_rate']:6.1f} | {mi['fn_rate']:6.1f}")
        
        # Connected results
        mc = result['metrics_mc_dil']
        print(f"{result['pair']:12} | {'Connected':10} | "
              f"{mc['precision']:8.4f} | {mc['recall']:8.4f} | {mc['f1_score']:8.4f} | {mc['dice']:8.4f} | "
              f"{mc['fp_rate']:6.1f} | {mc['fn_rate']:6.1f}")
        print()
    
    if results:
        # Calculate averages
        avg_metrics = {}
        for method in ['mi_dil', 'mc_dil']:
            avg_metrics[method] = {}
            for metric in ['precision', 'recall', 'f1_score', 'dice', 'fp_rate', 'fn_rate']:
                values = [r[f'metrics_{method}'][metric] for r in results]
                avg_metrics[method][metric] = np.mean(values)
        
        print("-"*120)
        print(f"{'AVERAGE':12} | {'Ilastik':10} | "
              f"{avg_metrics['mi_dil']['precision']:8.4f} | {avg_metrics['mi_dil']['recall']:8.4f} | "
              f"{avg_metrics['mi_dil']['f1_score']:8.4f} | {avg_metrics['mi_dil']['dice']:8.4f} | "
              f"{avg_metrics['mi_dil']['fp_rate']:6.1f} | {avg_metrics['mi_dil']['fn_rate']:6.1f}")
        
        print(f"{'AVERAGE':12} | {'Connected':10} | "
              f"{avg_metrics['mc_dil']['precision']:8.4f} | {avg_metrics['mc_dil']['recall']:8.4f} | "
              f"{avg_metrics['mc_dil']['f1_score']:8.4f} | {avg_metrics['mc_dil']['dice']:8.4f} | "
              f"{avg_metrics['mc_dil']['fp_rate']:6.1f} | {avg_metrics['mc_dil']['fn_rate']:6.1f}")
        
        print("\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        print(f"Precision improvement: {avg_metrics['mc_dil']['precision'] - avg_metrics['mi_dil']['precision']:+.4f}")
        print(f"Recall improvement:    {avg_metrics['mc_dil']['recall'] - avg_metrics['mi_dil']['recall']:+.4f}")
        print(f"F1 Score improvement:  {avg_metrics['mc_dil']['f1_score'] - avg_metrics['mi_dil']['f1_score']:+.4f}")
        print(f"Overcorrection change: {avg_metrics['mc_dil']['fp_rate'] - avg_metrics['mi_dil']['fp_rate']:+.1f}%")
        print(f"Undercorrection change: {avg_metrics['mc_dil']['fn_rate'] - avg_metrics['mi_dil']['fn_rate']:+.1f}%")

if __name__ == "__main__":
    main()