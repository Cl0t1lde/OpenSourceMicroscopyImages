from PIL import Image
import numpy as np
from scipy.ndimage import label, binary_dilation
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

def find_enclosed_areas_per_region(image_array, red_threshold=125):
    """Find enclosed areas in a manual trace image."""
    if len(image_array.shape) == 3:
        # RGB image - look for red pixels
        red_mask = (image_array[:, :, 0] > 200) & (image_array[:, :, 1] < red_threshold) & (image_array[:, :, 2] < red_threshold)
    else:
        # Grayscale image - assume non-zero pixels are traces
        red_mask = image_array > 128
    
    # Invert to find enclosed regions
    non_red_mask = ~red_mask
    
    # Label connected components
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(non_red_mask, structure=structure)
    
    # Exclude border-touching components
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    
    # Measure area for each enclosed region
    region_areas = []
    region_labels = []
    for label_id in range(1, num_features + 1):
        if label_id not in border_labels:
            area = np.sum(labeled == label_id)
            region_areas.append(area)
            region_labels.append(label_id)
    
    return region_areas, labeled, region_labels

def find_areas_from_segmentation(seg_array, target_label=2):
    """Find cell areas from ilastik segmentation (label 2 = cytoplasm)."""
    # Find cytoplasm regions
    cytoplasm_mask = (seg_array == target_label)
    
    # Label connected components
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(cytoplasm_mask, structure=structure)
    
    # Exclude border-touching components
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    
    # Measure area for each cell
    region_areas = []
    region_labels = []
    for label_id in range(1, num_features + 1):
        if label_id not in border_labels:
            area = np.sum(labeled == label_id)
            region_areas.append(area)
            region_labels.append(label_id)
    
    return region_areas, labeled, region_labels

def find_areas_from_skeleton(skeleton_array):
    """Find enclosed areas from skeleton by filling and analyzing."""
    # Convert skeleton to binary - skeletons have WHITE lines (255) on black background
    skeleton_bin = (skeleton_array > 128).astype(np.uint8)
    
    # Dilate the skeleton by 2 pixels to thicken lines and close small gaps
    skeleton_dilated = binary_dilation(skeleton_bin, iterations=2).astype(np.uint8)
    
    # The skeleton lines are the BOUNDARIES, not the enclosed areas
    # So we need to find the areas ENCLOSED by these white lines
    
    # Invert to find non-skeleton regions (potential enclosed areas)
    non_skeleton_mask = ~skeleton_dilated.astype(bool)
    
    # Label connected components in non-skeleton regions
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(non_skeleton_mask, structure=structure)
    
    # Exclude border-touching components (these are external areas, not cells)
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    
    # Measure area for each enclosed region
    region_areas = []
    region_labels = []
    for label_id in range(1, num_features + 1):
        if label_id not in border_labels:
            area = np.sum(labeled == label_id)
            # Filter out very small regions (noise)
            if area > 10:  # Minimum area threshold
                region_areas.append(area)
                region_labels.append(label_id)
    
    return region_areas, labeled, region_labels

def calculate_area_statistics(areas):
    """Calculate comprehensive statistics for area distributions."""
    if not areas:
        return {
            'count': 0, 'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'q25': 0, 'q75': 0,
            'total_area': 0
        }
    
    areas = np.array(areas)
    return {
        'count': len(areas),
        'mean': np.mean(areas),
        'median': np.median(areas),
        'std': np.std(areas),
        'min': np.min(areas),
        'max': np.max(areas),
        'q25': np.percentile(areas, 25),
        'q75': np.percentile(areas, 75),
        'total_area': np.sum(areas)
    }

def compare_distributions(areas1, areas2, method1_name, method2_name):
    """Compare two area distributions statistically."""
    if not areas1 or not areas2:
        return {
            'ks_statistic': np.nan,
            'ks_pvalue': np.nan,
            'mean_diff': np.nan,
            'median_diff': np.nan,
            'count_diff': len(areas1) - len(areas2) if areas1 or areas2 else 0
        }
    
    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_pval = stats.ks_2samp(areas1, areas2)
    
    stats1 = calculate_area_statistics(areas1)
    stats2 = calculate_area_statistics(areas2)
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mean_diff': stats1['mean'] - stats2['mean'],
        'median_diff': stats1['median'] - stats2['median'],
        'count_diff': stats1['count'] - stats2['count']
    }

def biological_analysis(manual_path, ilastik_path, connected_path, pair_name):
    """Perform comprehensive biological analysis comparing cell detection methods."""
    figures_dir = "biological_analysis_figures"
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n=== BIOLOGICAL ANALYSIS: {pair_name} ===")
    
    # Load images
    manual_img = Image.open(manual_path).convert('RGB')
    ilastik_img = Image.open(ilastik_path).convert('L')
    connected_img = Image.open(connected_path).convert('L')
    
    manual_array = np.array(manual_img)
    ilastik_array = np.array(ilastik_img)
    connected_array = np.array(connected_img)
    
    # Find enclosed areas for each method
    print("Analyzing manual trace...")
    manual_areas, manual_labeled, manual_labels = find_enclosed_areas_per_region(manual_array)
    
    print("Analyzing Ilastik segmentation...")
    ilastik_areas, ilastik_labeled, ilastik_labels = find_areas_from_segmentation(ilastik_array, target_label=2)
    
    print("Analyzing connected skeleton...")
    connected_areas, connected_labeled, connected_labels = find_areas_from_skeleton(connected_array)
    
    # Calculate statistics
    manual_stats = calculate_area_statistics(manual_areas)
    ilastik_stats = calculate_area_statistics(ilastik_areas)
    connected_stats = calculate_area_statistics(connected_areas)
    
    # Compare distributions
    manual_vs_ilastik = compare_distributions(manual_areas, ilastik_areas, "Manual", "Ilastik")
    manual_vs_connected = compare_distributions(manual_areas, connected_areas, "Manual", "Connected")
    ilastik_vs_connected = compare_distributions(ilastik_areas, connected_areas, "Ilastik", "Connected")
    
    # Print results
    print(f"\nCELL COUNT COMPARISON:")
    print(f"Manual trace:      {manual_stats['count']:3d} cells")
    print(f"Ilastik segm.:     {ilastik_stats['count']:3d} cells (diff: {ilastik_stats['count'] - manual_stats['count']:+3d})")
    print(f"Connected skel.:   {connected_stats['count']:3d} cells (diff: {connected_stats['count'] - manual_stats['count']:+3d})")
    
    print(f"\nCELL AREA STATISTICS (pixels):")
    print(f"{'Method':<15} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 70)
    print(f"{'Manual':<15} | {manual_stats['mean']:8.1f} | {manual_stats['median']:8.1f} | {manual_stats['std']:8.1f} | {manual_stats['min']:6.0f} | {manual_stats['max']:6.0f}")
    print(f"{'Ilastik':<15} | {ilastik_stats['mean']:8.1f} | {ilastik_stats['median']:8.1f} | {ilastik_stats['std']:8.1f} | {ilastik_stats['min']:6.0f} | {ilastik_stats['max']:6.0f}")
    print(f"{'Connected':<15} | {connected_stats['mean']:8.1f} | {connected_stats['median']:8.1f} | {connected_stats['std']:8.1f} | {connected_stats['min']:6.0f} | {connected_stats['max']:6.0f}")
    
    print(f"\nDISTRIBUTION COMPARISON (Manual as reference):")
    print(f"Manual vs Ilastik    - KS test p-value: {manual_vs_ilastik['ks_pvalue']:.4f}, Mean diff: {manual_vs_ilastik['mean_diff']:+8.1f}")
    print(f"Manual vs Connected  - KS test p-value: {manual_vs_connected['ks_pvalue']:.4f}, Mean diff: {manual_vs_connected['mean_diff']:+8.1f}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Row 1: Original images with detected regions
    ax1 = plt.subplot(4, 4, 1)
    plt.imshow(manual_array)
    plt.title(f'Manual Trace\n{manual_stats["count"]} cells detected')
    plt.axis('off')
    
    ax2 = plt.subplot(4, 4, 2)
    plt.imshow(ilastik_array, cmap='viridis')
    plt.title(f'Ilastik Segmentation\n{ilastik_stats["count"]} cells detected')
    plt.axis('off')
    
    ax3 = plt.subplot(4, 4, 3)
    plt.imshow(connected_array, cmap='gray')
    plt.title(f'Connected Skeleton\n{connected_stats["count"]} cells detected')
    plt.axis('off')
    
    # Cell count comparison
    ax4 = plt.subplot(4, 4, 4)
    methods = ['Manual', 'Ilastik', 'Connected']
    counts = [manual_stats['count'], ilastik_stats['count'], connected_stats['count']]
    colors = ['red', 'orange', 'blue']
    bars = plt.bar(methods, counts, color=colors, alpha=0.7)
    plt.title('Cell Count Comparison')
    plt.ylabel('Number of Cells')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # Row 2: Detected regions overlaid
    ax5 = plt.subplot(4, 4, 5)
    colored_manual = np.zeros_like(manual_array)
    colored_manual[:,:,0] = manual_array[:,:,0]  # Keep red channel
    if len(manual_labels) > 0:
        for i, label_id in enumerate(manual_labels):
            mask = manual_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_manual[mask] = [int(c*255) for c in color[:3]]
    plt.imshow(colored_manual)
    plt.title('Manual - Detected Regions')
    plt.axis('off')
    
    ax6 = plt.subplot(4, 4, 6)
    colored_ilastik = np.zeros((ilastik_array.shape[0], ilastik_array.shape[1], 3))
    if len(ilastik_labels) > 0:
        for i, label_id in enumerate(ilastik_labels):
            mask = ilastik_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_ilastik[mask] = color[:3]
    plt.imshow(colored_ilastik)
    plt.title('Ilastik - Detected Regions')
    plt.axis('off')
    
    ax7 = plt.subplot(4, 4, 7)
    colored_connected = np.zeros((connected_array.shape[0], connected_array.shape[1], 3))
    if len(connected_labels) > 0:
        for i, label_id in enumerate(connected_labels):
            mask = connected_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_connected[mask] = color[:3]
    plt.imshow(colored_connected)
    plt.title('Connected - Detected Regions')
    plt.axis('off')
    
    # Area statistics comparison
    ax8 = plt.subplot(4, 4, 8)
    stats_categories = ['Mean', 'Median', 'Std']
    manual_vals = [manual_stats['mean'], manual_stats['median'], manual_stats['std']]
    ilastik_vals = [ilastik_stats['mean'], ilastik_stats['median'], ilastik_stats['std']]
    connected_vals = [connected_stats['mean'], connected_stats['median'], connected_stats['std']]
    
    x = np.arange(len(stats_categories))
    width = 0.25
    plt.bar(x - width, manual_vals, width, label='Manual', color='red', alpha=0.7)
    plt.bar(x, ilastik_vals, width, label='Ilastik', color='orange', alpha=0.7)
    plt.bar(x + width, connected_vals, width, label='Connected', color='blue', alpha=0.7)
    plt.xlabel('Statistic')
    plt.ylabel('Area (pixels)')
    plt.title('Area Statistics Comparison')
    plt.xticks(x, stats_categories)
    plt.legend()
    
    # Row 3: Histograms
    ax9 = plt.subplot(4, 4, 9)
    if manual_areas:
        plt.hist(manual_areas, bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.title(f'Manual Area Distribution\n(n={len(manual_areas)})')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    
    ax10 = plt.subplot(4, 4, 10)
    if ilastik_areas:
        plt.hist(ilastik_areas, bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.title(f'Ilastik Area Distribution\n(n={len(ilastik_areas)})')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    
    ax11 = plt.subplot(4, 4, 11)
    if connected_areas:
        plt.hist(connected_areas, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Connected Area Distribution\n(n={len(connected_areas)})')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Frequency')
    
    # Overlaid histograms
    ax12 = plt.subplot(4, 4, 12)
    if manual_areas:
        plt.hist(manual_areas, bins=20, alpha=0.5, color='red', label='Manual', density=True)
    if ilastik_areas:
        plt.hist(ilastik_areas, bins=20, alpha=0.5, color='orange', label='Ilastik', density=True)
    if connected_areas:
        plt.hist(connected_areas, bins=20, alpha=0.5, color='blue', label='Connected', density=True)
    plt.title('Overlaid Area Distributions')
    plt.xlabel('Area (pixels)')
    plt.ylabel('Density')
    plt.legend()
    
    # Row 4: Error analysis
    ax13 = plt.subplot(4, 4, 13)
    count_errors = [abs(ilastik_stats['count'] - manual_stats['count']), 
                   abs(connected_stats['count'] - manual_stats['count'])]
    plt.bar(['Ilastik', 'Connected'], count_errors, color=['orange', 'blue'], alpha=0.7)
    plt.title('Cell Count Error\n(vs Manual)')
    plt.ylabel('Absolute Difference')
    
    ax14 = plt.subplot(4, 4, 14)
    mean_errors = [abs(manual_vs_ilastik['mean_diff']), abs(manual_vs_connected['mean_diff'])]
    plt.bar(['Ilastik', 'Connected'], mean_errors, color=['orange', 'blue'], alpha=0.7)
    plt.title('Mean Area Error\n(vs Manual)')
    plt.ylabel('Absolute Difference (pixels)')
    
    ax15 = plt.subplot(4, 4, 15)
    ks_stats = [manual_vs_ilastik['ks_statistic'], manual_vs_connected['ks_statistic']]
    plt.bar(['Ilastik', 'Connected'], ks_stats, color=['orange', 'blue'], alpha=0.7)
    plt.title('Distribution Similarity\n(KS statistic, lower=better)')
    plt.ylabel('KS Statistic')
    
    # Summary text
    ax16 = plt.subplot(4, 4, 16)
    summary_text = f"""
BIOLOGICAL ACCURACY SUMMARY:

Cell Count:
  Manual: {manual_stats['count']}
  Ilastik: {ilastik_stats['count']} ({ilastik_stats['count'] - manual_stats['count']:+d})
  Connected: {connected_stats['count']} ({connected_stats['count'] - manual_stats['count']:+d})

Mean Cell Area:
  Manual: {manual_stats['mean']:.1f}
  Ilastik: {ilastik_stats['mean']:.1f} ({manual_vs_ilastik['mean_diff']:+.1f})
  Connected: {connected_stats['mean']:.1f} ({manual_vs_connected['mean_diff']:+.1f})

Distribution Similarity (p-value):
  Manual vs Ilastik: {manual_vs_ilastik['ks_pvalue']:.3f}
  Manual vs Connected: {manual_vs_connected['ks_pvalue']:.3f}
"""
    plt.text(0.05, 0.95, summary_text, transform=ax16.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    plt.title('Summary')
    plt.axis('off')
    
    plt.suptitle(f'Biological Analysis: {pair_name}', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(figures_dir, f"{pair_name}_analysis.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved figure to {figure_path}")
    
    plt.show()
    
    # Save area distributions as CSV
    csv_path = os.path.join(figures_dir, f"{pair_name}_areas.csv")
    with open(csv_path, 'w') as f:
        f.write("Method,Area\n")
        for area in manual_areas:
            f.write(f"Manual,{area}\n")
        for area in ilastik_areas:
            f.write(f"Ilastik,{area}\n")
        for area in connected_areas:
            f.write(f"Connected,{area}\n")
    print(f"  📊 Saved area data to {csv_path}")
    
    return {
        'manual_stats': manual_stats,
        'ilastik_stats': ilastik_stats,
        'connected_stats': connected_stats,
        'manual_vs_ilastik': manual_vs_ilastik,
        'manual_vs_connected': manual_vs_connected,
        'manual_areas': manual_areas,
        'ilastik_areas': ilastik_areas,
        'connected_areas': connected_areas
    }

def main():
    """Process all image trios for biological analysis."""
    
    # Find all manual trace files
    manual_pattern = "manual_trace/manual-trace-*.png"
    manual_files = glob.glob(manual_pattern)
    
    all_results = []
    
    for manual_file in manual_files:
        try:
            # Extract identifier
            identifier = manual_file.replace("manual_trace/manual-trace-", "").replace(".png", "")
            
            # Look for corresponding files
            ilastik_file = f"simple_segmentation/simple-segmentation-{identifier}.png"
            connected_file = f"connected_skeletons/{identifier}_connected.png"
            
            if os.path.exists(ilastik_file) and os.path.exists(connected_file):
                pair_name = f"Pair_{identifier}"
                
                # Perform biological analysis
                results = biological_analysis(manual_file, ilastik_file, connected_file, pair_name)
                results['pair'] = pair_name
                results['identifier'] = identifier
                all_results.append(results)
                
            else:
                print(f"Missing files for {identifier}")
                
        except Exception as e:
            print(f"Error processing {manual_file}: {e}")
            continue
    
    # Overall summary
    if all_results:
        print("\n" + "="*100)
        print("OVERALL BIOLOGICAL PERFORMANCE SUMMARY")
        print("="*100)
        
        # Calculate average statistics
        manual_counts = [r['manual_stats']['count'] for r in all_results]
        ilastik_counts = [r['ilastik_stats']['count'] for r in all_results]
        connected_counts = [r['connected_stats']['count'] for r in all_results]
        
        manual_means = [r['manual_stats']['mean'] for r in all_results]
        ilastik_means = [r['ilastik_stats']['mean'] for r in all_results]
        connected_means = [r['connected_stats']['mean'] for r in all_results]
        
        count_errors_ilastik = [abs(i - m) for i, m in zip(ilastik_counts, manual_counts)]
        count_errors_connected = [abs(c - m) for c, m in zip(connected_counts, manual_counts)]
        
        mean_errors_ilastik = [abs(r['manual_vs_ilastik']['mean_diff']) for r in all_results]
        mean_errors_connected = [abs(r['manual_vs_connected']['mean_diff']) for r in all_results]
        
        print(f"Average cell count accuracy:")
        
        print(f"  Ilastik error:       {np.mean(count_errors_ilastik):.1f} ± {np.std(count_errors_ilastik):.1f}")
        print(f"  Connected error:     {np.mean(count_errors_connected):.1f} ± {np.std(count_errors_connected):.1f}")
        
        print(f"\nAverage cell area accuracy:")
        
        print(f"  Ilastik error:       {np.mean(mean_errors_ilastik):.1f} ± {np.std(mean_errors_ilastik):.1f}")
        print(f"  Connected error:     {np.mean(mean_errors_connected):.1f} ± {np.std(mean_errors_connected):.1f}")
        
        # Determine which method is better
        count_improvement = np.mean(count_errors_ilastik) - np.mean(count_errors_connected)
        area_improvement = np.mean(mean_errors_ilastik) - np.mean(mean_errors_connected)
        
        print(f"\nIMPROVEMENT ANALYSIS:")
        print(f"  Cell count accuracy improvement: {count_improvement:+.1f} cells")
        print(f"  Cell area accuracy improvement:  {area_improvement:+.1f} pixels")

if __name__ == "__main__":
    main()