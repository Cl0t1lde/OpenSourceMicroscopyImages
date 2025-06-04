from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
import glob
from scipy import stats

def crop_to_square(image):
    """Crop image to square (center crop) - same as preprocessing"""
    w, h = image.size
    min_dim = min(w, h)
    
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    return image.crop((left, top, right, bottom))

def load_and_preprocess_image(image_path, target_size=(384, 384)):
    """Load and preprocess a single image using same pipeline as training"""
    image = Image.open(image_path).convert('RGB')
    square_image = crop_to_square(image)
    resized_image = square_image.resize(target_size, Image.LANCZOS)
    return resized_image

def rotate_90_clockwise(image):
    """Rotate image 90 degrees clockwise"""
    return image.rotate(-90, expand=True)

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
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # F1 score (harmonic mean of precision and recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dice coefficient
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    
    # Calculate rates as percentages of manual trace
    manual_pixels = np.sum(manual_mask == 1)
    fp_rate = (FP / manual_pixels * 100) if manual_pixels > 0 else 0
    fn_rate = (FN / manual_pixels * 100) if manual_pixels > 0 else 0
    
    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'f1_score': f1_score, 'dice': dice,
        'fp_rate': fp_rate, 'fn_rate': fn_rate,
        'manual_pixels': manual_pixels
    }

def find_optimal_dilation(manual_mask, predicted_mask, max_iterations=6):
    """Find optimal dilation iterations based on Dice score."""
    best_dice = 0
    best_dilation = 1
    
    for iterations in range(1, max_iterations + 1):
        dilated_mask = binary_dilation(predicted_mask, iterations=iterations).astype(np.uint8)
        metrics = calculate_metrics(manual_mask, dilated_mask)
        
        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            best_dilation = iterations
    
    return best_dilation, best_dice

def create_binary_mask_from_red(image_array, red_threshold=10, other_threshold=127):
    """Create binary mask from RGB image looking for red pixels."""
    if len(image_array.shape) == 3:
        red_mask = (
            (image_array[:, :, 0] > red_threshold) &
            (image_array[:, :, 1] < other_threshold) &
            (image_array[:, :, 2] < other_threshold)
        )
    else:
        # Grayscale - assume non-zero pixels are traces
        red_mask = image_array > 128
    
    return red_mask.astype(np.uint8)

def create_binary_mask_from_grayscale(image_array, threshold=128):
    """Create binary mask from grayscale image."""
    if len(image_array.shape) == 3:
        # Convert to grayscale first
        gray = np.mean(image_array, axis=2)
    else:
        gray = image_array
    
    return (gray > threshold).astype(np.uint8)

def find_enclosed_areas_from_binary_mask(binary_mask):
    """Find enclosed areas from a binary mask where 1s are boundaries and 0s are potential cells."""
    from scipy.ndimage import label
    
    # For binary masks, 1s are the boundaries/lines, 0s are the enclosed areas
    # Invert to find enclosed regions (areas surrounded by boundaries)
    non_boundary_mask = ~binary_mask.astype(bool)
    
    # Label connected components in non-boundary regions
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(non_boundary_mask, structure=structure)
    
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

def create_individual_comparison_figure(neural_path, manual_path, traditional_path, pair_name, output_dir):
    """Create individual comparison figure with adaptive dilation and error visualization."""
    
    print(f"\n=== Processing {pair_name} ===")
    
    # Load and preprocess images
    neural_img_raw = Image.open(neural_path)
    manual_img_raw = Image.open(manual_path)
    traditional_img_raw = Image.open(traditional_path)
    
    # Check if neural network image needs rotation
    neural_basename = os.path.basename(neural_path)
    identifier = neural_basename.replace('.png', '')
    needs_neural_rotation = identifier[0].isalpha()
    
    # Rotate neural network image if needed
    if needs_neural_rotation:
        neural_img_rotated = rotate_90_clockwise(neural_img_raw)
        traditional_img_raw = rotate_90_clockwise(traditional_img_raw)
    else:
        neural_img_rotated = neural_img_raw
    
    # Preprocess all images to same format (384x384)
    neural_square = crop_to_square(neural_img_rotated)
    neural_img = neural_square.resize((384, 384), Image.LANCZOS)
    manual_img = load_and_preprocess_image(manual_path)
    traditional_square = crop_to_square(traditional_img_raw)
    traditional_img = traditional_square.resize((384, 384), Image.LANCZOS)
    
    # Convert to numpy arrays
    neural_array = np.array(neural_img)
    manual_array = np.array(manual_img)
    traditional_array = np.array(traditional_img)
    
    # Create binary masks
    neural_bin = create_binary_mask_from_grayscale(neural_array, threshold=50)
    manual_bin = create_binary_mask_from_red(manual_array)
    traditional_bin = create_binary_mask_from_grayscale(traditional_array, threshold=128)
    
    # Find optimal dilation for each method
    print(f"  Finding optimal dilation...")
    neural_dilation, neural_best_dice = find_optimal_dilation(manual_bin, neural_bin)
    traditional_dilation, traditional_best_dice = find_optimal_dilation(manual_bin, traditional_bin)
    
    print(f"  Neural optimal: {neural_dilation} iterations (Dice: {neural_best_dice:.4f})")
    print(f"  Traditional optimal: {traditional_dilation} iterations (Dice: {traditional_best_dice:.4f})")
    
    # Apply optimal dilation
    neural_bin_dilated = binary_dilation(neural_bin, iterations=neural_dilation).astype(np.uint8)
    traditional_bin_dilated = binary_dilation(traditional_bin, iterations=traditional_dilation).astype(np.uint8)
    
    # Calculate final metrics with optimal dilation
    neural_metrics = calculate_metrics(manual_bin, neural_bin_dilated)
    traditional_metrics = calculate_metrics(manual_bin, traditional_bin_dilated)
    
    # Create error visualizations
    # Neural vs Manual
    tp_mask_neural = (manual_bin == 1) & (neural_bin_dilated == 1)
    fp_mask_neural = (manual_bin == 0) & (neural_bin_dilated == 1)
    fn_mask_neural = (manual_bin == 1) & (neural_bin_dilated == 0)
    
    error_vis_neural = np.zeros((manual_bin.shape[0], manual_bin.shape[1], 3))
    error_vis_neural[tp_mask_neural, 1] = 1.0  # Green for correct
    error_vis_neural[fp_mask_neural, 0] = 1.0  # Red for overcorrection
    error_vis_neural[fn_mask_neural, 2] = 1.0  # Blue for undercorrection
    
    # Traditional vs Manual
    tp_mask_traditional = (manual_bin == 1) & (traditional_bin_dilated == 1)
    fp_mask_traditional = (manual_bin == 0) & (traditional_bin_dilated == 1)
    fn_mask_traditional = (manual_bin == 1) & (traditional_bin_dilated == 0)
    
    error_vis_traditional = np.zeros((manual_bin.shape[0], manual_bin.shape[1], 3))
    error_vis_traditional[tp_mask_traditional, 1] = 1.0  # Green for correct
    error_vis_traditional[fp_mask_traditional, 0] = 1.0  # Red for overcorrection
    error_vis_traditional[fn_mask_traditional, 2] = 1.0  # Blue for undercorrection
    
    # Area analysis
    manual_areas, manual_labeled, manual_area_labels = find_enclosed_areas_from_binary_mask(manual_bin)
    neural_areas, neural_labeled, neural_area_labels = find_enclosed_areas_from_binary_mask(neural_bin_dilated)
    traditional_areas, traditional_labeled, traditional_area_labels = find_enclosed_areas_from_binary_mask(traditional_bin_dilated)
    
    manual_area_stats = calculate_area_statistics(manual_areas)
    neural_area_stats = calculate_area_statistics(neural_areas)
    traditional_area_stats = calculate_area_statistics(traditional_areas)
    
    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    rotation_text = " (Rotated)" if needs_neural_rotation else ""
    fig.suptitle(f'{pair_name} - Neural vs Traditional Comparison{rotation_text}', fontsize=16, fontweight='bold')
    
    # Top row: Original images
    axes[0, 0].imshow(neural_img)
    axes[0, 0].set_title(f'Neural Net Prediction\n(Optimal dilation: {neural_dilation})', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(manual_img)
    axes[0, 1].set_title('Manual Groundtruth', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(traditional_img)
    axes[0, 2].set_title(f'Traditional Method\n(Optimal dilation: {traditional_dilation})', fontsize=12)
    axes[0, 2].axis('off')
    
    # Bottom row: Error analysis and cell detection
    axes[1, 0].imshow(error_vis_neural)
    axes[1, 0].set_title(f'Neural Error Analysis\nDice: {neural_metrics["dice"]:.4f}\nGreen=Correct, Red=Over, Blue=Under', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(error_vis_traditional)
    axes[1, 1].set_title(f'Traditional Error Analysis\nDice: {traditional_metrics["dice"]:.4f}\nGreen=Correct, Red=Over, Blue=Under', fontsize=10)
    axes[1, 1].axis('off')
    
    # Cell count and area comparison
    methods = ['Manual', 'Neural', 'Traditional']
    counts = [manual_area_stats['count'], neural_area_stats['count'], traditional_area_stats['count']]
    means = [manual_area_stats['mean'], neural_area_stats['mean'], traditional_area_stats['mean']]
    colors = ['red', 'blue', 'orange']
    
    # Bar plot with both count and mean area
    x = np.arange(len(methods))
    width = 0.35
    
    ax2 = axes[1, 2]
    bars1 = ax2.bar(x - width/2, counts, width, label='Cell Count', color=colors, alpha=0.7)
    
    # Add count labels
    for bar, count in zip(bars1, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Cell Count', fontsize=11)
    ax2.set_title('Cell Detection Results', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.grid(True, alpha=0.3)
    
    # Add mean area as text
    area_text = f"Mean Areas:\nManual: {manual_area_stats['mean']:.1f}\nNeural: {neural_area_stats['mean']:.1f}\nTraditional: {traditional_area_stats['mean']:.1f}"
    ax2.text(0.02, 0.98, area_text, transform=ax2.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/{pair_name}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Calculate improvements
    improvement = {
        'precision': neural_metrics['precision'] - traditional_metrics['precision'],
        'recall': neural_metrics['recall'] - traditional_metrics['recall'],
        'f1_score': neural_metrics['f1_score'] - traditional_metrics['f1_score'],
        'dice': neural_metrics['dice'] - traditional_metrics['dice'],
        'fp_rate': neural_metrics['fp_rate'] - traditional_metrics['fp_rate'],
        'fn_rate': neural_metrics['fn_rate'] - traditional_metrics['fn_rate'],
        'cell_count_error': abs(neural_area_stats['count'] - manual_area_stats['count']) - abs(traditional_area_stats['count'] - manual_area_stats['count']),
        'mean_area_error': abs(neural_area_stats['mean'] - manual_area_stats['mean']) - abs(traditional_area_stats['mean'] - manual_area_stats['mean']) if manual_area_stats['count'] > 0 else 0
    }
    
    return {
        'pair': pair_name,
        'neural_metrics': neural_metrics,
        'traditional_metrics': traditional_metrics,
        'improvement': improvement,
        'neural_dilation': neural_dilation,
        'traditional_dilation': traditional_dilation,
        'neural_rotated': needs_neural_rotation,
        'manual_area_stats': manual_area_stats,
        'neural_area_stats': neural_area_stats,
        'traditional_area_stats': traditional_area_stats,
        'manual_areas': manual_areas,
        'neural_areas': neural_areas,
        'traditional_areas': traditional_areas
    }

def create_final_summary_figure(all_results, output_dir):
    """Create comprehensive final summary figure."""
    
    # Collect all data for statistical analysis
    all_manual_areas = []
    all_neural_areas = []
    all_traditional_areas = []
    
    for result in all_results:
        all_manual_areas.extend(result['manual_areas'])
        all_neural_areas.extend(result['neural_areas'])
        all_traditional_areas.extend(result['traditional_areas'])
    
    # Statistical tests
    from scipy.stats import ks_2samp, mannwhitneyu, ttest_rel
    
    # Distribution tests
    if len(all_manual_areas) > 0 and len(all_neural_areas) > 0:
        ks_stat_mn, ks_p_mn = ks_2samp(all_manual_areas, all_neural_areas)
        mw_stat_mn, mw_p_mn = mannwhitneyu(all_manual_areas, all_neural_areas, alternative='two-sided')
    else:
        ks_stat_mn, ks_p_mn = 0, 1
        mw_stat_mn, mw_p_mn = 0, 1
        
    if len(all_manual_areas) > 0 and len(all_traditional_areas) > 0:
        ks_stat_mt, ks_p_mt = ks_2samp(all_manual_areas, all_traditional_areas)
        mw_stat_mt, mw_p_mt = mannwhitneyu(all_manual_areas, all_traditional_areas, alternative='two-sided')
    else:
        ks_stat_mt, ks_p_mt = 0, 1
        mw_stat_mt, mw_p_mt = 0, 1
    
    # Performance metrics for significance testing
    neural_dices = [r['neural_metrics']['dice'] for r in all_results]
    traditional_dices = [r['traditional_metrics']['dice'] for r in all_results]
    neural_precisions = [r['neural_metrics']['precision'] for r in all_results]
    traditional_precisions = [r['traditional_metrics']['precision'] for r in all_results]
    
    if len(all_results) >= 3:
        _, p_dice = ttest_rel(neural_dices, traditional_dices)
        _, p_precision = ttest_rel(neural_precisions, traditional_precisions)
    else:
        p_dice = p_precision = 1.0
    
    # Create comprehensive summary figure
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    fig.suptitle('Final Summary: Neural Network vs Traditional Method Analysis', fontsize=18, fontweight='bold')
    
    # Panel 1: Overall performance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Precision', 'Recall', 'F1 Score', 'Dice']
    neural_values = [np.mean([r['neural_metrics'][cat.lower().replace(' ', '_')] for r in all_results]) for cat in categories]
    traditional_values = [np.mean([r['traditional_metrics'][cat.lower().replace(' ', '_')] for r in all_results]) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, neural_values, width, label='Neural Net', alpha=0.7, color='blue')
    bars2 = ax1.bar(x + width/2, traditional_values, width, label='Traditional', alpha=0.7, color='orange')
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Average Performance Metrics', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel 2: Cell count accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    total_manual = sum(r['manual_area_stats']['count'] for r in all_results)
    total_neural = sum(r['neural_area_stats']['count'] for r in all_results)
    total_traditional = sum(r['traditional_area_stats']['count'] for r in all_results)
    
    counts = [total_manual, total_neural, total_traditional]
    methods = ['Manual', 'Neural', 'Traditional']
    colors = ['red', 'blue', 'orange']
    
    bars = ax2.bar(methods, counts, color=colors, alpha=0.7)
    ax2.set_ylabel('Total Cell Count', fontsize=11)
    ax2.set_title('Total Cells Detected', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Panel 3: Dilation optimization results
    ax3 = fig.add_subplot(gs[0, 2])
    neural_dilations = [r['neural_dilation'] for r in all_results]
    traditional_dilations = [r['traditional_dilation'] for r in all_results]
    
    dilation_counts = {}
    for dil in neural_dilations:
        dilation_counts[f'Neural_{dil}'] = dilation_counts.get(f'Neural_{dil}', 0) + 1
    for dil in traditional_dilations:
        dilation_counts[f'Traditional_{dil}'] = dilation_counts.get(f'Traditional_{dil}', 0) + 1
    
    # Show average optimal dilation
    avg_neural_dil = np.mean(neural_dilations)
    avg_traditional_dil = np.mean(traditional_dilations)
    
    ax3.bar(['Neural', 'Traditional'], [avg_neural_dil, avg_traditional_dil], 
           color=['blue', 'orange'], alpha=0.7)
    ax3.set_ylabel('Average Optimal Dilation', fontsize=11)
    ax3.set_title('Adaptive Dilation Results', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    for i, (method, avg_dil) in enumerate([('Neural', avg_neural_dil), ('Traditional', avg_traditional_dil)]):
        ax3.text(i, avg_dil + 0.05, f'{avg_dil:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Panel 4: Per-image Dice comparison
    ax4 = fig.add_subplot(gs[0, 3])
    image_names = [r['pair'].replace('Comparison_', '') for r in all_results]
    neural_dices = [r['neural_metrics']['dice'] for r in all_results]
    traditional_dices = [r['traditional_metrics']['dice'] for r in all_results]
    
    x = np.arange(len(image_names))
    ax4.plot(x, neural_dices, 'bo-', label='Neural', alpha=0.7, linewidth=2, markersize=8)
    ax4.plot(x, traditional_dices, 'o-', color='orange', label='Traditional', alpha=0.7, linewidth=2, markersize=8)
    ax4.set_xlabel('Images', fontsize=11)
    ax4.set_ylabel('Dice Score', fontsize=11)
    ax4.set_title('Per-Image Dice Comparison', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(image_names, rotation=45, fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Panel 5: Combined area distribution
    ax5 = fig.add_subplot(gs[1, :2])
    if all_manual_areas:
        ax5.hist(all_manual_areas, bins=30, alpha=0.6, color='red', 
                label=f'Manual (n={len(all_manual_areas)})', density=True)
        ax5.axvline(np.mean(all_manual_areas), color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    if all_neural_areas:
        ax5.hist(all_neural_areas, bins=30, alpha=0.6, color='blue', 
                label=f'Neural (n={len(all_neural_areas)})', density=True)
        ax5.axvline(np.mean(all_neural_areas), color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    if all_traditional_areas:
        ax5.hist(all_traditional_areas, bins=30, alpha=0.6, color='orange', 
                label=f'Traditional (n={len(all_traditional_areas)})', density=True)
        ax5.axvline(np.mean(all_traditional_areas), color='orange', linestyle='--', alpha=0.8, linewidth=2)
    
    ax5.set_xlabel('Cell Area (pixels)', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Combined Cell Size Distribution', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Error analysis summary
    ax6 = fig.add_subplot(gs[1, 2:])
    
    # Calculate average errors
    avg_neural_fp = np.mean([r['neural_metrics']['fp_rate'] for r in all_results])
    avg_neural_fn = np.mean([r['neural_metrics']['fn_rate'] for r in all_results])
    avg_traditional_fp = np.mean([r['traditional_metrics']['fp_rate'] for r in all_results])
    avg_traditional_fn = np.mean([r['traditional_metrics']['fn_rate'] for r in all_results])
    
    error_types = ['Overcorrection\n(FP%)', 'Undercorrection\n(FN%)']
    neural_errors = [avg_neural_fp, avg_neural_fn]
    traditional_errors = [avg_traditional_fp, avg_traditional_fn]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    ax6.bar(x - width/2, neural_errors, width, label='Neural', color='blue', alpha=0.7)
    ax6.bar(x + width/2, traditional_errors, width, label='Traditional', color='orange', alpha=0.7)
    
    ax6.set_ylabel('Error Rate (%)', fontsize=11)
    ax6.set_title('Average Error Rates', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(error_types)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Panel 7: Statistical test results
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    
    stats_text = f"""
STATISTICAL TESTS:

Distribution Similarity (KS test):
Manual vs Neural:
• K-S statistic: {ks_stat_mn:.4f}
• p-value: {ks_p_mn:.4f}
• Significant: {'Yes' if ks_p_mn < 0.05 else 'No'}

Manual vs Traditional:
• K-S statistic: {ks_stat_mt:.4f}
• p-value: {ks_p_mt:.4f}
• Significant: {'Yes' if ks_p_mt < 0.05 else 'No'}

Performance (Paired t-test):
Dice: p = {p_dice:.4f} {'***' if p_dice < 0.001 else '**' if p_dice < 0.01 else '*' if p_dice < 0.05 else 'ns'}
    """
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # Panel 8: Improvement summary
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    
    # Calculate overall improvements
    avg_dice_improvement = np.mean([r['improvement']['dice'] for r in all_results])
    avg_precision_improvement = np.mean([r['improvement']['precision'] for r in all_results])
    avg_recall_improvement = np.mean([r['improvement']['recall'] for r in all_results])
    neural_count_error = np.mean([abs(r['neural_area_stats']['count'] - r['manual_area_stats']['count']) for r in all_results])
    traditional_count_error = np.mean([abs(r['traditional_area_stats']['count'] - r['manual_area_stats']['count']) for r in all_results])
    count_improvement = traditional_count_error - neural_count_error
    
    improvement_text = f"""
OVERALL IMPROVEMENTS:
(Neural - Traditional)

Pixel-Level:
• Dice: {avg_dice_improvement:+.4f}
• Precision: {avg_precision_improvement:+.4f}
• Recall: {avg_recall_improvement:+.4f}

Cell-Level:
• Count error: {count_improvement:+.1f} cells
• Total cells: {total_neural - total_traditional:+d}

Best Method: {'Neural Network' if avg_dice_improvement > 0 else 'Traditional'}
    """
    
    ax8.text(0.05, 0.95, improvement_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    
    # Panel 9: Individual results table
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')
    
    table_text = f"""
INDIVIDUAL RESULTS:

{'Image':<8} | {'Neural Dice':<11} | {'Trad Dice':<10} | {'Neural Dil':<10} | {'Trad Dil':<9}
{'-'*60}
"""
    
    for r in all_results:
        identifier = r['pair'].replace('Comparison_', '')
        neural_dice = r['neural_metrics']['dice']
        trad_dice = r['traditional_metrics']['dice']
        neural_dil = r['neural_dilation']
        trad_dil = r['traditional_dilation']
        table_text += f"{identifier:<8} | {neural_dice:10.4f} | {trad_dice:9.4f} | {neural_dil:9d} | {trad_dil:8d}\n"
    
    ax9.text(0.05, 0.95, table_text, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    
    # Panel 10: Final summary
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    
    # Calculate win/loss record
    neural_wins = sum(1 for r in all_results if r['neural_metrics']['dice'] > r['traditional_metrics']['dice'])
    total_comparisons = len(all_results)
    win_percentage = (neural_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
    
    final_text = f"""
FINAL COMPREHENSIVE SUMMARY:

Dataset: {total_comparisons} image pairs analyzed with adaptive dilation optimization

PERFORMANCE SUMMARY:
• Neural Network wins: {neural_wins}/{total_comparisons} comparisons ({win_percentage:.1f}%)
• Average Dice improvement: {avg_dice_improvement:+.4f} (Neural vs Traditional)
• Average optimal dilation: Neural={avg_neural_dil:.1f}, Traditional={avg_traditional_dil:.1f}

CELL DETECTION ACCURACY:
• Total manual cells: {total_manual}
• Neural detection: {total_neural} cells (error: {abs(total_neural - total_manual)} cells, {abs(total_neural - total_manual)/total_manual*100:.1f}%)
• Traditional detection: {total_traditional} cells (error: {abs(total_traditional - total_manual)} cells, {abs(total_traditional - total_manual)/total_manual*100:.1f}%)

STATISTICAL SIGNIFICANCE:
• Performance difference (Dice): p = {p_dice:.4f} {'(significant)' if p_dice < 0.05 else '(not significant)'}
• Cell area distributions significantly different from manual: Neural p={ks_p_mn:.4f}, Traditional p={ks_p_mt:.4f}

CONCLUSION: {'Neural Network outperforms traditional method' if avg_dice_improvement > 0 and p_dice < 0.05 else 'Neural Network shows improvement but not statistically significant' if avg_dice_improvement > 0 else 'Traditional method performs better'}
"""
    
    ax10.text(0.02, 0.98, final_text, transform=ax10.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/FINAL_NEURAL_VS_TRADITIONAL_SUMMARY.png", 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'total_comparisons': total_comparisons,
        'neural_wins': neural_wins,
        'avg_dice_improvement': avg_dice_improvement,
        'statistical_significance': p_dice,
        'cell_accuracy': {
            'manual': total_manual,
            'neural': total_neural,
            'traditional': total_traditional
        }
    }

def main():
    """Main function to process all neural network comparisons."""
    
    # Define paths
    neural_dir = "final-test-set/predictions"
    manual_dir = "final-test-set/groundtruth"
    traditional_dir = "final-test-set/connected_skeletons"
    
    # Create output directory
    output_dir = 'neural_network_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all prediction files
    neural_files = glob.glob(os.path.join(neural_dir, "*.png"))
    
    print(f"Found {len(neural_files)} neural network prediction files")
    print("Creating individual comparison figures with adaptive dilation...")
    
    results = []
    
    for neural_file in neural_files:
        try:
            # Extract identifier from neural file
            neural_basename = os.path.basename(neural_file)
            identifier = neural_basename.replace('.png', '')
            
            # Find corresponding files
            manual_file = os.path.join(manual_dir, f"{identifier}.png")
            traditional_file = os.path.join(traditional_dir, f"{identifier}_connected.png")
            
            if os.path.exists(manual_file) and os.path.exists(traditional_file):
                pair_name = f"Comparison_{identifier}"
                
                # Create individual comparison figure
                comparison_results = create_individual_comparison_figure(
                    neural_file, manual_file, traditional_file, pair_name, output_dir
                )
                
                results.append(comparison_results)
                
                print(f"✅ {pair_name} completed")
                print(f"   Neural Dice: {comparison_results['neural_metrics']['dice']:.4f} (dilation: {comparison_results['neural_dilation']})")
                print(f"   Traditional Dice: {comparison_results['traditional_metrics']['dice']:.4f} (dilation: {comparison_results['traditional_dilation']})")
                print(f"   Improvement: {comparison_results['improvement']['dice']:+.4f}")
                
            else:
                missing = []
                if not os.path.exists(manual_file):
                    missing.append(f"Manual: {manual_file}")
                if not os.path.exists(traditional_file):
                    missing.append(f"Traditional: {traditional_file}")
                print(f"✗ Missing files for {identifier}: {', '.join(missing)}")
                
        except Exception as e:
            print(f"❌ Error processing {neural_file}: {e}")
            continue
    
    # Create final summary
    if results:
        print(f"\nCreating final summary figure...")
        summary_stats = create_final_summary_figure(results, output_dir)
        
        # Print console summary
        print("\n" + "="*100)
        print("FINAL NEURAL NETWORK vs TRADITIONAL METHOD SUMMARY")
        print("="*100)
        
        print(f"Total comparisons: {summary_stats['total_comparisons']}")
        print(f"Neural Network wins: {summary_stats['neural_wins']}/{summary_stats['total_comparisons']} ({summary_stats['neural_wins']/summary_stats['total_comparisons']*100:.1f}%)")
        print(f"Average Dice improvement: {summary_stats['avg_dice_improvement']:+.4f}")
        print(f"Statistical significance (p-value): {summary_stats['statistical_significance']:.4f}")
        
        if summary_stats['statistical_significance'] < 0.05:
            print(f"✅ Neural Network significantly outperforms traditional method!")
        else:
            print(f"⚠️  Improvement not statistically significant")
        
        print(f"\nCell detection accuracy:")
        print(f"Manual reference: {summary_stats['cell_accuracy']['manual']} cells")
        print(f"Neural Network: {summary_stats['cell_accuracy']['neural']} cells")
        print(f"Traditional: {summary_stats['cell_accuracy']['traditional']} cells")
        
    else:
        print("❌ No valid file pairs found for comparison!")
    
    print(f"\n📁 All results saved to: {output_dir}/")
    print(f"📊 Final summary: {output_dir}/FINAL_NEURAL_VS_TRADITIONAL_SUMMARY.png")

if __name__ == "__main__":
    main()