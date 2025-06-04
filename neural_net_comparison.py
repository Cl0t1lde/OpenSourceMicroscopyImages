from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
import glob

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

def rotate_90_counterclockwise(image):
    """Rotate image 90 degrees counterclockwise"""
    return image.rotate(90, expand=True)

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

def find_enclosed_areas_per_region(image_array, red_threshold=100):
    """Find enclosed areas in a manual trace image."""
    from scipy.ndimage import label
    
    if len(image_array.shape) == 3:
        # RGB image - look for red pixels
        red_mask = (image_array[:, :, 0] > 120) & (image_array[:, :, 1] < red_threshold) & (image_array[:, :, 2] < red_threshold)
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
            if area > 10:  # Minimum area threshold
                region_areas.append(area)
                region_labels.append(label_id)
    
    return region_areas, labeled, region_labels

def find_areas_from_prediction(pred_array, threshold=50):
    """Find cell areas from neural network prediction or traditional skeleton."""
    from scipy.ndimage import label
    
    # Convert to binary
    if len(pred_array.shape) == 3:
        # Convert to grayscale first
        gray = np.mean(pred_array, axis=2)
    else:
        gray = pred_array
    
    # For predictions/skeletons, lines are boundaries, so invert to find enclosed areas
    pred_bin = (gray > threshold).astype(np.uint8)
    
    # Dilate to close small gaps
    pred_dilated = binary_dilation(pred_bin, iterations=2).astype(np.uint8)
    
    # Invert to find enclosed regions
    non_pred_mask = ~pred_dilated.astype(bool)
    
    # Label connected components
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(non_pred_mask, structure=structure)
    
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

def compare_area_distributions(areas1, areas2, method1_name, method2_name):
    """Compare two area distributions statistically."""
    from scipy import stats
    
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

def analyze_neural_net_comparison(neural_path, manual_path, traditional_path, pair_name):
    """Compare neural network predictions with manual traces and traditional methods."""
    
    print(f"\n=== NEURAL NETWORK COMPARISON: {pair_name} ===")
    print(f"Neural net prediction: {neural_path}")
    print(f"Manual groundtruth: {manual_path}")
    print(f"Traditional method: {traditional_path}")
    
    # Load raw images
    neural_img_raw = Image.open(neural_path)
    manual_img_raw = Image.open(manual_path)
    traditional_img_raw = Image.open(traditional_path)
    
    print(f"Raw image sizes - Neural: {neural_img_raw.size}, Manual: {manual_img_raw.size}, Traditional: {traditional_img_raw.size}")
    
    # Check if neural network image needs rotation (if filename starts with a letter)
    neural_basename = os.path.basename(neural_path)
    identifier = neural_basename.replace('.png', '')
    needs_neural_rotation = identifier[0].isalpha()
    
    print(f"Identifier: {identifier}, starts with letter: {needs_neural_rotation}")
    
    # Rotate neural network image 90° clockwise if filename starts with a letter
    if needs_neural_rotation:
        neural_img_rotated = rotate_90_clockwise(neural_img_raw)
        print(f"Neural after rotation: {neural_img_rotated.size}")
    else:
        neural_img_rotated = neural_img_raw
        print("Neural: No rotation needed")
    
    # Traditional method: no rotation needed
    traditional_img_rotated = traditional_img_raw
    print("Traditional: No rotation applied")
    
    # Preprocess all images to same format (384x384)
    # For neural: apply preprocessing to the potentially rotated image
    neural_square = crop_to_square(neural_img_rotated)
    neural_img = neural_square.resize((384, 384), Image.LANCZOS)
    
    manual_img = load_and_preprocess_image(manual_path)
    
    # For traditional: apply preprocessing to the non-rotated image
    traditional_square = crop_to_square(traditional_img_rotated)
    traditional_img = traditional_square.resize((384, 384), Image.LANCZOS)
    
    print(f"After preprocessing - All images: {neural_img.size}")
    
    # Convert to numpy arrays
    neural_array = np.array(neural_img)
    manual_array = np.array(manual_img)
    traditional_array = np.array(traditional_img)
    
    # Create binary masks
    print("Creating binary masks...")
    
    # Neural net - assume it outputs cell wall predictions (look for non-black pixels)
    neural_bin = create_binary_mask_from_grayscale(neural_array, threshold=50)
    
    # Manual trace - look for red pixels
    manual_bin = create_binary_mask_from_red(manual_array)
    
    # Traditional - assume grayscale skeleton
    traditional_bin = create_binary_mask_from_grayscale(traditional_array, threshold=128)
    
    print(f"Mask pixel counts - Manual: {np.sum(manual_bin)}, Neural: {np.sum(neural_bin)}, Traditional: {np.sum(traditional_bin)}")
    
    # Apply fixed dilation of 2 iterations
    dilation_iterations = 2
    print(f"Applying fixed dilation of {dilation_iterations} iterations...")
    neural_bin_dilated = binary_dilation(neural_bin, iterations=dilation_iterations).astype(np.uint8)
    traditional_bin_dilated = binary_dilation(traditional_bin, iterations=dilation_iterations).astype(np.uint8)
    
    # Calculate pixel-level metrics
    neural_metrics = calculate_metrics(manual_bin, neural_bin_dilated)
    traditional_metrics = calculate_metrics(manual_bin, traditional_bin_dilated)
    neural_vs_traditional = calculate_metrics(neural_bin_dilated, traditional_bin_dilated)
    
    # AREA-BASED ANALYSIS using binary masks
    print("\nPerforming area-based analysis using binary masks...")
    
    # Find enclosed areas for each method using binary masks
    manual_areas, manual_labeled, manual_area_labels = find_enclosed_areas_from_binary_mask(manual_bin)
    neural_areas, neural_labeled, neural_area_labels = find_enclosed_areas_from_binary_mask(neural_bin_dilated)
    traditional_areas, traditional_labeled, traditional_area_labels = find_enclosed_areas_from_binary_mask(traditional_bin_dilated)
    
    print(f"Detected cell counts - Manual: {len(manual_areas)}, Neural: {len(neural_areas)}, Traditional: {len(traditional_areas)}")
    
    # Calculate area statistics
    manual_area_stats = calculate_area_statistics(manual_areas)
    neural_area_stats = calculate_area_statistics(neural_areas)
    traditional_area_stats = calculate_area_statistics(traditional_areas)
    
    # Compare area distributions
    manual_vs_neural_areas = compare_area_distributions(manual_areas, neural_areas, "Manual", "Neural")
    manual_vs_traditional_areas = compare_area_distributions(manual_areas, traditional_areas, "Manual", "Traditional")
    neural_vs_traditional_areas = compare_area_distributions(neural_areas, traditional_areas, "Neural", "Traditional")
    
    # Print results
    rotation_info = f" (Neural rotated: {'Yes' if needs_neural_rotation else 'No'})"
    print(f"\nPIXEL-LEVEL PERFORMANCE COMPARISON (with {dilation_iterations} iterations dilation){rotation_info}:")
    print(f"{'Method':<12} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Dice':<9} | {'Over%':<7} | {'Under%':<7}")
    print("-" * 80)
    print(f"{'Neural Net':<12} | {neural_metrics['precision']:8.4f} | {neural_metrics['recall']:8.4f} | "
          f"{neural_metrics['f1_score']:8.4f} | {neural_metrics['dice']:8.4f} | "
          f"{neural_metrics['fp_rate']:6.1f} | {neural_metrics['fn_rate']:6.1f}")
    print(f"{'Traditional':<12} | {traditional_metrics['precision']:8.4f} | {traditional_metrics['recall']:8.4f} | "
          f"{traditional_metrics['f1_score']:8.4f} | {traditional_metrics['dice']:8.4f} | "
          f"{traditional_metrics['fp_rate']:6.1f} | {traditional_metrics['fn_rate']:6.1f}")
    
    print(f"\nCELL COUNT COMPARISON (from binary masks):")
    print(f"Manual reference:    {manual_area_stats['count']:3d} cells")
    print(f"Neural Net:          {neural_area_stats['count']:3d} cells (diff: {neural_area_stats['count'] - manual_area_stats['count']:+3d})")
    print(f"Traditional:         {traditional_area_stats['count']:3d} cells (diff: {traditional_area_stats['count'] - manual_area_stats['count']:+3d})")
    
    print(f"\nCELL AREA STATISTICS (pixels):")
    print(f"{'Method':<12} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 70)
    print(f"{'Manual':<12} | {manual_area_stats['mean']:8.1f} | {manual_area_stats['median']:8.1f} | {manual_area_stats['std']:8.1f} | {manual_area_stats['min']:6.0f} | {manual_area_stats['max']:6.0f}")
    print(f"{'Neural':<12} | {neural_area_stats['mean']:8.1f} | {neural_area_stats['median']:8.1f} | {neural_area_stats['std']:8.1f} | {neural_area_stats['min']:6.0f} | {neural_area_stats['max']:6.0f}")
    print(f"{'Traditional':<12} | {traditional_area_stats['mean']:8.1f} | {traditional_area_stats['median']:8.1f} | {traditional_area_stats['std']:8.1f} | {traditional_area_stats['min']:6.0f} | {traditional_area_stats['max']:6.0f}")
    
    if manual_area_stats['count'] > 0:
        print(f"\nAREA DISTRIBUTION COMPARISON (Manual as reference):")
        print(f"Manual vs Neural     - KS test p-value: {manual_vs_neural_areas['ks_pvalue']:.4f}, Mean diff: {manual_vs_neural_areas['mean_diff']:+8.1f}")
        print(f"Manual vs Traditional - KS test p-value: {manual_vs_traditional_areas['ks_pvalue']:.4f}, Mean diff: {manual_vs_traditional_areas['mean_diff']:+8.1f}")
    else:
        print(f"\nWARNING: No cells detected in manual mask for area distribution comparison")
    
    improvement = {
        'precision': neural_metrics['precision'] - traditional_metrics['precision'],
        'recall': neural_metrics['recall'] - traditional_metrics['recall'],
        'f1_score': neural_metrics['f1_score'] - traditional_metrics['f1_score'],
        'dice': neural_metrics['dice'] - traditional_metrics['dice'],
        'fp_rate': neural_metrics['fp_rate'] - traditional_metrics['fp_rate'],
        'fn_rate': neural_metrics['fn_rate'] - traditional_metrics['fn_rate'],
        'cell_count_error': abs(neural_area_stats['count'] - manual_area_stats['count']) - abs(traditional_area_stats['count'] - manual_area_stats['count']),
        'mean_area_error': abs(manual_vs_neural_areas['mean_diff']) - abs(manual_vs_traditional_areas['mean_diff']) if manual_area_stats['count'] > 0 else 0
    }
    
    print(f"\nIMPROVEMENT (Neural - Traditional):")
    print(f"  Pixel-level Dice: {improvement['dice']:+.4f}")
    print(f"  Cell count error: {improvement['cell_count_error']:+.1f} cells")
    if manual_area_stats['count'] > 0:
        print(f"  Mean area error: {improvement['mean_area_error']:+.1f} pixels")
    else:
        print(f"  Mean area error: N/A (no manual cells detected)")
    
    # Create comprehensive visualization including area analysis
    fig, axes = plt.subplots(6, 4, figsize=(20, 30))
    neural_rotation_text = " (Rotated)" if needs_neural_rotation else ""
    fig.suptitle(f'Neural Network vs Traditional Comparison with Area Analysis: {pair_name}{neural_rotation_text}', fontsize=16)
    
    # Row 1: Raw images
    axes[0, 0].imshow(neural_img_raw)
    axes[0, 0].set_title('Neural Net Prediction (Raw)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(manual_img)
    axes[0, 1].set_title('Manual Groundtruth (Preprocessed)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(traditional_img_raw)
    axes[0, 2].set_title('Traditional Method (Raw)')
    axes[0, 2].axis('off')
    
    if needs_neural_rotation:
        axes[0, 3].imshow(neural_img_rotated)
        axes[0, 3].set_title('Neural (Rotated 90° CW)')
    else:
        axes[0, 3].imshow(neural_img)
        axes[0, 3].set_title('Neural (No Rotation)')
    axes[0, 3].axis('off')
    
    # Row 2: Binary masks
    axes[1, 0].imshow(neural_bin, cmap='gray')
    axes[1, 0].set_title(f'Neural Binary Mask\n{np.sum(neural_bin)} pixels')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(manual_bin, cmap='gray')
    axes[1, 1].set_title(f'Manual Binary Mask\n{np.sum(manual_bin)} pixels')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(traditional_bin, cmap='gray')
    axes[1, 2].set_title(f'Traditional Binary Mask\n{np.sum(traditional_bin)} pixels')
    axes[1, 2].axis('off')
    
    # Cell count comparison
    methods = ['Manual', 'Neural', 'Traditional']
    counts = [manual_area_stats['count'], neural_area_stats['count'], traditional_area_stats['count']]
    colors = ['red', 'blue', 'orange']
    bars = axes[1, 3].bar(methods, counts, color=colors, alpha=0.7)
    axes[1, 3].set_title('Cell Count Comparison\n(from binary masks)')
    axes[1, 3].set_ylabel('Number of Cells')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        axes[1, 3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(count), ha='center', va='bottom')
    
    # Row 3: Detected regions overlaid on binary masks
    axes[2, 0].imshow(neural_bin, cmap='gray')
    colored_neural = np.zeros((neural_bin.shape[0], neural_bin.shape[1], 3))
    if len(neural_area_labels) > 0:
        for i, label_id in enumerate(neural_area_labels):
            mask = neural_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_neural[mask] = color[:3]
    axes[2, 0].imshow(colored_neural, alpha=0.6)
    axes[2, 0].set_title(f'Neural - Detected Regions\n{len(neural_areas)} cells')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(manual_bin, cmap='gray')
    colored_manual = np.zeros((manual_bin.shape[0], manual_bin.shape[1], 3))
    if len(manual_area_labels) > 0:
        for i, label_id in enumerate(manual_area_labels):
            mask = manual_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_manual[mask] = color[:3]
    axes[2, 1].imshow(colored_manual, alpha=0.6)
    axes[2, 1].set_title(f'Manual - Detected Regions\n{len(manual_areas)} cells')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(traditional_bin, cmap='gray')
    colored_traditional = np.zeros((traditional_bin.shape[0], traditional_bin.shape[1], 3))
    if len(traditional_area_labels) > 0:
        for i, label_id in enumerate(traditional_area_labels):
            mask = traditional_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_traditional[mask] = color[:3]
    axes[2, 2].imshow(colored_traditional, alpha=0.6)
    axes[2, 2].set_title(f'Traditional - Detected Regions\n{len(traditional_areas)} cells')
    axes[2, 2].axis('off')
    
    # Area statistics comparison
    if any([manual_area_stats['count'], neural_area_stats['count'], traditional_area_stats['count']]):
        stats_categories = ['Mean', 'Median', 'Std']
        manual_vals = [manual_area_stats['mean'], manual_area_stats['median'], manual_area_stats['std']]
        neural_vals = [neural_area_stats['mean'], neural_area_stats['median'], neural_area_stats['std']]
        traditional_vals = [traditional_area_stats['mean'], traditional_area_stats['median'], traditional_area_stats['std']]
        
        x = np.arange(len(stats_categories))
        width = 0.25
        axes[2, 3].bar(x - width, manual_vals, width, label='Manual', color='red', alpha=0.7)
        axes[2, 3].bar(x, neural_vals, width, label='Neural', color='blue', alpha=0.7)
        axes[2, 3].bar(x + width, traditional_vals, width, label='Traditional', color='orange', alpha=0.7)
        axes[2, 3].set_xlabel('Statistic')
        axes[2, 3].set_ylabel('Area (pixels)')
        axes[2, 3].set_title('Area Statistics Comparison')
        axes[2, 3].set_xticks(x)
        axes[2, 3].set_xticklabels(stats_categories)
        axes[2, 3].legend()
    else:
        axes[2, 3].text(0.5, 0.5, 'No cells detected\nfor statistics', 
                       transform=axes[2, 3].transAxes, ha='center', va='center')
        axes[2, 3].set_title('Area Statistics Comparison')
    
    # Row 4: Area histograms
    if manual_areas:
        axes[3, 0].hist(manual_areas, bins=15, alpha=0.7, color='red', edgecolor='black')
    else:
        axes[3, 0].text(0.5, 0.5, 'No cells detected', transform=axes[3, 0].transAxes, ha='center', va='center')
    axes[3, 0].set_title(f'Manual Area Distribution\n(n={len(manual_areas)})')
    axes[3, 0].set_xlabel('Area (pixels)')
    axes[3, 0].set_ylabel('Frequency')
    
    if neural_areas:
        axes[3, 1].hist(neural_areas, bins=15, alpha=0.7, color='blue', edgecolor='black')
    else:
        axes[3, 1].text(0.5, 0.5, 'No cells detected', transform=axes[3, 1].transAxes, ha='center', va='center')
    axes[3, 1].set_title(f'Neural Area Distribution\n(n={len(neural_areas)})')
    axes[3, 1].set_xlabel('Area (pixels)')
    axes[3, 1].set_ylabel('Frequency')
    
    if traditional_areas:
        axes[3, 2].hist(traditional_areas, bins=15, alpha=0.7, color='orange', edgecolor='black')
    else:
        axes[3, 2].text(0.5, 0.5, 'No cells detected', transform=axes[3, 2].transAxes, ha='center', va='center')
    axes[3, 2].set_title(f'Traditional Area Distribution\n(n={len(traditional_areas)})')
    axes[3, 2].set_xlabel('Area (pixels)')
    axes[3, 2].set_ylabel('Frequency')
    
    # Overlaid histograms
    if any([manual_areas, neural_areas, traditional_areas]):
        if manual_areas:
            axes[3, 3].hist(manual_areas, bins=15, alpha=0.5, color='red', label='Manual', density=True)
        if neural_areas:
            axes[3, 3].hist(neural_areas, bins=15, alpha=0.5, color='blue', label='Neural', density=True)
        if traditional_areas:
            axes[3, 3].hist(traditional_areas, bins=15, alpha=0.5, color='orange', label='Traditional', density=True)
        axes[3, 3].legend()
    else:
        axes[3, 3].text(0.5, 0.5, 'No cells detected\nfor comparison', 
                       transform=axes[3, 3].transAxes, ha='center', va='center')
    axes[3, 3].set_title('Overlaid Area Distributions')
    axes[3, 3].set_xlabel('Area (pixels)')
    axes[3, 3].set_ylabel('Density')
    
    # Row 5: Error analysis
    axes[4, 0].imshow(neural_bin_dilated, cmap='gray')
    axes[4, 0].set_title(f'Neural Dilated (dil={dilation_iterations})\nDice: {neural_metrics["dice"]:.4f}')
    axes[4, 0].axis('off')
    
    axes[4, 1].imshow(manual_bin, cmap='gray')
    axes[4, 1].set_title('Manual Reference')
    axes[4, 1].axis('off')
    
    axes[4, 2].imshow(traditional_bin_dilated, cmap='gray')
    axes[4, 2].set_title(f'Traditional Dilated (dil={dilation_iterations})\nDice: {traditional_metrics["dice"]:.4f}')
    axes[4, 2].axis('off')
    
    # Area error analysis
    count_errors = [abs(neural_area_stats['count'] - manual_area_stats['count']), 
                   abs(traditional_area_stats['count'] - manual_area_stats['count'])]
    axes[4, 3].bar(['Neural', 'Traditional'], count_errors, color=['blue', 'orange'], alpha=0.7)
    axes[4, 3].set_title('Cell Count Error\n(vs Manual)')
    axes[4, 3].set_ylabel('Absolute Difference')
    
    # Row 6: Performance comparison and summary
    # Performance metrics bar chart
    categories = ['Precision', 'Recall', 'F1 Score', 'Dice']
    neural_values = [neural_metrics['precision'], neural_metrics['recall'], 
                    neural_metrics['f1_score'], neural_metrics['dice']]
    traditional_values = [traditional_metrics['precision'], traditional_metrics['recall'], 
                         traditional_metrics['f1_score'], traditional_metrics['dice']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[5, 0].bar(x - width/2, neural_values, width, label='Neural Net', alpha=0.7, color='blue')
    axes[5, 0].bar(x + width/2, traditional_values, width, label='Traditional', alpha=0.7, color='orange')
    axes[5, 0].set_ylabel('Score')
    axes[5, 0].set_title('Pixel-Level Performance')
    axes[5, 0].set_xticks(x)
    axes[5, 0].set_xticklabels(categories, rotation=45)
    axes[5, 0].legend()
    axes[5, 0].set_ylim(0, 1)
    
    # Area accuracy comparison
    if manual_area_stats['count'] > 0:
        mean_errors = [abs(manual_vs_neural_areas['mean_diff']), abs(manual_vs_traditional_areas['mean_diff'])]
        axes[5, 1].bar(['Neural', 'Traditional'], mean_errors, color=['blue', 'orange'], alpha=0.7)
        axes[5, 1].set_title('Mean Area Error\n(vs Manual)')
        axes[5, 1].set_ylabel('Absolute Difference (pixels)')
    else:
        axes[5, 1].text(0.5, 0.5, 'No manual cells\nfor comparison', 
                       transform=axes[5, 1].transAxes, ha='center', va='center')
        axes[5, 1].set_title('Mean Area Error\n(vs Manual)')
    
    # Distribution similarity
    if manual_area_stats['count'] > 0:
        ks_stats = [manual_vs_neural_areas['ks_statistic'], manual_vs_traditional_areas['ks_statistic']]
        axes[5, 2].bar(['Neural', 'Traditional'], ks_stats, color=['blue', 'orange'], alpha=0.7)
        axes[5, 2].set_title('Distribution Similarity\n(KS statistic, lower=better)')
        axes[5, 2].set_ylabel('KS Statistic')
    else:
        axes[5, 2].text(0.5, 0.5, 'No manual cells\nfor comparison', 
                       transform=axes[5, 2].transAxes, ha='center', va='center')
        axes[5, 2].set_title('Distribution Similarity\n(KS statistic)')
    
    # Summary text
    summary_text = f"""
COMPREHENSIVE COMPARISON SUMMARY:

Pixel-Level Performance:
  Neural Dice: {neural_metrics['dice']:.4f}
  Traditional Dice: {traditional_metrics['dice']:.4f}
  Improvement: {improvement['dice']:+.4f}

Cell Count Accuracy (from binary masks):
  Manual: {manual_area_stats['count']}
  Neural: {neural_area_stats['count']} ({neural_area_stats['count'] - manual_area_stats['count']:+d})
  Traditional: {traditional_area_stats['count']} ({traditional_area_stats['count'] - manual_area_stats['count']:+d})
  
Cell Area Accuracy:"""
    
    if manual_area_stats['count'] > 0:
        summary_text += f"""
  Neural error: {abs(manual_vs_neural_areas['mean_diff']):.1f}
  Traditional error: {abs(manual_vs_traditional_areas['mean_diff']):.1f}
  Improvement: {improvement['mean_area_error']:+.1f}

Distribution Similarity (p-value):
  Neural vs Manual: {manual_vs_neural_areas['ks_pvalue']:.3f}
  Traditional vs Manual: {manual_vs_traditional_areas['ks_pvalue']:.3f}"""
    else:
        summary_text += """
  N/A (no manual cells detected)

Distribution Similarity:
  N/A (no manual cells detected)"""
    
    summary_text += f"""

Fixed Dilation: {dilation_iterations} iterations
Neural Rotated: {'Yes' if needs_neural_rotation else 'No'}
"""
    
    axes[5, 3].text(0.05, 0.95, summary_text, transform=axes[5, 3].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[5, 3].set_title('Summary')
    axes[5, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'neural_metrics': neural_metrics,
        'traditional_metrics': traditional_metrics,
        'neural_vs_traditional': neural_vs_traditional,
        'improvement': improvement,
        'dilation_iterations': dilation_iterations,
        'neural_rotated': needs_neural_rotation,
        # Area analysis results
        'manual_area_stats': manual_area_stats,
        'neural_area_stats': neural_area_stats,
        'traditional_area_stats': traditional_area_stats,
        'manual_vs_neural_areas': manual_vs_neural_areas,
        'manual_vs_traditional_areas': manual_vs_traditional_areas,
        'manual_areas': manual_areas,
        'neural_areas': neural_areas,
        'traditional_areas': traditional_areas
    }

def main():
    """Compare neural network predictions with manual and traditional methods."""
    
    # Define paths
    neural_dir = "final-test-set/predictions"
    manual_dir = "final-test-set/groundtruth"
    traditional_dir = "final-test-set/connected_skeletons"
    
    # Find all prediction files
    neural_files = glob.glob(os.path.join(neural_dir, "*.png"))
    
    results = []
    
    for neural_file in neural_files:
        try:
            # Extract identifier from neural file
            neural_basename = os.path.basename(neural_file)
            identifier = neural_basename.replace('.png', '')  # e.g., "S_6", "2-4", etc.
            
            print(f"\nLooking for files matching identifier: {identifier}")
            
            # Find corresponding files with same base name
            manual_file = os.path.join(manual_dir, f"{identifier}.png")
            traditional_file = os.path.join(traditional_dir, f"{identifier}_connected.png")
            
            print(f"  Neural: {neural_file} (exists: {os.path.exists(neural_file)})")
            print(f"  Manual: {manual_file} (exists: {os.path.exists(manual_file)})")
            print(f"  Traditional: {traditional_file} (exists: {os.path.exists(traditional_file)})")
            
            if os.path.exists(manual_file) and os.path.exists(traditional_file):
                pair_name = f"Comparison_{identifier}"
                
                print(f"\n✓ Processing {pair_name}...")
                
                # Perform comparison
                comparison_results = analyze_neural_net_comparison(
                    neural_file, manual_file, traditional_file, pair_name
                )
                
                comparison_results['pair'] = pair_name
                comparison_results['neural_file'] = neural_file
                comparison_results['manual_file'] = manual_file
                comparison_results['traditional_file'] = traditional_file
                
                results.append(comparison_results)
                
            else:
                missing = []
                if not os.path.exists(manual_file):
                    missing.append(f"Manual: {manual_file}")
                if not os.path.exists(traditional_file):
                    missing.append(f"Traditional: {traditional_file}")
                print(f"✗ Missing files for {identifier}: {', '.join(missing)}")
                
        except Exception as e:
            print(f"Error processing {neural_file}: {e}")
            continue
    
    # Overall summary including area analysis
    if results:
        print("\n" + "="*100)
        print("OVERALL NEURAL NETWORK PERFORMANCE SUMMARY")
        print("="*100)
        
        # Calculate pixel-level averages
        neural_precisions = [r['neural_metrics']['precision'] for r in results]
        neural_recalls = [r['neural_metrics']['recall'] for r in results]
        neural_f1s = [r['neural_metrics']['f1_score'] for r in results]
        neural_dices = [r['neural_metrics']['dice'] for r in results]
        
        traditional_precisions = [r['traditional_metrics']['precision'] for r in results]
        traditional_recalls = [r['traditional_metrics']['recall'] for r in results]
        traditional_f1s = [r['traditional_metrics']['f1_score'] for r in results]
        traditional_dices = [r['traditional_metrics']['dice'] for r in results]
        
        # Calculate area-level averages
        manual_counts = [r['manual_area_stats']['count'] for r in results]
        neural_counts = [r['neural_area_stats']['count'] for r in results]
        traditional_counts = [r['traditional_area_stats']['count'] for r in results]
        
        manual_means = [r['manual_area_stats']['mean'] for r in results]
        neural_means = [r['neural_area_stats']['mean'] for r in results]
        traditional_means = [r['traditional_area_stats']['mean'] for r in results]
        
        count_errors_neural = [abs(n - m) for n, m in zip(neural_counts, manual_counts)]
        count_errors_traditional = [abs(t - m) for t, m in zip(traditional_counts, manual_counts)]
        
        mean_errors_neural = [abs(r['manual_vs_neural_areas']['mean_diff']) for r in results]
        mean_errors_traditional = [abs(r['manual_vs_traditional_areas']['mean_diff']) for r in results]
        
        improvements_precision = [r['improvement']['precision'] for r in results]
        improvements_recall = [r['improvement']['recall'] for r in results]
        improvements_f1 = [r['improvement']['f1_score'] for r in results]
        improvements_dice = [r['improvement']['dice'] for r in results]
        
        print(f"PIXEL-LEVEL Average Performance (n={len(results)} comparisons):")
        print(f"{'Method':<12} | {'Precision':<9} | {'Recall':<9} | {'F1':<9} | {'Dice':<9}")
        print("-" * 60)
        print(f"{'Neural Net':<12} | {np.mean(neural_precisions):8.4f} | {np.mean(neural_recalls):8.4f} | "
              f"{np.mean(neural_f1s):8.4f} | {np.mean(neural_dices):8.4f}")
        print(f"{'Traditional':<12} | {np.mean(traditional_precisions):8.4f} | {np.mean(traditional_recalls):8.4f} | "
              f"{np.mean(traditional_f1s):8.4f} | {np.mean(traditional_dices):8.4f}")
        
        print(f"\nCELL-LEVEL Average Performance:")
        print(f"Average cell count accuracy:")
        print(f"  Neural error:        {np.mean(count_errors_neural):.1f} ± {np.std(count_errors_neural):.1f} cells")
        print(f"  Traditional error:   {np.mean(count_errors_traditional):.1f} ± {np.std(count_errors_traditional):.1f} cells")
        
        print(f"\nAverage cell area accuracy:")
        print(f"  Neural error:        {np.mean(mean_errors_neural):.1f} ± {np.std(mean_errors_neural):.1f} pixels")
        print(f"  Traditional error:   {np.mean(mean_errors_traditional):.1f} ± {np.std(mean_errors_traditional):.1f} pixels")
        
        print(f"\nAverage Improvements (Neural - Traditional):")
        print(f"  Precision: {np.mean(improvements_precision):+.4f} ± {np.std(improvements_precision):.4f}")
        print(f"  Recall: {np.mean(improvements_recall):+.4f} ± {np.std(improvements_recall):.4f}")
        print(f"  F1 Score: {np.mean(improvements_f1):+.4f} ± {np.std(improvements_f1):.4f}")
        print(f"  Dice: {np.mean(improvements_dice):+.4f} ± {np.std(improvements_dice):.4f}")
        
        # Cell-level improvements
        count_improvement = np.mean(count_errors_traditional) - np.mean(count_errors_neural)
        area_improvement = np.mean(mean_errors_traditional) - np.mean(mean_errors_neural)
        
        print(f"  Cell count accuracy: {count_improvement:+.1f} cells")
        print(f"  Cell area accuracy:  {area_improvement:+.1f} pixels")
        
        # Individual results table
        print(f"\nIndividual Results:")
        print(f"{'Image':<8} | {'Neural Dice':<11} | {'Trad Dice':<11} | {'Neural Cells':<12} | {'Trad Cells':<11} | {'Manual Cells':<12}")
        print("-" * 80)
        for r in results:
            identifier = r['pair'].replace('Comparison_', '')
            neural_dice = r['neural_metrics']['dice']
            trad_dice = r['traditional_metrics']['dice']
            neural_cells = r['neural_area_stats']['count']
            trad_cells = r['traditional_area_stats']['count']
            manual_cells = r['manual_area_stats']['count']
            print(f"{identifier:<8} | {neural_dice:10.4f} | {trad_dice:10.4f} | {neural_cells:11d} | {trad_cells:10d} | {manual_cells:11d}")
        
        # Determine statistical significance if we have enough samples
        if len(results) >= 3:
            from scipy import stats
            
            _, p_precision = stats.ttest_rel(neural_precisions, traditional_precisions)
            _, p_recall = stats.ttest_rel(neural_recalls, traditional_recalls)
            _, p_f1 = stats.ttest_rel(neural_f1s, traditional_f1s)
            _, p_dice = stats.ttest_rel(neural_dices, traditional_dices)
            _, p_count = stats.ttest_rel(count_errors_neural, count_errors_traditional)
            _, p_area = stats.ttest_rel(mean_errors_neural, mean_errors_traditional)
            
            print(f"\nStatistical Significance (paired t-test p-values):")
            print(f"  Precision: p = {p_precision:.4f} {'***' if p_precision < 0.001 else '**' if p_precision < 0.01 else '*' if p_precision < 0.05 else 'ns'}")
            print(f"  Recall: p = {p_recall:.4f} {'***' if p_recall < 0.001 else '**' if p_recall < 0.01 else '*' if p_recall < 0.05 else 'ns'}")
            print(f"  F1 Score: p = {p_f1:.4f} {'***' if p_f1 < 0.001 else '**' if p_f1 < 0.01 else '*' if p_f1 < 0.05 else 'ns'}")
            print(f"  Dice: p = {p_dice:.4f} {'***' if p_dice < 0.001 else '**' if p_dice < 0.01 else '*' if p_dice < 0.05 else 'ns'}")
            print(f"  Cell count accuracy: p = {p_count:.4f} {'***' if p_count < 0.001 else '**' if p_count < 0.01 else '*' if p_count < 0.05 else 'ns'}")
            print(f"  Cell area accuracy: p = {p_area:.4f} {'***' if p_area < 0.001 else '**' if p_area < 0.01 else '*' if p_area < 0.05 else 'ns'}")
    else:
        print("No valid file pairs found for comparison!")

if __name__ == "__main__":
    main()