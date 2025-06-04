from analysis_based_on_area import find_enclosed_areas_per_region, find_areas_from_segmentation, find_areas_from_skeleton, calculate_area_statistics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
import glob
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu

def get_dilation_iterations(file_path):
    """Get appropriate dilation iterations based on zoom level in filename."""
    basename = os.path.basename(file_path)
    if "6" in basename:  # 6 is in the filename of the zoomed pictures
        return 5
    elif "7" in basename:  # 7 is in the filename of the a bit less zoomed pictures
        return 4
    else:
        return 2

def create_error_visualization_with_adaptive_dilation(manual_path, ilastik_path, connected_path, pair_name):
    """Create error visualization with adaptive dilation based on zoom level."""
    
    # Load images
    manual_img = Image.open(manual_path).convert('RGB')
    ilastik_img = Image.open(ilastik_path).convert('L')
    connected_img = Image.open(connected_path).convert('L')
    
    manual_array = np.array(manual_img)
    ilastik_array = np.array(ilastik_img)
    connected_array = np.array(connected_img)
    
    # Get adaptive dilation iterations
    manual_dilation = get_dilation_iterations(manual_path)
    connected_dilation = get_dilation_iterations(connected_path)
    
    print(f"  Using dilation - Manual: {manual_dilation}, Connected: {connected_dilation}")
    
    # Create binary mask for manual trace (red pixels only)
    manual_bin = (
        (manual_array[:, :, 0] > 10) &
        (manual_array[:, :, 1] < 127) &
        (manual_array[:, :, 2] < 127)
    ).astype(np.uint8)
    
    # Dilate manual trace with adaptive iterations
    manual_bin_dilated = binary_dilation(manual_bin, iterations=manual_dilation).astype(np.uint8)
    
    # Create binary masks for predictions
    cell_wall_mask = (ilastik_array == 1).astype(np.uint8)
    connected_bin = (connected_array > 128).astype(np.uint8)
    # Dilate connected with adaptive iterations
    connected_bin_dilated = binary_dilation(connected_bin, iterations=connected_dilation).astype(np.uint8)
    
    # Calculate error masks for Connected vs Manual
    tp_mask_connected = (manual_bin_dilated == 1) & (connected_bin_dilated == 1)
    fp_mask_connected = (manual_bin_dilated == 0) & (connected_bin_dilated == 1)
    fn_mask_connected = (manual_bin_dilated == 1) & (connected_bin_dilated == 0)
    
    # Calculate error masks for Ilastik vs Manual
    tp_mask_ilastik = (manual_bin_dilated == 1) & (cell_wall_mask == 1)
    fp_mask_ilastik = (manual_bin_dilated == 0) & (cell_wall_mask == 1)
    fn_mask_ilastik = (manual_bin_dilated == 1) & (cell_wall_mask == 0)
    
    # Create error visualizations
    error_vis_connected = np.zeros((manual_bin_dilated.shape[0], manual_bin_dilated.shape[1], 3))
    error_vis_connected[tp_mask_connected, 1] = 1.0  # Green for correct
    error_vis_connected[fp_mask_connected, 0] = 1.0  # Red for overcorrection
    error_vis_connected[fn_mask_connected, 2] = 1.0  # Blue for undercorrection
    
    error_vis_ilastik = np.zeros((manual_bin_dilated.shape[0], manual_bin_dilated.shape[1], 3))
    error_vis_ilastik[tp_mask_ilastik, 1] = 1.0  # Green for correct
    error_vis_ilastik[fp_mask_ilastik, 0] = 1.0  # Red for overcorrection
    error_vis_ilastik[fn_mask_ilastik, 2] = 1.0  # Blue for undercorrection
    
    # Calculate statistics
    stats_connected = {
        'TP': np.sum(tp_mask_connected),
        'FP': np.sum(fp_mask_connected),
        'FN': np.sum(fn_mask_connected),
        'manual_pixels': np.sum(manual_bin_dilated)
    }
    
    stats_ilastik = {
        'TP': np.sum(tp_mask_ilastik),
        'FP': np.sum(fp_mask_ilastik),
        'FN': np.sum(fn_mask_ilastik),
        'manual_pixels': np.sum(manual_bin_dilated)
    }
    
    stats_connected['fp_rate'] = (stats_connected['FP'] / stats_connected['manual_pixels'] * 100) if stats_connected['manual_pixels'] > 0 else 0
    stats_connected['fn_rate'] = (stats_connected['FN'] / stats_connected['manual_pixels'] * 100) if stats_connected['manual_pixels'] > 0 else 0
    
    stats_ilastik['fp_rate'] = (stats_ilastik['FP'] / stats_ilastik['manual_pixels'] * 100) if stats_ilastik['manual_pixels'] > 0 else 0
    stats_ilastik['fn_rate'] = (stats_ilastik['FN'] / stats_ilastik['manual_pixels'] * 100) if stats_ilastik['manual_pixels'] > 0 else 0
    
    return error_vis_connected, error_vis_ilastik, stats_connected, stats_ilastik, manual_array, ilastik_array, connected_array

def analyze_enclosed_areas_adaptive(manual_array, connected_array, manual_path, connected_path, pair_name):
    """Analyze enclosed areas with adaptive dilation."""
    
    # Get adaptive dilation iterations
    connected_dilation = get_dilation_iterations(connected_path)
    
    # Process manual trace
    manual_bin = (
        (manual_array[:, :, 0] > 10) &
        (manual_array[:, :, 1] < 127) &
        (manual_array[:, :, 2] < 127)
    ).astype(bool)
    
    # Process connected skeleton with adaptive dilation
    connected_bin = (connected_array > 128).astype(bool)
    connected_dilated = binary_dilation(connected_bin, iterations=connected_dilation)
    
    # Find enclosed areas in manual trace
    manual_inverted = ~manual_bin
    from scipy.ndimage import label
    manual_labeled, manual_num = label(manual_inverted, structure=np.ones((3,3)))
    
    # Find enclosed areas in connected skeleton
    connected_inverted = ~connected_dilated
    connected_labeled, connected_num = label(connected_inverted, structure=np.ones((3,3)))
    
    # Remove border-touching regions
    def remove_border_regions(labeled_img, num_features):
        border_labels = set(np.unique(np.concatenate([
            labeled_img[0, :], labeled_img[-1, :], 
            labeled_img[:, 0], labeled_img[:, -1]
        ])))
        
        areas = []
        clean_labeled = labeled_img.copy()
        
        for i in range(1, num_features + 1):
            if i not in border_labels:
                area = np.sum(labeled_img == i)
                if area > 50:  # Minimum area threshold
                    areas.append(area)
            else:
                clean_labeled[labeled_img == i] = 0
        
        return areas, clean_labeled
    
    manual_areas, manual_clean = remove_border_regions(manual_labeled, manual_num)
    connected_areas, connected_clean = remove_border_regions(connected_labeled, connected_num)
    
    return {
        'manual_areas': manual_areas,
        'connected_areas': connected_areas,
        'manual_labeled': manual_clean,
        'connected_labeled': connected_clean,
        'manual_count': len(manual_areas),
        'connected_count': len(connected_areas)
    }

def create_cell_analysis_square_adaptive(manual_path, ilastik_path, connected_path, pair_name):
    """Create 2x2 square figure with adaptive dilation for cell detection analysis."""
    
    # Load images
    manual_img = Image.open(manual_path).convert('RGB')
    ilastik_img = Image.open(ilastik_path).convert('L')
    connected_img = Image.open(connected_path).convert('L')
    
    manual_array = np.array(manual_img)
    ilastik_array = np.array(ilastik_img)
    connected_array = np.array(connected_img)
    
    # Use your biological analysis functions with adaptive processing
    print(f"Analyzing {pair_name}...")
    
    # Get zoom level info
    manual_dilation = get_dilation_iterations(manual_path)
    connected_dilation = get_dilation_iterations(connected_path)
    print(f"  Zoom-adaptive dilation - Manual: {manual_dilation}, Connected: {connected_dilation}")
    
    # Use the analysis functions (assuming they handle adaptive dilation internally or we modify them)
    manual_areas, manual_labeled, manual_labels = find_enclosed_areas_per_region(manual_array)
    ilastik_areas, ilastik_labeled, ilastik_labels = find_areas_from_segmentation(ilastik_array, target_label=2)
    
    # For connected, we'll need to do custom processing since we need adaptive dilation
    connected_bin = (connected_array > 128).astype(bool)
    connected_dilated = binary_dilation(connected_bin, iterations=connected_dilation)
    
    # Find enclosed areas in connected skeleton
    connected_inverted = ~connected_dilated
    from scipy.ndimage import label
    connected_labeled_temp, connected_num = label(connected_inverted, structure=np.ones((3,3)))
    
    # Remove border-touching regions for connected
    border_labels = set(np.unique(np.concatenate([
        connected_labeled_temp[0, :], connected_labeled_temp[-1, :], 
        connected_labeled_temp[:, 0], connected_labeled_temp[:, -1]
    ])))
    
    connected_areas = []
    connected_labels = []
    connected_labeled = connected_labeled_temp.copy()
    
    for i in range(1, connected_num + 1):
        if i not in border_labels:
            area = np.sum(connected_labeled_temp == i)
            if area > 50:  # Minimum area threshold
                connected_areas.append(area)
                connected_labels.append(i)
        else:
            connected_labeled[connected_labeled_temp == i] = 0
    
    # Calculate statistics
    manual_stats = calculate_area_statistics(manual_areas)
    ilastik_stats = calculate_area_statistics(ilastik_areas)
    connected_stats = calculate_area_statistics(connected_areas)
    
    # Create 2x2 square figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'{pair_name} - Cell Detection Analysis (Adaptive Dilation: M={manual_dilation}, C={connected_dilation})', 
                fontsize=14, fontweight='bold')
    
    # Top left: Manual cells with colored regions
    colored_manual = np.zeros_like(manual_array)
    colored_manual[:,:,0] = manual_array[:,:,0]  # Keep red channel
    if len(manual_labels) > 0:
        for i, label_id in enumerate(manual_labels):
            mask = manual_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_manual[mask] = [int(c*255) for c in color[:3]]
    
    axes[0, 0].imshow(colored_manual)
    axes[0, 0].set_title(f'Manual Cells Detected\nCount: {manual_stats["count"]}', fontsize=14)
    axes[0, 0].axis('off')
    
    # Top right: Ilastik cells with colored regions
    colored_ilastik = np.zeros((ilastik_array.shape[0], ilastik_array.shape[1], 3))
    if len(ilastik_labels) > 0:
        for i, label_id in enumerate(ilastik_labels):
            mask = ilastik_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_ilastik[mask] = color[:3]
    
    axes[0, 1].imshow(colored_ilastik)
    axes[0, 1].set_title(f'Ilastik Cells Detected\nCount: {ilastik_stats["count"]}', fontsize=14)
    axes[0, 1].axis('off')
    
    # Bottom left: Connected cells with colored regions
    colored_connected = np.zeros((connected_array.shape[0], connected_array.shape[1], 3))
    if len(connected_labels) > 0:
        for i, label_id in enumerate(connected_labels):
            mask = connected_labeled == label_id
            color = plt.cm.Set3(i % 12)
            colored_connected[mask] = color[:3]
    
    axes[1, 0].imshow(colored_connected)
    axes[1, 0].set_title(f'Connected Cells Detected\nCount: {connected_stats["count"]}', fontsize=14)
    axes[1, 0].axis('off')
    
    # Bottom right: Cell area distribution comparison
    axes[1, 1].clear()
    if manual_areas:
        axes[1, 1].hist(manual_areas, bins=20, alpha=0.6, color='red', 
                       label=f'Manual (n={len(manual_areas)})', density=True)
        axes[1, 1].axvline(manual_stats['mean'], color='red', linestyle='--', 
                          alpha=0.8, linewidth=2)
    if ilastik_areas:
        axes[1, 1].hist(ilastik_areas, bins=20, alpha=0.6, color='orange', 
                       label=f'Ilastik (n={len(ilastik_areas)})', density=True)
        axes[1, 1].axvline(ilastik_stats['mean'], color='orange', linestyle='--', 
                          alpha=0.8, linewidth=2)
    if connected_areas:
        axes[1, 1].hist(connected_areas, bins=20, alpha=0.6, color='blue', 
                       label=f'Connected (n={len(connected_areas)})', density=True)
        axes[1, 1].axvline(connected_stats['mean'], color='blue', linestyle='--', 
                          alpha=0.8, linewidth=2)
    
    axes[1, 1].set_xlabel('Cell Area (pixels)', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Cell Size Distribution', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, {
        'manual_stats': manual_stats,
        'ilastik_stats': ilastik_stats,
        'connected_stats': connected_stats,
        'manual_areas': manual_areas,
        'ilastik_areas': ilastik_areas,
        'connected_areas': connected_areas,
        'dilation_info': {
            'manual_dilation': manual_dilation,
            'connected_dilation': connected_dilation
        }
    }

def create_final_summary_graph(all_results):
    """Create final summary graph with combined metrics and statistical tests."""
    
    # Collect all areas for statistical analysis
    all_manual_areas = []
    all_ilastik_areas = []
    all_connected_areas = []
    
    for result in all_results:
        all_manual_areas.extend(result['manual_areas'])
        all_ilastik_areas.extend(result['ilastik_areas'])
        all_connected_areas.extend(result['connected_areas'])
    
    # Statistical tests
    print("\nPerforming statistical tests on cell size distributions...")
    
    # Kolmogorov-Smirnov tests
    if len(all_manual_areas) > 0 and len(all_ilastik_areas) > 0:
        ks_stat_mi, ks_p_mi = ks_2samp(all_manual_areas, all_ilastik_areas)
    else:
        ks_stat_mi, ks_p_mi = 0, 1
        
    if len(all_manual_areas) > 0 and len(all_connected_areas) > 0:
        ks_stat_mc, ks_p_mc = ks_2samp(all_manual_areas, all_connected_areas)
    else:
        ks_stat_mc, ks_p_mc = 0, 1
        
    if len(all_ilastik_areas) > 0 and len(all_connected_areas) > 0:
        ks_stat_ic, ks_p_ic = ks_2samp(all_ilastik_areas, all_connected_areas)
    else:
        ks_stat_ic, ks_p_ic = 0, 1
    
    # Mann-Whitney U tests (non-parametric alternative)
    if len(all_manual_areas) > 0 and len(all_ilastik_areas) > 0:
        mw_stat_mi, mw_p_mi = mannwhitneyu(all_manual_areas, all_ilastik_areas, alternative='two-sided')
    else:
        mw_stat_mi, mw_p_mi = 0, 1
        
    if len(all_manual_areas) > 0 and len(all_connected_areas) > 0:
        mw_stat_mc, mw_p_mc = mannwhitneyu(all_manual_areas, all_connected_areas, alternative='two-sided')
    else:
        mw_stat_mc, mw_p_mc = 0, 1
    
    # Create comprehensive summary figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    fig.suptitle('Final Summary: Cell Detection Analysis with Adaptive Dilation', fontsize=18, fontweight='bold')
    
    # Panel 1: Cell count comparison
    ax1 = fig.add_subplot(gs[0, 0])
    cell_counts = []
    methods = ['Manual', 'Ilastik', 'Connected']
    
    total_manual = sum(r['manual_stats']['count'] for r in all_results)
    total_ilastik = sum(r['ilastik_stats']['count'] for r in all_results)
    total_connected = sum(r['connected_stats']['count'] for r in all_results)
    
    cell_counts = [total_manual, total_ilastik, total_connected]
    colors = ['red', 'orange', 'blue']
    
    bars = ax1.bar(methods, cell_counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Total Cell Count', fontsize=12)
    ax1.set_title('Total Cells Detected\nAcross All Images', fontsize=14)
    
    # Add count labels on bars
    for bar, count in zip(bars, cell_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cell_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Cell count accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if total_manual > 0:
        ilastik_accuracy = min(total_manual, total_ilastik) / max(total_manual, total_ilastik) * 100
        connected_accuracy = min(total_manual, total_connected) / max(total_manual, total_connected) * 100
    else:
        ilastik_accuracy = connected_accuracy = 0
    
    accuracies = [100, ilastik_accuracy, connected_accuracy]  # Manual is 100% by definition
    bars2 = ax2.bar(methods, accuracies, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Cell Counting Accuracy\nvs Manual Reference', fontsize=14)
    ax2.set_ylim(0, 105)
    
    # Add accuracy labels
    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Distribution comparison with statistical tests
    ax3 = fig.add_subplot(gs[0, 2])
    
    if all_manual_areas:
        ax3.hist(all_manual_areas, bins=30, alpha=0.6, color='red', 
                label=f'Manual (n={len(all_manual_areas)})', density=True)
        manual_mean = np.mean(all_manual_areas)
        ax3.axvline(manual_mean, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    if all_ilastik_areas:
        ax3.hist(all_ilastik_areas, bins=30, alpha=0.6, color='orange', 
                label=f'Ilastik (n={len(all_ilastik_areas)})', density=True)
        ilastik_mean = np.mean(all_ilastik_areas)
        ax3.axvline(ilastik_mean, color='orange', linestyle='--', alpha=0.8, linewidth=2)
    
    if all_connected_areas:
        ax3.hist(all_connected_areas, bins=30, alpha=0.6, color='blue', 
                label=f'Connected (n={len(all_connected_areas)})', density=True)
        connected_mean = np.mean(all_connected_areas)
        ax3.axvline(connected_mean, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Cell Area (pixels)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Combined Cell Size Distribution', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Per-image accuracy plot with dilation info
    ax4 = fig.add_subplot(gs[1, :])
    
    image_names = [r['pair'] for r in all_results]
    manual_counts = [r['manual_stats']['count'] for r in all_results]
    ilastik_counts = [r['ilastik_stats']['count'] for r in all_results]
    connected_counts = [r['connected_stats']['count'] for r in all_results]
    
    # Add dilation info to labels
    image_labels = []
    for i, result in enumerate(all_results):
        dilation_info = result.get('dilation_info', {})
        m_dil = dilation_info.get('manual_dilation', '?')
        c_dil = dilation_info.get('connected_dilation', '?')
        image_labels.append(f"{result['pair']}\n(M:{m_dil},C:{c_dil})")
    
    x = np.arange(len(image_names))
    width = 0.25
    
    ax4.bar(x - width, manual_counts, width, label='Manual', color='red', alpha=0.7)
    ax4.bar(x, ilastik_counts, width, label='Ilastik', color='orange', alpha=0.7)
    ax4.bar(x + width, connected_counts, width, label='Connected', color='blue', alpha=0.7)
    
    ax4.set_xlabel('Images (with dilation iterations)', fontsize=12)
    ax4.set_ylabel('Cell Count', fontsize=12)
    ax4.set_title('Cell Count Comparison Per Image (Adaptive Dilation)', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(image_labels, rotation=45, ha='right', fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Statistical test results
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    
    stats_text = f"""
    KOLMOGOROV-SMIRNOV TESTS:
    
    Manual vs Ilastik:
    • K-S statistic: {ks_stat_mi:.4f}
    • p-value: {ks_p_mi:.4f}
    • Significant: {'Yes' if ks_p_mi < 0.05 else 'No'}
    
    Manual vs Connected:
    • K-S statistic: {ks_stat_mc:.4f}
    • p-value: {ks_p_mc:.4f}
    • Significant: {'Yes' if ks_p_mc < 0.05 else 'No'}
    
    Ilastik vs Connected:
    • K-S statistic: {ks_stat_ic:.4f}
    • p-value: {ks_p_ic:.4f}
    • Significant: {'Yes' if ks_p_ic < 0.05 else 'No'}
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # Panel 6: Mann-Whitney U test results
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    mw_text = f"""
    MANN-WHITNEY U TESTS:
    
    Manual vs Ilastik:
    • U statistic: {mw_stat_mi:.0f}
    • p-value: {mw_p_mi:.4f}
    • Significant: {'Yes' if mw_p_mi < 0.05 else 'No'}
    
    Manual vs Connected:
    • U statistic: {mw_stat_mc:.0f}
    • p-value: {mw_p_mc:.4f}
    • Significant: {'Yes' if mw_p_mc < 0.05 else 'No'}
    
    Note: Significant p < 0.05 indicates
    distributions are different
    """
    
    ax6.text(0.05, 0.95, mw_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    # Panel 7: Summary statistics with dilation info
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Count dilation usage
    dilation_counts = {'2': 0, '4': 0, '5': 0}
    for result in all_results:
        dilation_info = result.get('dilation_info', {})
        c_dil = str(dilation_info.get('connected_dilation', '2'))
        if c_dil in dilation_counts:
            dilation_counts[c_dil] += 1
    
    summary_stats = f"""
    SUMMARY STATISTICS:
    
    Total Images Analyzed: {len(all_results)}
    
    Dilation Usage:
    • 2 iterations: {dilation_counts['2']} images
    • 4 iterations: {dilation_counts['4']} images  
    • 5 iterations: {dilation_counts['5']} images
    
    Cell Area Statistics:
    Manual:    μ={np.mean(all_manual_areas):.1f}, σ={np.std(all_manual_areas):.1f}
    Ilastik:   μ={np.mean(all_ilastik_areas):.1f}, σ={np.std(all_ilastik_areas):.1f}
    Connected: μ={np.mean(all_connected_areas):.1f}, σ={np.std(all_connected_areas):.1f}
    
    Counting Errors:
    Ilastik:   {abs(total_ilastik - total_manual)} cells ({abs(total_ilastik - total_manual)/total_manual*100:.1f}%)
    Connected: {abs(total_connected - total_manual)} cells ({abs(total_connected - total_manual)/total_manual*100:.1f}%)
    
    Best Method: {'Connected' if abs(total_connected - total_manual) < abs(total_ilastik - total_manual) else 'Ilastik'}
    """
    
    ax7.text(0.05, 0.95, summary_stats, transform=ax7.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    
    plt.tight_layout()
    
    return fig, {
        'ks_tests': {
            'manual_vs_ilastik': (ks_stat_mi, ks_p_mi),
            'manual_vs_connected': (ks_stat_mc, ks_p_mc),
            'ilastik_vs_connected': (ks_stat_ic, ks_p_ic)
        },
        'mw_tests': {
            'manual_vs_ilastik': (mw_stat_mi, mw_p_mi),
            'manual_vs_connected': (mw_stat_mc, mw_p_mc)
        },
        'total_counts': {
            'manual': total_manual,
            'ilastik': total_ilastik,
            'connected': total_connected
        },
        'all_areas': {
            'manual': all_manual_areas,
            'ilastik': all_ilastik_areas,
            'connected': all_connected_areas
        }
    }

def process_cell_analysis_adaptive():
    """Process all images and create cell analysis figures with adaptive dilation."""
    
    # Find manual trace files
    manual_pattern = "manual_trace/manual-trace-*.png"
    manual_files = glob.glob(manual_pattern)
    
    if not manual_files:
        print("No manual trace files found!")
        return
    
    # Create output directory
    output_dir = 'cell_analysis_adaptive'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating cell detection analysis with adaptive dilation...")
    
    all_results = []
    
    for manual_file in manual_files:
        try:
            identifier = manual_file.replace("manual_trace/manual-trace-", "").replace(".png", "")
            ilastik_file = f"simple_segmentation/simple-segmentation-{identifier}.png"
            connected_file = f"connected_skeletons/{identifier}_connected.png"
            
            if os.path.exists(ilastik_file) and os.path.exists(connected_file):
                pair_name = f"Image_{identifier}"
                print(f"Processing {pair_name}...")
                
                # Create cell analysis figure with adaptive dilation
                fig, results = create_cell_analysis_square_adaptive(
                    manual_file, ilastik_file, connected_file, pair_name
                )
                plt.savefig(f"{output_dir}/{identifier}_cell_analysis.png", 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Store results
                results['pair'] = pair_name
                results['identifier'] = identifier
                all_results.append(results)
                
                dilation_info = results['dilation_info']
                print(f"✅ {pair_name} completed (M:{dilation_info['manual_dilation']}, C:{dilation_info['connected_dilation']})")
                print(f"   Cells - Manual: {results['manual_stats']['count']}, "
                      f"Ilastik: {results['ilastik_stats']['count']}, "
                      f"Connected: {results['connected_stats']['count']}")
                
        except Exception as e:
            print(f"❌ Error processing {manual_file}: {e}")
    
    # Create final summary graph
    if all_results:
        print("\nCreating final summary analysis with adaptive dilation info...")
        fig_summary, stats_summary = create_final_summary_graph(all_results)
        plt.savefig(f"{output_dir}/FINAL_SUMMARY_ADAPTIVE.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig_summary)
        
        # Print summary with dilation info
        print("\n" + "="*90)
        print("ADAPTIVE DILATION CELL DETECTION SUMMARY")
        print("="*90)
        
        print(f"{'Image':15} | {'Manual':8} | {'Ilastik':8} | {'Connected':10} | {'Dilation':12}")
        print("-"*65)
        
        total_manual_cells = 0
        total_ilastik_cells = 0
        total_connected_cells = 0
        
        for result in all_results:
            manual_count = result['manual_stats']['count']
            ilastik_count = result['ilastik_stats']['count']
            connected_count = result['connected_stats']['count']
            dilation_info = result['dilation_info']
            dilation_str = f"M:{dilation_info['manual_dilation']},C:{dilation_info['connected_dilation']}"
            
            print(f"{result['pair']:15} | {manual_count:8} | {ilastik_count:8} | {connected_count:10} | {dilation_str:12}")
            
            total_manual_cells += manual_count
            total_ilastik_cells += ilastik_count
            total_connected_cells += connected_count
        
        print("-"*65)
        print(f"{'TOTAL':15} | {total_manual_cells:8} | {total_ilastik_cells:8} | {total_connected_cells:10} | {'Adaptive':12}")
        
        # Calculate accuracies
        ilastik_accuracy = min(total_manual_cells, total_ilastik_cells) / max(total_manual_cells, total_ilastik_cells) * 100
        connected_accuracy = min(total_manual_cells, total_connected_cells) / max(total_manual_cells, total_connected_cells) * 100
        
        print(f"\nCell counting accuracies with adaptive dilation:")
        print(f"Ilastik vs Manual: {ilastik_accuracy:.1f}%")
        print(f"Connected vs Manual: {connected_accuracy:.1f}%")
        
        # Dilation usage summary
        dilation_usage = {}
        for result in all_results:
            dilation_info = result['dilation_info']
            key = f"M:{dilation_info['manual_dilation']},C:{dilation_info['connected_dilation']}"
            dilation_usage[key] = dilation_usage.get(key, 0) + 1
        
        print(f"\nDilation usage summary:")
        for combo, count in dilation_usage.items():
            print(f"  {combo}: {count} images")
    
    print(f"\nAdaptive cell analysis figures saved to '{output_dir}/' directory")
    print(f"📊 FINAL ADAPTIVE SUMMARY: {output_dir}/FINAL_SUMMARY_ADAPTIVE.png")

if __name__ == "__main__":
    process_cell_analysis_adaptive()