from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label
import os
import glob

def create_error_visualization_mosaic(manual_path, ilastik_path, connected_path, pair_name):
    """Create error visualization for mosaic images."""
    
    # Load images
    manual_img = Image.open(manual_path).convert('RGB')
    ilastik_img = Image.open(ilastik_path).convert('L')
    connected_img = Image.open(connected_path).convert('L')
    
    # Convert to numpy arrays
    manual_array = np.array(manual_img)
    ilastik_array = np.array(ilastik_img)
    connected_array = np.array(connected_img)
    
    # Create binary mask for manual trace (red pixels only)
    manual_bin = (
        (manual_array[:, :, 0] > 10) &
        (manual_array[:, :, 1] < 127) &
        (manual_array[:, :, 2] < 127)
    ).astype(np.uint8)
    
    # Dilate manual trace
    manual_bin_dilated = binary_dilation(manual_bin, iterations=2).astype(np.uint8)
    
    # Create binary masks for predictions
    cell_wall_mask = (ilastik_array == 1).astype(np.uint8)
    connected_bin = (connected_array > 128).astype(np.uint8)
    connected_bin_dilated = binary_dilation(connected_bin, iterations=4).astype(np.uint8)
    
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

def analyze_enclosed_areas(manual_array, connected_array, pair_name):
    """Analyze enclosed areas in both manual and connected images."""
    
    # Process manual trace
    manual_bin = (
        (manual_array[:, :, 0] > 10) &
        (manual_array[:, :, 1] < 127) &
        (manual_array[:, :, 2] < 127)
    ).astype(bool)
    
    # Process connected skeleton
    connected_bin = (connected_array > 128).astype(bool)
    connected_dilated = binary_dilation(connected_bin, iterations=2)
    
    # Find enclosed areas in manual trace
    manual_inverted = ~manual_bin
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

def create_comprehensive_figure(manual_path, ilastik_path, connected_path, pair_name):
    """Create a comprehensive figure with all analysis."""
    
    # Get error visualizations
    error_vis_connected, error_vis_ilastik, stats_connected, stats_ilastik, manual_array, ilastik_array, connected_array = create_error_visualization_mosaic(
        manual_path, ilastik_path, connected_path, pair_name
    )
    
    # Get area analysis
    area_analysis = analyze_enclosed_areas(manual_array, connected_array, pair_name)
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Comprehensive Analysis: {pair_name}', fontsize=18, fontweight='bold')
    
    # Row 1: Original images
    axes[0, 0].imshow(manual_array)
    axes[0, 0].set_title('Manual Trace\n(Ground Truth)', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ilastik_array, cmap='viridis')
    axes[0, 1].set_title('Ilastik Segmentation', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(connected_array, cmap='gray')
    axes[0, 2].set_title('Connected Skeleton', fontsize=14)
    axes[0, 2].axis('off')
    
    # Overlay
    axes[0, 3].imshow(manual_array, alpha=0.7)
    axes[0, 3].imshow(connected_array, cmap='gray', alpha=0.5)
    axes[0, 3].set_title('Overlay:\nManual + Connected', fontsize=14)
    axes[0, 3].axis('off')
    
    # Row 2: Error analysis
    axes[1, 0].imshow(error_vis_ilastik)
    axes[1, 0].set_title(f'Ilastik Error Analysis\nCorrect: {stats_ilastik["TP"]} | Over: {stats_ilastik["FP"]} | Under: {stats_ilastik["FN"]}', 
                        fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(error_vis_connected)
    axes[1, 1].set_title(f'Connected Error Analysis\nCorrect: {stats_connected["TP"]} | Over: {stats_connected["FP"]} | Under: {stats_connected["FN"]}', 
                        fontsize=12)
    axes[1, 1].axis('off')
    
    # Performance comparison
    categories = ['Overcorrection %', 'Undercorrection %']
    ilastik_rates = [stats_ilastik['fp_rate'], stats_ilastik['fn_rate']]
    connected_rates = [stats_connected['fp_rate'], stats_connected['fn_rate']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, ilastik_rates, width, label='Ilastik', alpha=0.7, color='orange')
    axes[1, 2].bar(x + width/2, connected_rates, width, label='Connected', alpha=0.7, color='blue')
    axes[1, 2].set_ylabel('Error Rate (%)', fontsize=12)
    axes[1, 2].set_title('Error Rate Comparison', fontsize=14)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(categories)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Summary text
    improvement_text = f"""
    PERFORMANCE COMPARISON:
    
    Ilastik:
    - Overcorrection: {stats_ilastik['fp_rate']:.1f}%
    - Undercorrection: {stats_ilastik['fn_rate']:.1f}%
    
    Connected:
    - Overcorrection: {stats_connected['fp_rate']:.1f}%
    - Undercorrection: {stats_connected['fn_rate']:.1f}%
    
    Improvement:
    - Over: {stats_connected['fp_rate'] - stats_ilastik['fp_rate']:+.1f}%
    - Under: {stats_connected['fn_rate'] - stats_ilastik['fn_rate']:+.1f}%
    """
    
    axes[1, 3].text(0.05, 0.5, improvement_text, transform=axes[1, 3].transAxes, 
                    fontsize=10, verticalalignment='center', fontfamily='monospace')
    axes[1, 3].set_title('Performance Summary', fontsize=14)
    axes[1, 3].axis('off')
    
    # Row 3: Area analysis
    # Manual areas visualization
    manual_areas_vis = np.zeros_like(manual_array)
    unique_labels = np.unique(area_analysis['manual_labeled'])
    if len(unique_labels) > 1:  # More than just background
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label_id in enumerate(unique_labels):
            if label_id > 0:  # Skip background
                mask = area_analysis['manual_labeled'] == label_id
                manual_areas_vis[mask] = colors[i][:3]
    
    axes[2, 0].imshow(manual_areas_vis)
    axes[2, 0].set_title(f'Manual Enclosed Areas\nCount: {area_analysis["manual_count"]}', fontsize=14)
    axes[2, 0].axis('off')
    
    # Connected areas visualization
    connected_areas_vis = np.zeros((*connected_array.shape, 3))
    unique_labels = np.unique(area_analysis['connected_labeled'])
    if len(unique_labels) > 1:  # More than just background
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label_id in enumerate(unique_labels):
            if label_id > 0:  # Skip background
                mask = area_analysis['connected_labeled'] == label_id
                connected_areas_vis[mask] = colors[i][:3]
    
    axes[2, 1].imshow(connected_areas_vis)
    axes[2, 1].set_title(f'Connected Enclosed Areas\nCount: {area_analysis["connected_count"]}', fontsize=14)
    axes[2, 1].axis('off')
    
    # Area distribution comparison
    if area_analysis['manual_areas'] and area_analysis['connected_areas']:
        axes[2, 2].hist(area_analysis['manual_areas'], bins=15, alpha=0.7, 
                       label=f'Manual (n={len(area_analysis["manual_areas"])})', color='red')
        axes[2, 2].hist(area_analysis['connected_areas'], bins=15, alpha=0.7, 
                       label=f'Connected (n={len(area_analysis["connected_areas"])})', color='blue')
        axes[2, 2].set_xlabel('Area (pixels)', fontsize=12)
        axes[2, 2].set_ylabel('Frequency', fontsize=12)
        axes[2, 2].set_title('Area Distribution', fontsize=14)
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add mean lines
        if area_analysis['manual_areas']:
            manual_mean = np.mean(area_analysis['manual_areas'])
            axes[2, 2].axvline(manual_mean, color='red', linestyle='--', alpha=0.8)
        
        if area_analysis['connected_areas']:
            connected_mean = np.mean(area_analysis['connected_areas'])
            axes[2, 2].axvline(connected_mean, color='blue', linestyle='--', alpha=0.8)
    else:
        axes[2, 2].text(0.5, 0.5, 'No enclosed areas found', 
                       ha='center', va='center', transform=axes[2, 2].transAxes, fontsize=14)
        axes[2, 2].set_title('Area Distribution', fontsize=14)
        axes[2, 2].axis('off')
    
    # Area statistics
    if area_analysis['manual_areas'] or area_analysis['connected_areas']:
        area_text = f"""
        AREA STATISTICS:
        
        Manual Areas:
        - Count: {area_analysis['manual_count']}
        - Mean: {np.mean(area_analysis['manual_areas']):.0f} pixels
        - Total: {np.sum(area_analysis['manual_areas']):.0f} pixels
        
        Connected Areas:
        - Count: {area_analysis['connected_count']}
        - Mean: {np.mean(area_analysis['connected_areas']):.0f} pixels
        - Total: {np.sum(area_analysis['connected_areas']):.0f} pixels
        
        Difference:
        - Count: {area_analysis['connected_count'] - area_analysis['manual_count']:+d}
        """
        
        axes[2, 3].text(0.05, 0.5, area_text, transform=axes[2, 3].transAxes, 
                        fontsize=10, verticalalignment='center', fontfamily='monospace')
    else:
        axes[2, 3].text(0.5, 0.5, 'No area statistics available', 
                       ha='center', va='center', transform=axes[2, 3].transAxes, fontsize=14)
    
    axes[2, 3].set_title('Area Statistics', fontsize=14)
    axes[2, 3].axis('off')
    
    # Add legend for error visualization
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct (True Positive)'),
        Patch(facecolor='red', label='Overcorrection (False Positive)'),
        Patch(facecolor='blue', label='Undercorrection (False Negative)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    
    return fig, stats_connected, stats_ilastik, area_analysis

def create_error_comparison_figure(manual_path, ilastik_path, connected_path, pair_name):
    """Create a figure comparing error visualizations side by side."""
    
    error_vis_connected, error_vis_ilastik, stats_connected, stats_ilastik, manual_array, _, _ = create_error_visualization_mosaic(
        manual_path, ilastik_path, connected_path, pair_name
    )
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Error Analysis Comparison: {pair_name}', fontsize=16, fontweight='bold')
    
    # Manual trace reference
    axes[0].imshow(manual_array)
    axes[0].set_title('Manual Trace\n(Ground Truth)', fontsize=14)
    axes[0].axis('off')
    
    # Ilastik error visualization
    axes[1].imshow(error_vis_ilastik)
    axes[1].set_title(f'Ilastik vs Manual\n'
                     f'Over: {stats_ilastik["FP"]} ({stats_ilastik["fp_rate"]:.1f}%) | '
                     f'Under: {stats_ilastik["FN"]} ({stats_ilastik["fn_rate"]:.1f}%)', 
                     fontsize=12)
    axes[1].axis('off')
    
    # Connected error visualization
    axes[2].imshow(error_vis_connected)
    axes[2].set_title(f'Connected vs Manual\n'
                     f'Over: {stats_connected["FP"]} ({stats_connected["fp_rate"]:.1f}%) | '
                     f'Under: {stats_connected["FN"]} ({stats_connected["fn_rate"]:.1f}%)', 
                     fontsize=12)
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct (True Positive)'),
        Patch(facecolor='red', label='Overcorrection (False Positive)'),
        Patch(facecolor='blue', label='Undercorrection (False Negative)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    return fig, stats_connected, stats_ilastik

def main():
    """Process mosaic images and create publication-ready visualizations."""
    
    # Find all manual trace files
    manual_pattern = "manual_trace/manual-trace-*.png"
    manual_files = glob.glob(manual_pattern)
    
    if not manual_files:
        print("No manual trace files found! Checking pattern...")
        print(f"Looking for: {manual_pattern}")
        print("Available files in manual_trace/:")
        for f in glob.glob("manual_trace/*"):
            print(f"  {f}")
        return
    
    # Create output directories
    output_dirs = {
        'comprehensive': 'publication_figures/comprehensive',
        'error_comparison': 'publication_figures/error_comparison',
        'error_only': 'publication_figures/error_only'
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print("="*80)
    print("MOSAIC PUBLICATION FIGURE GENERATION")
    print("="*80)
    
    all_stats_connected = []
    all_stats_ilastik = []
    all_area_stats = []
    
    for manual_file in manual_files:
        try:
            # Extract identifier
            identifier = manual_file.replace("manual_trace/manual-trace-", "").replace(".png", "")
            
            # Find corresponding files
            ilastik_file = f"simple_segmentation/simple-segmentation-{identifier}.png"
            connected_file = f"connected_skeletons/{identifier}_connected.png"
            
            print(f"\nLooking for files for identifier '{identifier}':")
            print(f"  Manual: {manual_file} - {'✓' if os.path.exists(manual_file) else '✗'}")
            print(f"  Ilastik: {ilastik_file} - {'✓' if os.path.exists(ilastik_file) else '✗'}")
            print(f"  Connected: {connected_file} - {'✓' if os.path.exists(connected_file) else '✗'}")
            
            if os.path.exists(ilastik_file) and os.path.exists(connected_file):
                pair_name = f"Image_{identifier}"
                
                print(f"\n🔄 Processing {pair_name}...")
                
                # Create comprehensive figure
                fig_comprehensive, stats_connected, stats_ilastik, area_stats = create_comprehensive_figure(
                    manual_file, ilastik_file, connected_file, pair_name
                )
                plt.savefig(f"{output_dirs['comprehensive']}/{identifier}_comprehensive.png", 
                           dpi=300, bbox_inches='tight')
                plt.close(fig_comprehensive)
                
                # Create error comparison figure
                fig_comparison, _, _ = create_error_comparison_figure(
                    manual_file, ilastik_file, connected_file, pair_name
                )
                plt.savefig(f"{output_dirs['error_comparison']}/{identifier}_error_comparison.png", 
                           dpi=300, bbox_inches='tight')
                plt.close(fig_comparison)
                
                # Store statistics
                all_stats_connected.append({
                    'pair': pair_name,
                    'stats': stats_connected
                })
                
                all_stats_ilastik.append({
                    'pair': pair_name,
                    'stats': stats_ilastik
                })
                
                all_area_stats.append({
                    'pair': pair_name,
                    'area_stats': area_stats
                })
                
                print(f"✅ {pair_name} completed")
                print(f"   Ilastik - Over: {stats_ilastik['fp_rate']:.1f}%, Under: {stats_ilastik['fn_rate']:.1f}%")
                print(f"   Connected - Over: {stats_connected['fp_rate']:.1f}%, Under: {stats_connected['fn_rate']:.1f}%")
                print(f"   Areas - Manual: {area_stats['manual_count']}, Connected: {area_stats['connected_count']}")
                
            else:
                missing = []
                if not os.path.exists(ilastik_file):
                    missing.append("Ilastik")
                if not os.path.exists(connected_file):
                    missing.append("Connected")
                print(f"❌ Missing files for {manual_file}: {', '.join(missing)}")
                
        except Exception as e:
            print(f"❌ Error processing {manual_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comprehensive summary
    if all_stats_connected and all_stats_ilastik:
        print("\n" + "="*90)
        print("COMPREHENSIVE ERROR ANALYSIS SUMMARY")
        print("="*90)
        print(f"{'Image':15} | {'Method':10} | {'Correct':8} | {'Over%':7} | {'Under%':8} | {'Total Err%':11}")
        print("-"*75)
        
        for i, stat_conn in enumerate(all_stats_connected):
            stat_ila = all_stats_ilastik[i]
            
            # Ilastik stats
            s_ila = stat_ila['stats']
            total_err_ila = s_ila['fp_rate'] + s_ila['fn_rate']
            print(f"{stat_ila['pair']:15} | {'Ilastik':10} | {s_ila['TP']:8} | {s_ila['fp_rate']:6.1f} | {s_ila['fn_rate']:7.1f} | {total_err_ila:10.1f}")
            
            # Connected stats
            s_conn = stat_conn['stats']
            total_err_conn = s_conn['fp_rate'] + s_conn['fn_rate']
            print(f"{stat_conn['pair']:15} | {'Connected':10} | {s_conn['TP']:8} | {s_conn['fp_rate']:6.1f} | {s_conn['fn_rate']:7.1f} | {total_err_conn:10.1f}")
            print()
        
        # Calculate averages
        total_tp_ila = sum(s['stats']['TP'] for s in all_stats_ilastik)
        total_fp_ila = sum(s['stats']['FP'] for s in all_stats_ilastik)
        total_fn_ila = sum(s['stats']['FN'] for s in all_stats_ilastik)
        total_manual_ila = sum(s['stats']['manual_pixels'] for s in all_stats_ilastik)
        
        total_tp_conn = sum(s['stats']['TP'] for s in all_stats_connected)
        total_fp_conn = sum(s['stats']['FP'] for s in all_stats_connected)
        total_fn_conn = sum(s['stats']['FN'] for s in all_stats_connected)
        total_manual_conn = sum(s['stats']['manual_pixels'] for s in all_stats_connected)
        
        avg_fp_rate_ila = (total_fp_ila / total_manual_ila * 100) if total_manual_ila > 0 else 0
        avg_fn_rate_ila = (total_fn_ila / total_manual_ila * 100) if total_manual_ila > 0 else 0
        avg_total_err_ila = avg_fp_rate_ila + avg_fn_rate_ila
        
        avg_fp_rate_conn = (total_fp_conn / total_manual_conn * 100) if total_manual_conn > 0 else 0
        avg_fn_rate_conn = (total_fn_conn / total_manual_conn * 100) if total_manual_conn > 0 else 0
        avg_total_err_conn = avg_fp_rate_conn + avg_fn_rate_conn
        
        print("-"*75)
        print(f"{'AVERAGE':15} | {'Ilastik':10} | {total_tp_ila:8} | {avg_fp_rate_ila:6.1f} | {avg_fn_rate_ila:7.1f} | {avg_total_err_ila:10.1f}")
        print(f"{'AVERAGE':15} | {'Connected':10} | {total_tp_conn:8} | {avg_fp_rate_conn:6.1f} | {avg_fn_rate_conn:7.1f} | {avg_total_err_conn:10.1f}")
        
        print(f"\nIMPROVEMENT (Connected vs Ilastik):")
        print(f"  Overcorrection: {avg_fp_rate_conn - avg_fp_rate_ila:+.1f}%")
        print(f"  Undercorrection: {avg_fn_rate_conn - avg_fn_rate_ila:+.1f}%")
        print(f"  Total Error: {avg_total_err_conn - avg_total_err_ila:+.1f}%")
    
    if all_area_stats:
        print("\n" + "="*70)
        print("AREA ANALYSIS SUMMARY")
        print("="*70)
        print(f"{'Image':15} | {'Manual Areas':12} | {'Connected Areas':15} | {'Difference':10}")
        print("-"*60)
        
        total_manual_areas = 0
        total_connected_areas = 0
        
        for stat in all_area_stats:
            a = stat['area_stats']
            diff = a['connected_count'] - a['manual_count']
            print(f"{stat['pair']:15} | {a['manual_count']:12} | {a['connected_count']:15} | {diff:+9}")
            total_manual_areas += a['manual_count']
            total_connected_areas += a['connected_count']
        
        print("-"*60)
        total_diff = total_connected_areas - total_manual_areas
        print(f"{'TOTAL':15} | {total_manual_areas:12} | {total_connected_areas:15} | {total_diff:+9}")
    
    print(f"\nPublication figures saved to 'publication_figures/' directory:")
    print("- Comprehensive analysis: publication_figures/comprehensive/")
    print("- Error comparisons: publication_figures/error_comparison/")

if __name__ == "__main__":
    main()