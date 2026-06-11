import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.interpolate import interp1d


def read_surround_map(filepath):
    """Read a TIFF file and extract the center-surround map.
    
    Parameters
    ----------
    filepath : str
        Path to the TIFF file
        
    Returns
    -------
    map_2d : ndarray
        31x31 center-surround map
    """
    img = tifffile.imread(filepath)
    
    # If the image is not already 31x31, extract the relevant portion
    if img.shape != (31, 31):
        # Assume we need to get a 31x31 center region
        h, w = img.shape[-2:]
        start_h = (h - 31) // 2
        start_w = (w - 31) // 2
        map_2d = img[start_h:start_h+31, start_w:start_w+31]
    else:
        map_2d = img
    
    return map_2d.astype(float)


def radial_average_linescans(map_2d, num_angles=16):
    """Perform radial averaging by line scanning through center at various angles.
    
    Parameters
    ----------
    map_2d : ndarray
        31x31 center-surround map
    num_angles : int
        Number of angles to sample (default 16)
        
    Returns
    -------
    distances : ndarray
        Distance from center
    avg_values : ndarray
        Radially averaged values
    all_linescan_values : ndarray
        All values along each linescan for analysis
    """
    h, w = map_2d.shape
    center_y, center_x = h / 2 - 0.5, w / 2 - 0.5  # 31x31 center at 15, 15
    
    # Maximum distance from center to edge
    max_dist = np.sqrt((center_x)**2 + (center_y)**2)
    # Distance from -max_dist to +max_dist (center at 0)
    distances = np.arange(-max_dist, max_dist + 0.5, 0.5)
    
    # Store all values at each distance across all angles
    radial_values = {dist: [] for dist in distances}
    
    # Scan at different angles
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    all_linescan_values = []
    
    for angle in angles:
        # Direction vector
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Sample along this direction
        linescan = []
        for dist in distances:
            y = center_y + dist * dy
            x = center_x + dist * dx
            
            # Bilinear interpolation
            if 0 <= y < h-1 and 0 <= x < w-1:
                y_low = int(y)
                x_low = int(x)
                wy = y - y_low
                wx = x - x_low
                
                val = (map_2d[y_low, x_low] * (1-wx) * (1-wy) +
                       map_2d[y_low, x_low+1] * wx * (1-wy) +
                       map_2d[y_low+1, x_low] * (1-wx) * wy +
                       map_2d[y_low+1, x_low+1] * wx * wy)
                linescan.append(val)
                radial_values[dist].append(val)
            else:
                linescan.append(0)
        
        all_linescan_values.append(linescan)
    
    # Average values at each distance across all angles
    avg_values = np.array([np.nanmean(radial_values[d]) for d in distances])
    
    return distances, avg_values, np.array(all_linescan_values)


def create_symmetrical_map(distances, avg_values, map_shape=(31, 31)):
    """Reconstruct a symmetrical 2D map from radial average values.
    
    Parameters
    ----------
    distances : ndarray
        Distance from center
    avg_values : ndarray
        Radially averaged values
    map_shape : tuple
        Shape of output map (default 31x31)
        
    Returns
    -------
    symmetrical_map : ndarray
        Reconstructed 2D map with radial symmetry
    """
    h, w = map_shape
    center_y, center_x = h / 2 - 0.5, w / 2 - 0.5
    
    symmetrical_map = np.zeros(map_shape)
    
    # Create interpolation function for radial values
    f_radial = interp1d(distances, avg_values, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
    
    # Fill each pixel based on its distance from center
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            symmetrical_map[y, x] = f_radial(dist)
    
    # Fill any remaining NaN values with 0
    symmetrical_map = np.nan_to_num(symmetrical_map, nan=0.0)
    
    return symmetrical_map


def save_symmetrical_map(symmetrical_map, output_filepath='symmetrical_map.tif'):
    """Save the symmetrical map to a TIFF file.
    
    Parameters
    ----------
    symmetrical_map : ndarray
        The symmetrical 2D map
    output_filepath : str
        Path to save the TIFF file
    """
    tifffile.imwrite(output_filepath, symmetrical_map.astype(np.float32))
    print(f"Saved symmetrical map to {output_filepath}")


def plot_results(map_2d, distances, avg_values, all_linescan_values, num_angles=16):
    """Plot the center-surround map with scan lines overlaid and individual linescan values.
    
    Parameters
    ----------
    map_2d : ndarray
        Original 31x31 map
    distances : ndarray
        Distance from center
    avg_values : ndarray
        Radially averaged values
    all_linescan_values : ndarray
        All linescan values
    num_angles : int
        Number of angles used in scanning
    """
    h, w = map_2d.shape
    center_y, center_x = h / 2 - 0.5, w / 2 - 0.5
    max_dist = np.sqrt((center_x)**2 + (center_y)**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original map with scan lines overlaid
    im0 = axes[0].imshow(map_2d, cmap='viridis')
    axes[0].set_title('Center-Surround Map with Scan Lines')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot scan lines with color coding
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    cmap = plt.cm.hsv
    for i, angle in enumerate(angles):
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Color based on angle
        color = cmap(i / num_angles)
        
        # Plot line from center through the map (both directions)
        x_start = center_x - max_dist * dx
        y_start = center_y - max_dist * dy
        x_end = center_x + max_dist * dx
        y_end = center_y + max_dist * dy
        axes[0].plot([x_start, x_end], [y_start, y_end], color=color, alpha=0.6, linewidth=1.5)
    
    # Mark center
    axes[0].plot(center_x, center_y, 'k+', markersize=10, markeredgewidth=2)
    
    # Individual linescans with color coding
    cmap = plt.cm.hsv
    for i, linescan in enumerate(all_linescan_values):
        color = cmap(i / len(all_linescan_values))
        axes[1].plot(distances, linescan, alpha=0.7, linewidth=1.5, color=color)
    
    axes[1].set_title('Individual Line Scans')
    axes[1].set_xlabel('Distance from Center (negative = opposite direction)')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main(filepath, num_angles=36, show_plot=True, output_filepath='symmetrical_map.tif'):
    """Main function to process center-surround map.
    
    Parameters
    ----------
    filepath : str
        Path to TIFF file
    num_angles : int
        Number of angles for line scans
    show_plot : bool
        Whether to display plots
    output_filepath : str
        Path to save the symmetrical map TIFF file
    """
    print(f"Reading {filepath}...")
    map_2d = read_surround_map(filepath)
    print(f"Map shape: {map_2d.shape}")
    print(f"Map value range: [{map_2d.min():.3f}, {map_2d.max():.3f}]")
    
    print(f"\nPerforming radial averaging with {num_angles} angles...")
    distances, avg_values, all_linescan_values = radial_average_linescans(
        map_2d, num_angles=num_angles
    )
    
    print(f"Distance range: [0, {distances.max():.2f}]")
    print(f"Average value range: [{np.nanmin(avg_values):.3f}, {np.nanmax(avg_values):.3f}]")
    
    # Create and save symmetrical map
    print(f"\nCreating symmetrical map...")
    symmetrical_map = create_symmetrical_map(distances, avg_values, map_shape=map_2d.shape)
    save_symmetrical_map(symmetrical_map, output_filepath)
    
    if show_plot:
        fig = plot_results(map_2d, distances, avg_values, all_linescan_values, num_angles=num_angles)
        plt.show()
    
    return map_2d, distances, avg_values, all_linescan_values, symmetrical_map


if __name__ == "__main__":
    # Example usage - modify the filepath as needed
    import sys
    import os
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "center_surround_map.tif"
    
    # Set output path to same directory as input
    input_dir = os.path.dirname(filepath)
    output_filepath = os.path.join(input_dir, 'symmetrical_map.tif')
    
    map_2d, distances, avg_values, all_linescan_values, symmetrical_map = main(
        filepath, 
        num_angles=36, 
        show_plot=True,
        output_filepath=output_filepath
    )









