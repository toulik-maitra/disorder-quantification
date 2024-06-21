import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean, cosine
from frechetdist import frdist
import pingouin as pg

from scipy.stats import gaussian_kde, norm
from scipy.optimize import curve_fit
from scipy.stats import chisquare

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))



def load_data(file_path):
    data = pd.read_csv(file_path)
    energy = data.iloc[:, 0].values
    ratio = data.iloc[:, 1].values
    
    # Retrieve volume and number of hydrogen atoms from the first row
    volume = data.iloc[0, 2]
    print(volume)
    #num_hydrogen = data.iloc[0, 3]

    # Normalize the ratio by unit cell volume and number of hydrogen atoms
    normalized_ratio = ratio / (volume)#ratio / (volume * num_hydrogen)


    # Filter data to include only energy values up to 3500
    mask = energy <= 3500
    energy = energy[mask]
    normalized_ratio = ratio[mask]

    return energy, normalized_ratio.flatten()  # Ensure ratio is 1-dimensional

def align_energy(file_paths):
    # Determine the common energy range and create a grid
    min_energy = float('inf')
    max_energy = float('-inf')

    # Load data and find the min and max energy
    datasets = []
    for file_path in file_paths:
        energy, ratio = load_data(file_path)
        datasets.append((energy, ratio))
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())

    # Create a common energy grid
    common_energy = np.linspace(min_energy, max_energy, 500)
    
    # Interpolate ratios onto the common energy grid
    interpolated_ratios = []
    for energy, ratio in datasets:
        interpolated_ratio = np.interp(common_energy, energy, ratio)
        interpolated_ratios.append(interpolated_ratio)

    return common_energy, np.array(interpolated_ratios)

def plot_vertical_lines(common_energy, interpolated_ratios, file_paths):
    plt.figure(figsize=(12, 8))
    
    num_samples = interpolated_ratios.shape[0]
    colors = sns.color_palette("husl", num_samples)

    for i, (energy, ratios) in enumerate(zip(common_energy, interpolated_ratios.T)):
        for j, ratio in enumerate(ratios):
            plt.plot([energy, energy], [0, ratio], color=colors[j], alpha=0.6)

    for i, file_path in enumerate(file_paths):
        label = os.path.basename(file_path)
        plt.plot([], [], color=colors[i], label=label)  # Empty plot for legend

    plt.xlabel('Energy (cm-1)')
    plt.ylabel('Peak to Baseline Ratio')
    plt.title('Vertical Lines Plot: Peak to Baseline Ratios')
    plt.legend()
    plt.show()

def plot_line(file_paths):
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        energy, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        plt.plot(energy, ratio, label=label)

    plt.xlabel('Energy (cm-1)')
    plt.ylabel('Peak to Baseline Ratio')
    plt.title('Line Plot: Peak to Baseline Ratios')
    plt.legend()
    plt.show()

def plot_scatter(file_paths):
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        energy, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        plt.scatter(energy, ratio, label=label)

    plt.xlabel('Energy (cm-1)')
    plt.ylabel('Peak to Baseline Ratio')
    plt.title('Scatter Plot: Peak to Baseline Ratios')
    plt.legend()
    plt.show()

def plot_histogram(file_paths):
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        _, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        plt.hist(ratio, bins=30, alpha=0.5, label=label)

    plt.xlabel('Peak to Baseline Ratio')
    plt.ylabel('Frequency')
    plt.title('Histogram: Distribution of Ratios')
    plt.legend()
    plt.show()

def plot_box(file_paths):
    data = []

    for file_path in file_paths:
        _, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        data.append(pd.Series(ratio, name=label))

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xlabel('Files')
    plt.ylabel('Peak to Baseline Ratio')
    plt.title('Box Plot: Statistical Summary of Ratios')
    plt.show()
    


def plot_violin(file_paths):
    data = []
    violin_stats = {}

    for file_path in file_paths:
        _, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        series = pd.Series(ratio, name=label)
        data.append(series)

        # Calculate and print the width, mean, height, and asymmetry factor for each violin plot
        mean_val = np.mean(series)
        height_val = np.max(series) - np.min(series)
        skewness_val = skew(series)

        violin_stats[label] = {
            'mean': mean_val,
            'height': height_val,
            'skewness': skewness_val
        }

        print(f"Dataset: {series.name}")
        print(f"Mean: {mean_val}")
        print(f"Height: {height_val}")
        print(f"Asymmetry (Skewness): {skewness_val}\n")

    plt.figure(figsize=(10, 6))
    ax_violin = sns.violinplot(data=data)

    # Add lines connecting median and 95th percentile across groups
    for i in range(len(data) - 1):
        # Median connection
        median1 = np.median(data[i])
        median2 = np.median(data[i + 1])
        ax_violin.plot([i, i + 1], [median1, median2], color='k', lw=1.5, ls='-', marker='o', markersize=8, markerfacecolor='w')

        # 95th percentile connection
        percentile_951 = np.percentile(data[i], 95)
        percentile_952 = np.percentile(data[i + 1], 95)
        ax_violin.plot([i, i + 1], [percentile_951, percentile_952], color='b', lw=1.5, ls='--', marker='s', markersize=8, markerfacecolor='w')

    plt.xlabel('Files')
    plt.ylabel('Peak to Baseline Ratio')
    plt.title('Violin Plot: Distribution and Density of Ratios')
    plt.show()

    return violin_stats

def plot_ridgeline(file_paths):
  """
  Plots a ridgeline graph and a scatter plot to visualize the distribution of ratios in multiple files.

  Args:
      file_paths: A list of file paths containing ratio data.

  Returns:
      A dictionary containing quantile information for each file.
  """

  data = []
  quantiles_info = {}

  plt.figure(figsize=(12, 8))

  for file_path in file_paths:
      _, ratio = load_data(file_path) # Assuming load_data is defined elsewhere to load data
      label = os.path.basename(file_path)
      series = pd.Series(ratio, name=label)
      data.append(series)

      # Calculate and store quantiles and heights
      quantiles = np.percentile(series, [5, 50, 95])
      quantiles_info[label] = {'quantiles': quantiles, 'heights': gaussian_kde(series)(quantiles) * 2}

      # Print quantile information for debugging (optional)
      print(f"{label}:")
      print(f"  5% quantile: {quantiles[0]:.2f}, width: {quantiles_info[label]['heights'][0]:.5f}")
      print(f"  50% quantile (median): {quantiles[1]:.2f}, width: {quantiles_info[label]['heights'][1]:.5f}")
      print(f"  95% quantile: {quantiles[2]:.2f}, width: {quantiles_info[label]['heights'][2]:.5f}")

  # Ridgeline plot
  plt.subplot(211)
  for i, series in enumerate(data):
      sns.kdeplot(series.to_numpy(), fill=True, alpha=0.5, label=series.name)

      # Annotate quantiles on the plot
      quantiles = quantiles_info[series.name]['quantiles']
      heights = quantiles_info[series.name]['heights'] / 2
      for q, h in zip(quantiles, heights):
          plt.plot([q], [h], marker='o', markersize=10, label=f'{series.name} Quantile {q:.2f}')

  plt.xlabel('Peak to Baseline Ratio')
  plt.ylabel('Density')
  plt.title('Ridgeline Plot: Distribution of Ratios')

  # Quantile scatter plot
  plt.subplot(212)
  colors = sns.color_palette("husl", len(file_paths))  # Get a list of colors

  for i, file_path in enumerate(file_paths):
      label = os.path.basename(file_path)
      quantiles = quantiles_info[label]['quantiles']
      heights = quantiles_info[label]['heights']
      plt.scatter([5, 50, 95], heights, color=colors[i], label=label)
      plt.plot([5, 50, 95], heights, color=colors[i], linestyle='-', marker='o')

  plt.xlabel('Percentile')
  plt.ylabel('Widths')
  plt.title('Widths at 5%, 50%, and 95% Quantiles')
  plt.legend()
  plt.tight_layout()  # Adjust spacing between subplots
  plt.show()

  return quantiles_info



def plot_3d_structure(file_paths):
    violin_stats = plot_violin(file_paths)
    quantiles_info = plot_ridgeline(file_paths)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette("viridis", len(file_paths))  # Get a list of colors
    offset_mean=range(0, len(file_paths) * 5, 5)
    
    for i, file_path in enumerate(file_paths):
        label = os.path.basename(file_path)
        v_stats = violin_stats[label]
        q_info = quantiles_info[label]

        mean = v_stats['mean']
        skewness = v_stats['skewness']
        heights = v_stats['height']
        widths = q_info['heights']

        # Create ellipsoid points
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x =  offset_mean[i] + widths[1] * np.outer(np.cos(u), np.sin(v)) #offset_mean[i] + 
        y = skewness* np.outer(np.sin(u), np.sin(v))#skewness + widths[1] * np.outer(np.sin(u), np.sin(v))
        z = (heights) * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot ellipsoid surface
        ax.plot_surface(x, y, z, color=colors[i], alpha=0.5)

        # Draw lines over the ellipsoid
        ax.plot(x[:, 0], y[:, 0], z[:, 0], color=colors[i], alpha=0.6)
        ax.plot(x[:, -1], y[:, -1], z[:, -1], color=colors[i], alpha=0.6)
        ax.plot(x[0, :], y[0, :], z[0, :], color=colors[i], alpha=0.6)
        ax.plot(x[-1, :], y[-1, :], z[-1, :], color=colors[i], alpha=0.6)

        # Print the structure details
        print(f"{label}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Skewness: {skewness:.2f}")
        print(f"  Height: {heights:.2f}")
        print(f"  Widths (5%, 50%, 95%): {widths[0]:.2f}, {widths[1]:.2f}, {widths[2]:.2f}\n")

    ax.set_xlabel('Widths at 50%')
    ax.set_ylabel('Skewness')
    ax.set_zlabel('Height')
    ax.set_title('3D Structure of Samples')
    
    # Create custom legend
    handles = [plt.Line2D([0, 1], [0, 0], color=colors[i], linestyle='-') for i in range(len(file_paths))]
    plt.legend(handles, [os.path.basename(fp) for fp in file_paths], loc='best')

    plt.show()
    
def plot_ratio(file_paths):
    violin_stats = plot_violin(file_paths)
    quantiles_info = plot_ridgeline(file_paths)
    
    ratios = []
    labels = []

    for file_path in file_paths:
        label = os.path.basename(file_path)
        v_stats = violin_stats[label]
        q_info = quantiles_info[label]

        heights = v_stats['height']
        widths = q_info['heights']

        # Calculate the ratio
        ratio = widths[1] / heights

        # Store the results
        ratios.append(ratio)
        labels.append(label)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, ratios, color=sns.color_palette("viridis", len(file_paths)))

    ax.set_xlabel('Samples')
    ax.set_ylabel('Ratio of Width[1] to Height')
    ax.set_title('Ratio of Width[1] to Height for Different Samples')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
def plot_waterfall(file_paths):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for idx, file_path in enumerate(file_paths):
        energy, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        ax.plot(energy, ratio, zs=idx, zdir='y', label=label, alpha=0.7)

    ax.set_xlabel('Energy (cm-1)')
    ax.set_ylabel('File Index')
    ax.set_zlabel('Peak to Baseline Ratio')
    ax.set_title('Waterfall Plot: Peak to Baseline Ratios')
    ax.legend()
    plt.show()
    
def plot_contour(file_paths):
    # Determine the common energy range and create a grid
    min_energy = float('inf')
    max_energy = float('-inf')

    # Load data and find the min and max energy
    datasets = []
    for file_path in file_paths:
        energy, ratio = load_data(file_path)
        datasets.append((energy, ratio))
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())

    # Create a common energy grid
    common_energy = np.linspace(min_energy, max_energy, 500)
    
    # Interpolate ratios onto the common energy grid
    interpolated_ratios = []
    for energy, ratio in datasets:
        interpolated_ratio = np.interp(common_energy, energy, ratio)
        interpolated_ratios.append(interpolated_ratio)

    # Convert to numpy array for plotting
    interpolated_ratios = np.array(interpolated_ratios)

    # Create contour plot
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(common_energy, np.arange(len(file_paths)))
    Z = interpolated_ratios

    cp = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(cp)
    plt.xlabel('Energy (cm-1)')
    plt.ylabel('Sample Index')
    plt.title('Contour Plot: Peak to Baseline Ratios')
    plt.show()


def calculate_statistics(file_paths):
    stats = {}

    for file_path in file_paths:
        _, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        stats[label] = {
            'mean': np.mean(ratio),
            'median': np.median(ratio),
            'std_dev': np.std(ratio)
        }

    stats_df = pd.DataFrame(stats).T
    print("Statistical Summary:")
    print(stats_df)
    return stats_df

def calculate_frechet_distances(datasets):
    num_datasets = len(datasets)
    distances = np.zeros((num_datasets, num_datasets))

    for i in range(num_datasets):
        for j in range(i + 1, num_datasets):
            distance = frdist(datasets[i][1], datasets[j][1])
            distances[i, j] = distance
            distances[j, i] = distance  # distances matrix is symmetric
    
    return distances

    
    
def plot_correlation(file_paths):
    ratios = []

    # Load and process data
    for file_path in file_paths:
        _, ratio = load_data(file_path)
        ratios.append(ratio)

    # Check shapes of ratios
    for idx, r in enumerate(ratios):
        print(f"Ratio {idx}: Shape={r.shape}")

    # Ensure all ratios have the same length
    min_length = min(len(ratio) for ratio in ratios)
    ratios = [ratio[:min_length] for ratio in ratios]

    # Convert ratios to DataFrame for correlation calculation
    ratios_df = pd.DataFrame(ratios).transpose()
    ratios_df.columns = [os.path.basename(f) for f in file_paths]

    def plot_correlation_matrix(correlation_matrix, method):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', xticklabels=ratios_df.columns, yticklabels=ratios_df.columns)
        plt.title(f'Correlation Matrix of Ratios ({method.capitalize()})')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # Pearson, Spearman, Kendall correlations
    for method in ['pearson', 'spearman', 'kendall']:
        correlation_matrix = ratios_df.corr(method=method)
        plot_correlation_matrix(correlation_matrix, method)

    # Covariance
    covariance_matrix = ratios_df.cov()
    plot_correlation_matrix(covariance_matrix, 'covariance')

    # Partial correlation
    partial_corr_matrix = np.zeros((len(ratios_df.columns), len(ratios_df.columns)))

    for i, col_i in enumerate(ratios_df.columns):
        for j, col_j in enumerate(ratios_df.columns):
            if i == j:
                partial_corr_matrix[i, j] = 1.0
            else:
                pcorr = pg.partial_corr(data=ratios_df, x=col_i, y=col_j, covar=[c for k, c in enumerate(ratios_df.columns) if k != i and k != j])
                partial_corr_matrix[i, j] = pcorr['r'].values[0]  # Adjusted to fetch the correct value

    partial_corr_df = pd.DataFrame(partial_corr_matrix, index=ratios_df.columns, columns=ratios_df.columns)
    plot_correlation_matrix(partial_corr_df, 'partial correlation')


def calc_kurtosis_skewness(file_paths):
    results = []

    for file_path in file_paths:
        _, ratio = load_data(file_path)
        label = os.path.basename(file_path)
        
        # Calculate kurtosis and skewness using scipy.stats
        kurtosis_value = kurtosis(ratio)
        skewness_value = skew(ratio)
        
        # Store the results
        results.append({
            'File': label,
            'Kurtosis': kurtosis_value,
            'Skewness': skewness_value
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

def calc_similarity(file_paths):
    if len(file_paths) < 2:
        print("Please provide at least two files to calculate similarity.")
        return

    base_file_path = file_paths[0]
    _, base_ratio = load_data(base_file_path)
    base_label = os.path.basename(base_file_path)

    similarity_results = []

    for file_path in file_paths[1:]:
        _, ratio = load_data(file_path)
        label = os.path.basename(file_path)

        # Normalize the ratios by length
        min_len = min(len(base_ratio), len(ratio))
        base_ratio_normalized = base_ratio[:min_len]
        ratio_normalized = ratio[:min_len]

        # Calculate Euclidean distance
        euclidean_distance = euclidean(base_ratio_normalized, ratio_normalized)

        # Calculate Cosine similarity
        cosine_similarity = 1 - cosine(base_ratio_normalized, ratio_normalized)

        # Prepare data for Frechet distance
        base_ratio_points = [[value] for value in base_ratio_normalized]
        ratio_points = [[value] for value in ratio_normalized]

        # Calculate Frechet distance
        frechet_distance = frdist(base_ratio_points, ratio_points)

        similarity_info = {
            'Comparison File': label,
            'Euclidean Distance': euclidean_distance,
            'Cosine Similarity': cosine_similarity,
            'Frechet Distance': frechet_distance
        }

        similarity_results.append(similarity_info)

    return similarity_results

def plot_similarity(similarity_info, base_file):
    labels = [info['Comparison File'] for info in similarity_info]
    euclidean_distances = [info['Euclidean Distance'] for info in similarity_info]
    cosine_similarities = [info['Cosine Similarity'] for info in similarity_info]
    frechet_distances = [info['Frechet Distance'] for info in similarity_info]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting Euclidean distance and Frechet distance on the primary y-axis
    ax1.set_xlabel('Comparison Files')
    ax1.set_ylabel('Distance Metrics')
    bar1 = ax1.bar(x - width, euclidean_distances, width, label='Euclidean Distance')
    bar2 = ax1.bar(x, frechet_distances, width, label='Frechet Distance')
    ax1.tick_params(axis='y')

    # Creating a secondary y-axis for Cosine similarity
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cosine Similarity')
    bar3 = ax2.bar(x + width, cosine_similarities, width, label='Cosine Similarity', color='green')
    ax2.tick_params(axis='y')

    # Adding labels and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_title('Similarity Metrics Comparison')

    # Combine the legends from both y-axes
    bars = bar1 + bar2 + bar3
    labels = [bar.get_label() for bar in bars]
    ax1.legend(bars, labels, loc='upper left')

    # Add annotation to mention the base file
    plt.annotate(f'Comparing to: {base_file}', xy=(0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=12, color='blue')

    plt.tight_layout()
    plt.show()

def align_and_plot_polar(file_paths):
    # Align energy values and interpolate ratios
    common_energy = np.linspace(0, 3500, 100)  # Example common energy values
    interpolated_ratios = []
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        energy = data['Energy']
        ratio = data['Peak/Baseline Ratio']
        interp_ratio = np.interp(common_energy, energy, ratio)
        interpolated_ratios.append(interp_ratio)
    
    # Create a square root transformed theta for polar plot
    transformed_energy = np.sqrt(common_energy)
    max_transformed_energy = np.max(transformed_energy)
    theta = (transformed_energy / max_transformed_energy) * 2 * np.pi

    def plot_polar(data, title, ylabel):
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, projection='polar')
        for label, values in data.items():
            # Ensure values are appropriately shaped to match theta
            if len(values) == 1:
                values = np.repeat(values, len(theta))  # Repeat single value
            ax.plot(theta, values, label=label)
        ax.set_title(title)
        ax.set_rmax(max([max(values) for values in data.values()]) * 1.1)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()

    # Plot polar plot for interpolated ratios
    plot_polar({os.path.basename(file_paths[i]): ratios for i, ratios in enumerate(interpolated_ratios)}, 
               'Polar Plot of Energy vs Peak/Intensity Ratio', 'Peak/Intensity Ratio')

    # Calculate and plot violin plot statistics
    violin_stats = plot_violin(file_paths)
    plot_polar({label: [stats['mean']] * len(theta) for label, stats in violin_stats.items()}, 
               'Polar Plot of Violin Means', 'Mean')
    plot_polar({label: [stats['height']] * len(theta) for label, stats in violin_stats.items()}, 
               'Polar Plot of Violin Heights', 'Height')
    plot_polar({label: [stats['skewness']] * len(theta) for label, stats in violin_stats.items()}, 
               'Polar Plot of Violin Skewness', 'Skewness')

    # Calculate and plot ridgeline plot statistics
    ridgeline_stats = plot_ridgeline(file_paths)
    for q in [5, 50, 95]:
        quantiles = {label: info['quantiles'][int(q/50)] * np.ones_like(theta) for label, info in ridgeline_stats.items()}
        plot_polar(quantiles, f'Polar Plot of {q}th Quantiles', f'{q}th Quantile')
        
    for q in [5, 50, 95]:
        heights = {label: info['heights'][int(q/50)] * np.ones_like(theta) for label, info in ridgeline_stats.items()}
        plot_polar(heights, f'Polar Plot of {q}th Quantile Heights', f'{q}th Quantile Height')


def main():
    num_directories = int(input("Enter the number of directories to compare: "))
    directories = []
    for i in range(num_directories):
        dir_path = input(f"Enter the path for directory {i+1}: ")
        directories.append(dir_path)

    file_paths = []
    for dir_path in directories:
        print(f"Files in directory '{dir_path}':")
        files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
        for idx, file_name in enumerate(files):
            print(f"{idx}: {file_name}")

        file_indices = input(f"Enter the indices of the files you want to compare from '{dir_path}' (comma separated): ")
        indices = list(map(int, file_indices.split(',')))
        selected_files = [os.path.join(dir_path, files[idx]) for idx in indices]
        file_paths.extend(selected_files)

    base_file = file_paths[0]
    
    common_energy, interpolated_ratios = align_energy(file_paths)
    align_and_plot_polar(file_paths)
    plot_vertical_lines(common_energy, interpolated_ratios, file_paths)
    plot_line(file_paths)
    plot_scatter(file_paths)
    plot_histogram(file_paths)
    plot_box(file_paths)
    plot_ratio(file_paths)
    #plot_violin(file_paths)
    #plot_ridgeline(file_paths)
    plot_3d_structure(file_paths)
    #plot_waterfall(file_paths)
    #plot_contour(file_paths)
    stats_df = calculate_statistics(file_paths)
    plot_correlation(file_paths)
    #kurtosis_skewness_df = calc_kurtosis_skewness(file_paths)
    similarity_info = calc_similarity(file_paths)
    plot_similarity(similarity_info,base_file)
    


if __name__ == '__main__':
    main()
