import matplotlib.pyplot as plt
import re

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def extract_times(file_content, operation):
    if operation == 'backbone':
        ref_pattern = r"Create backbone \(reference-based tree\):\nExecution Time: ([\d.]+) seconds"
        array_pattern = r"Create backbone \(array-based tree\):\nExecution Time: ([\d.]+) seconds"
    else:  # DSW
        ref_pattern = r"DSW on reference-based tree:\nExecution Time: ([\d.]+) seconds"
        array_pattern = r"DSW on array-based tree:\nExecution Time: ([\d.]+) seconds"
    
    ref_time = float(re.search(ref_pattern, file_content).group(1))
    array_time = float(re.search(array_pattern, file_content).group(1))
    
    return ref_time, array_time

def create_performance_plots(file_paths):
    # Initialize lists to store times
    ref_backbone_times = []
    array_backbone_times = []
    ref_dsw_times = []
    array_dsw_times = []
    
    # Extract times from each file
    for path in file_paths:
        content = read_file(path)
        ref_time, array_time = extract_times(content, 'backbone')
        ref_backbone_times.append(ref_time)
        array_backbone_times.append(array_time)
        
        ref_time, array_time = extract_times(content, 'dsw')
        ref_dsw_times.append(ref_time)
        array_dsw_times.append(array_time)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot backbone comparison
    x = list(range(1, len(file_paths) + 1))
    ax1.plot(x, ref_backbone_times, 'b-o', label='Reference-based')
    ax1.plot(x, array_backbone_times, 'r-o', label='Array-based')
    ax1.set_title('Backbone Creation Performance')
    ax1.set_xlabel('Test Run')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot DSW comparison
    ax2.plot(x, ref_dsw_times, 'b-o', label='Reference-based')
    ax2.plot(x, array_dsw_times, 'r-o', label='Array-based')
    ax2.set_title('DSW Algorithm Performance')
    ax2.set_xlabel('Test Run')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()

file_paths = [
    'results_data_10\step_analysis_results.txt',
    'results_data_15\step_analysis_results.txt',
    'results_data_20\step_analysis_results.txt',
    'results_data_25\step_analysis_results.txt'
]

create_performance_plots(file_paths)