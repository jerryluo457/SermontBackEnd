import json
import matplotlib.pyplot as plt

def view_simulation_output(file_path='simulation_output.json'):
    # 1. Load the JSON payload
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Make sure to run your simulation engine first.")
        return

    # 2. Extract the data layers
    ages = data['time_axis']
    summary = data['summary']
    sample_paths = data['sample_paths']
    success_rate = data['metrics']['probability_of_success']

    plt.figure(figsize=(12, 6))

    # 3. Plot the sample paths (background noise)
    for path in sample_paths:
        plt.plot(ages, path, color='blue', alpha=0.15)
        
    # 4. Plot the probability cone (Percentiles)
    # plt.plot(ages, summary['p99'], color='green', linestyle=':', linewidth=1.5, label='99th Percentile')
    plt.plot(ages, summary['p75'], color='blue', linestyle='--', linewidth=2, label='75th Percentile (Optimistic)')
    plt.plot(ages, summary['p50'], color='black', linewidth=3, label='50th Percentile (Median)')
    plt.plot(ages, summary['p25'], color='orange', linestyle='--', linewidth=2, label='25th Percentile (Pessimistic)')
    # plt.plot(ages, summary['p1'], color='red', linestyle=':', linewidth=1.5, label='1st Percentile')
    
    # 5. Formatting the chart
    plt.title(f'Monte Carlo Engine Output (Success Rate: {success_rate:.1f}%)')
    plt.xlabel('Age')
    plt.ylabel('Real Wealth ($)')
    
    # Add commas to the Y-axis for readability (e.g., 1,000,000)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.axhline(0, color='black', linewidth=1)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    
    plt.show()

if __name__ == "__main__":
    view_simulation_output()