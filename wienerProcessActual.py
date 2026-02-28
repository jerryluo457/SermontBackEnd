import numpy as np
import matplotlib.pyplot as plt
import json
from input_loader import load_simulation_inputs

# https://tinyurl.com/2kcedz9v Click this for more info

#USE THIS VERSION!!!

def get_value_at_t(data_list, t):
    """Helper to pad lists if the timeline is longer than the provided data."""
    if not data_list:
        return 0
    return data_list[t] if t < len(data_list) else data_list[-1]

def calculate_portfolio_params(assets, asset_desc, economic_regime, expected_inflation, tax_drag):
    """Calculates the blended drift (mu) and volatility (sigma) of the portfolio."""
    mu_p = 0.0
    var_p = 0.0
    
    for asset_name, weight in assets:
        if asset_name in asset_desc:
            mu_i, sigma_sq_i = asset_desc[asset_name]
            mu_p += weight * mu_i
            var_p += (weight ** 2) * sigma_sq_i 
            
    sigma_p = np.sqrt(var_p)
    
    # Apply Economic Regime and Inflation to get REAL drift
    mu_p += economic_regime
    mu_p -= expected_inflation
    
    # Approximate Tax Drag 
    taxable_weight = tax_drag.get('taxable', 0.0)
    mu_p = mu_p * (1 - (taxable_weight * 0.15))
    
    return mu_p, sigma_p

def run_monte_carlo_wiener(sim_data, num_runs=1000):
    # 1. Extract Inputs
    assets = sim_data.get('asset_allocation', [])
    asset_desc = sim_data.get('asset_description', {})
    timeline = sim_data.get('timeline', {})
    contrib_timeline = sim_data.get('contribution_timeline', [])
    expected_inflation = sim_data.get('expected_inflation', 0.02)
    # hazard_rates = sim_data.get('hazard_rate', [])
    liquidity_events = sim_data.get('liquidity_events', [])
    tax_drag = sim_data.get('tax_drag', {})
    economic_regime = float(sim_data.get('economic_regime', 0.0))
    debt_inventory = sim_data.get('debt_inventory', [])
    
# 2. Setup Timeline
    start_age = timeline.get('current_age', 25)
    retirement_age = timeline.get('retirement_age', 65)
    end_age = retirement_age  # Force the simulation to end at retirement
    years = end_age - start_age
    
    steps_per_year = 252  # <-- NEW: 12 for monthly, 252 for daily trading days
    total_steps = years * steps_per_year
    ages = np.linspace(start_age, end_age, total_steps + 1) # <-- UPDATED for fine grid
    
    # 3. Calculate Wiener Parameters (GBM)
    mu, sigma = calculate_portfolio_params(assets, asset_desc, economic_regime, expected_inflation, tax_drag)
    dt = 1.0 / steps_per_year # <-- CHANGED: dt is now a fraction of a year
    
    # Calculate constant annual debt drag
    annual_debt_drag = sum(balance * apr for balance, apr in debt_inventory)
    
    # 4. Initialize Wealth Matrix and State Vectors
    wealth_paths = np.zeros((num_runs, total_steps + 1))
    forced_retirement = np.zeros(num_runs, dtype=bool)
# 5. Run the Simulation Loop (Vectorized across paths, sequential across time)
    for t in range(total_steps): # <-- CHANGED: Loop over all fine steps
        current_age_float = start_age + t * dt
        year_idx = int(t * dt) # <-- NEW: Used to look up the correct year in your data lists
        
# A. Apply Market Move (Euler-Maruyama approximation of GBM)
        Z = np.random.standard_normal(num_runs)
        growth_factors = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        wealth_paths[:, t+1] = wealth_paths[:, t] * growth_factors
        
# B. Add Contributions & Subtract Expenses (Scaled by dt)
        contribution = get_value_at_t(contrib_timeline, year_idx) * dt 
        # Apply cash flows to ALL paths based on age
        if current_age_float < retirement_age:
            wealth_paths[:, t+1] += contribution
        else:
            wealth_paths[:, t+1] -= (wealth_paths[:, t+1] * (0.04 * dt))
            
        # D. Apply Debt Drag (Scaled by dt)
        wealth_paths[:, t+1] -= (annual_debt_drag * dt)
        
        # E. Process Liquidity Events
        for event in liquidity_events:
            # <-- CHANGED: Trigger event only on the specific step that hits the birthday
            if abs(current_age_float - event['age']) < (dt / 2.0): 
                wealth_paths[:, t+1] -= event['cost']
                
        # F. Bankruptcy Floor
        wealth_paths[:, t+1] = np.maximum(wealth_paths[:, t+1], 0)
# F. Bankruptcy Floor (Wealth cannot drop below 0 due to compounding)
        wealth_paths[:, t+1] = np.maximum(wealth_paths[:, t+1], 0)

    # Filter out wild outliers: only keep paths where terminal wealth is within the 1st-99th percentile
    terminal_wealth = wealth_paths[:, -1]
    p1_bound = np.percentile(terminal_wealth, 1)
    p99_bound = np.percentile(terminal_wealth, 99)
    
    valid_mask = (terminal_wealth >= p1_bound) & (terminal_wealth <= p99_bound)
    filtered_wealth_paths = wealth_paths[valid_mask]

    return ages, filtered_wealth_paths

def visualize_results(ages, wealth_paths, num_runs):
    plt.figure(figsize=(12, 6))
    
    # Plot a sample of individual paths (background noise)
    sample_size = min(100, num_runs)
    for i in range(sample_size):
        plt.plot(ages, wealth_paths[i], color='blue', alpha=0.05)
        
    # Calculate Percentiles
# Calculate Percentiles
    p1 = np.percentile(wealth_paths, 1, axis=0)
    p25 = np.percentile(wealth_paths, 25, axis=0)
    p50 = np.percentile(wealth_paths, 50, axis=0)
    p75 = np.percentile(wealth_paths, 75, axis=0)
    p99 = np.percentile(wealth_paths, 99, axis=0)
    
    # Plot Percentiles
    plt.plot(ages, p99, color='green', linestyle=':', linewidth=1.5, label='99th Percentile')
    plt.plot(ages, p75, color='blue', linestyle='--', linewidth=2, label='75th Percentile (Optimistic)')
    plt.plot(ages, p50, color='black', linewidth=3, label='50th Percentile (Median)')
    plt.plot(ages, p25, color='orange', linestyle='--', linewidth=2, label='25th Percentile (Pessimistic)')
    plt.plot(ages, p1, color='red', linestyle=':', linewidth=1.5, label='1st Percentile')
    
    # Formatting
    plt.title(f'Wiener Process (GBM) Wealth Simulation ({num_runs} Runs)')
    plt.xlabel('Age')
    plt.ylabel('Real Wealth ($)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    plt.axhline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    
    plt.show()

def generate_json_snapshot(ages, wealth_paths, success_rate):
    """Converts the raw NumPy matrices into a structured JSON payload for React."""
    # Calculate Percentiles
    p1 = np.percentile(wealth_paths, 1, axis=0)
    p25 = np.percentile(wealth_paths, 25, axis=0)
    p50 = np.percentile(wealth_paths, 50, axis=0)
    p75 = np.percentile(wealth_paths, 75, axis=0)
    p99 = np.percentile(wealth_paths, 99, axis=0)
    
    # Grab 5 random paths for the frontend background "noise"
    sample_size = min(5, wealth_paths.shape[0])
    sample_indices = np.random.choice(wealth_paths.shape[0], sample_size, replace=False)
    
    payload = {
        "time_axis": ages.tolist(),
        "summary": {
            "p99": p99.tolist(),
            "p75": p75.tolist(),
            "p50": p50.tolist(),
            "p25": p25.tolist(),
            "p1": p1.tolist()
        },
        "sample_paths": wealth_paths[sample_indices].tolist(),
        "metrics": {
            "probability_of_success": float(success_rate)
        }
    }
    return payload

if __name__ == "__main__":
    try:
        sim_data = load_simulation_inputs('input.json')
    except FileNotFoundError:
        print("Error: input.json not found. Make sure the file exists in the directory.")
        exit()

    try:
        user_runs = int(input("Enter the number of Monte Carlo runs (e.g., 1000, 10000): "))
    except ValueError:
        print("Invalid input. Defaulting to 10000 runs.")
        user_runs = 10000

    print(f"Running simulation with {user_runs} paths...")
    ages, wealth_paths = run_monte_carlo_wiener(sim_data, num_runs=user_runs)
    
    # success_rate = np.mean(wealth_paths[:, -1] > 0) * 100
    # print(f"Simulation Complete. Probability of ending with > $0 at age {ages[-1]}: {success_rate:.2f}%")
    
    # visualize_results(ages, wealth_paths, user_runs)
    success_rate = np.mean(wealth_paths[:, -1] > 0) * 100
    print(f"Simulation Complete. Probability of ending with > $0 at age {ages[-1]}: {success_rate:.2f}%")
        
    # Generate the JSON payload
    json_data = generate_json_snapshot(ages, wealth_paths, success_rate)
        
    # Write to a file that React can fetch
    with open('simulation_output.json', 'w') as f:
        json.dump(json_data, f)
            
    print("Results successfully saved to simulation_output.json")