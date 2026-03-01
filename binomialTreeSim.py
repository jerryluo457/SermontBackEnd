import numpy as np
import matplotlib.pyplot as plt
from input_loader import load_simulation_inputs  # Assuming this is your loader function
import json

def get_value_at_t(data_list, t):
    """Helper to pad lists if the timeline is longer than the provided data."""
    if not data_list:
        return 0
    return data_list[t] if t < len(data_list) else 0 #If the list ends there we return 0 (assume no furthur contribution)
    # data_list[-1]

def calculate_portfolio_params(assets, asset_desc, economic_regime, expected_inflation, tax_drag):
    """Calculates the blended drift (mu) and volatility (sigma) of the portfolio."""
    """The Wiener Process / GBM has two vital parameters: mu and sigma
    
    blended drift (mu) is the expected Brownian drift of the stochastic process. For example most financial
    instruments like the S&P 500 has an expected upwards drift of about 8 to 10% per year.

        Note that mu is influenced by expected inflation, 
    
    volatility (sigma) is present in all risky financial securities. They reflect prices that fluctuate even though it has
    a tendency to drift upwards
    """

    #initialize them with zero then add more information to it from the provided JSON file.
    mu_p = 0.0
    var_p = 0.0
    
    for asset_name, weight in assets:
        """
        Going over all asset classes
        """
        if asset_name in asset_desc:
            print(asset_desc)
            mu_i, sigma_sq_i = asset_desc[asset_name]
            #the expected return (raw drift) of the asset class
            mu_p += weight * mu_i
            # For now assume zero covariance, we will include the covariance matrix later to compute the
            #true var_p
            var_p += (weight ** 2) * sigma_sq_i 

    #sigma_p is the STANDARD DEVIATION of the expected return.  
    sigma_p = np.sqrt(var_p)
    
    # Apply Economic Regime and Inflation to get REAL drift
    mu_p += economic_regime
    mu_p -= expected_inflation
    
    # Approximate Tax Drag (Assuming taxable accounts lose ~15% to capital gains/dividends)
    taxable_weight = tax_drag.get('taxable', 0.0)
    mu_p = mu_p * (1 - (taxable_weight * 0.15))
    
    return mu_p, sigma_p

def run_monte_carlo_binomial(sim_data, num_runs=1000):
    # 1. Extract Inputs using .get()
    assets = sim_data.get('asset_allocation', [])
    asset_desc = sim_data.get('asset_description', {})
    timeline = sim_data.get('timeline', {})
    contrib_timeline = sim_data.get('contribution_timeline', [])
    expected_inflation = sim_data.get('expected_inflation', 0.02)
    hazard_rates = sim_data.get('hazard_rate', [])
    liquidity_events = sim_data.get('liquidity_events', [])
    tax_drag = sim_data.get('tax_drag', {})
    economic_regime = sim_data.get('economic_regime', 0.0)
    debt_inventory = sim_data.get('debt_inventory', [])
    
    # 2. Setup Timeline
    start_age = timeline.get('current_age', 25)
    end_age = timeline.get('retirement_age', 65)
    years = end_age - start_age
    ages = np.arange(start_age, end_age + 1)
    
    # 3. Calculate Binomial Parameters (CRR Model)
    mu, sigma = calculate_portfolio_params(assets, asset_desc, economic_regime, expected_inflation, tax_drag)
    dt = 1.0 # 1 year steps
    
    # Up/Down multipliers and Probability of Up
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p_up = (np.exp(mu * dt) - d) / (u - d)
    
    # Calculate constant annual debt drag
    annual_debt_drag = sum(balance * apr for balance, apr in debt_inventory)
    
    # 4. Initialize Wealth Matrix
    # Shape: (num_runs, years + 1)
    wealth_paths = np.zeros((num_runs, years + 1))
    
    # 5. Run the Simulation Loop
    for run in range(num_runs):
        forced_retirement = False
        current_wealth = 0.0 # Assuming starting wealth of 0, add to input.json if needed
        
        for t in range(years):
            wealth_paths[run, t] = current_wealth
            current_age = start_age + t
            
            # A. Check Hazard Rate (Health/Forced Retirement)
            hazard = get_value_at_t(hazard_rates, t)
            if not forced_retirement and np.random.rand() < hazard:
                forced_retirement = True
            
            # B. Apply Market Move (The Binomial Step)
            if current_wealth > 0:
                is_up_move = np.random.rand() < p_up
                current_wealth *= (u if is_up_move else d)
            
            # C. Add Contributions (if not forced to retire)
            if not forced_retirement and current_age < timeline.get('retirement_age', 65):
                current_wealth += get_value_at_t(contrib_timeline, t)
            elif current_age >= timeline.get('retirement_age', 65) or forced_retirement:
                # Basic survival withdrawal (could be mapped to a 'survival_expenses' input)
                current_wealth -= 40000 
                
            # D. Apply Debt Drag
            current_wealth -= annual_debt_drag
            
            # E. Process Liquidity Events (College, House, etc.)
            for event in liquidity_events:
                if event['age'] == current_age:
                    current_wealth -= event['cost']
                    
            # Prevent negative wealth compounding (bankruptcy state)
            if current_wealth < 0:
                current_wealth = 0
                
        # Store final year
        wealth_paths[run, years] = current_wealth

    return ages, wealth_paths

# def visualize_results(ages, wealth_paths, num_runs):
#     plt.figure(figsize=(12, 6))
    
#     # Plot a sample of individual paths (background noise)
#     sample_size = min(100, num_runs)
#     for i in range(sample_size):
#         plt.plot(ages, wealth_paths[i], color='blue', alpha=0.05)
        
#     # Calculate Percentiles
#     p5 = np.percentile(wealth_paths, 5, axis=0)
#     p50 = np.percentile(wealth_paths, 50, axis=0)
#     p95 = np.percentile(wealth_paths, 95, axis=0)
    
#     # Plot Percentiles
#     plt.plot(ages, p95, color='green', linestyle='--', linewidth=2, label='95th Percentile (Optimistic)')
#     plt.plot(ages, p50, color='black', linewidth=3, label='50th Percentile (Median)')
#     plt.plot(ages, p5, color='red', linestyle='--', linewidth=2, label='5th Percentile (Pessimistic)')
    
#     # Formatting
#     plt.title(f'Binomial Monte Carlo Wealth Simulation ({num_runs} Runs)')
#     plt.xlabel('Age')
#     plt.ylabel('Real Wealth ($)')
#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
#     plt.axhline(0, color='black', linewidth=1)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
    
#     plt.show()

def generate_json_snapshot(ages, wealth_paths, success_rate):
    # Calculate percentiles across all paths (axis=0)
    p5, p25, p50, p75, p95 = np.percentile(wealth_paths, [5, 25, 50, 75, 95], axis=0)
    
    # Grab 5 random paths for the frontend background "noise"
    sample_indices = np.random.choice(wealth_paths.shape[0], 5, replace=False)
    
    payload = {
        "time_axis": ages.tolist(),
        "summary": {
            "p95": p95.tolist(),
            "p75": p75.tolist(),
            "p50": p50.tolist(),
            "p25": p25.tolist(),
            "p5": p5.tolist()
        },
        "sample_paths": wealth_paths[sample_indices].tolist(),
        "metrics": {
            "probability_of_success": float(success_rate)
        }
    }
    return payload

if __name__ == "__main__":
    from input_loader import load_simulation_inputs
    import numpy as np
    import json
    
    # 1. Load data
    try:
        sim_data = load_simulation_inputs('input.json')
    except FileNotFoundError:
        print("Error: input.json not found.")
        exit()

    # 2. Set runs
    user_runs = 1000  # Hardcoded for now, or use your input() prompt

    print(f"Running simulation with {user_runs} paths...")
    
    # 3. THIS IS THE MISSING LINE: Run the simulation and define wealth_paths
    ages, wealth_paths = run_monte_carlo_binomial(sim_data, num_runs=user_runs)
    
    # 4. Calculate success
    success_rate = np.mean(wealth_paths[:, -1] > 0) * 100
    
    # 5. Generate and save the JSON payload
    json_data = generate_json_snapshot(ages, wealth_paths, success_rate)
    
    with open('simulation_output.json', 'w') as f:
        json.dump(json_data, f)
        
    print("Simulation Complete. Results saved to simulation_output.json")
    # # Load data from the mock JSON loader
    # try:
    #     sim_data = load_simulation_inputs('input.json')
    # except FileNotFoundError:
    #     print("Error: input.json not found. Make sure the file exists in the directory.")
    #     exit()

    # # Get custom number of runs from the user
    # try:
    #     user_runs = int(input("Enter the number of Monte Carlo runs (e.g., 1000, 10000): "))
    # except ValueError:
    #     print("Invalid input. Defaulting to 1000 runs.")
    #     user_runs = 1000

    # print(f"Running simulation with {user_runs} paths...")
    # ages, wealth_paths = run_monte_carlo_binomial(sim_data, num_runs=user_runs)
    
    # success_rate = np.mean(wealth_paths[:, -1] > 0) * 100
    # print(f"Simulation Complete. Probability of ending with > $0 at age {ages[-1]}: {success_rate:.2f}%")
    
    # visualize_results(ages, wealth_paths, user_runs)