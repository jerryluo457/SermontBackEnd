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

    returns 
    mu: expected upwards drift
    sigma
    """
    mu_p = 0.0
    var_p = 0.0
    
    for asset_name, weight in assets:
        # print(asset_desc)
        #asset_desc is {'S&P500': (0.09, 0.18), 'Agg_Bond': (0.04, 0.05), 'BTC': (0.45, 0.7), 'Cash': (0.02, 0.005)}
        #note the tuple format is (expected drift, STANDARD DEVIATION) and NOT variance.

        if asset_name in asset_desc:
            #mu_i is the expected return of the aggragate stock
            mu_i, sigma_i = asset_desc[asset_name]
            mu_p += weight * mu_i
            var_p += (weight ** 2) * sigma_i**2 #note that variance is stDev**2
            
    sigma_p = np.sqrt(var_p) #recover stDev from variance
    
    # Apply Economic Regime and Inflation to get REAL drift
    mu_p += 0 #not going to use the economic regeime anymore, as optimism and pessimism is built in
    #todo improvement: potentially make expected inflation a random variable?
    mu_p = ((1.0 + mu_p) / (1.0 + expected_inflation)) - 1.0 #apply fisher equation to discount inflation
    #Output is adjusted for inflation.
    
    # Approximate Tax Drag 
    taxable_weight = tax_drag.get('taxable', 0.0) #we are only considering taxable accounts for now
    mu_p = mu_p * (1 - (taxable_weight * 0.15)) #this simulates a 15% capital gains tax, reduces compounding power
    #due to capture from tax liability. 
    
    return mu_p, sigma_p #compute the final two parameters for the GBM without income input for now

def run_monte_carlo_wiener(sim_data, num_runs=1000):
    # 1) Get inputs, assets
    """
    FORMAT of the data
    "asset_allocation": "[('S&P500', 0.6), ('Agg_Bond', 0.3), ('BTC', 0.1)]",
    "asset_description": "{'S&P500': (0.09, 0.18), 'Agg_Bond': (0.04, 0.05), 'BTC': (0.45, 0.70), 'Cash': (0.02, 0.005)}",
    "timeline": "{'current_age': 25, 'retirement_age': 65, 'end_age': 95}",
    "contribution_timeline": "[10500, 11000, 11500, 12000, 12600, 13200, 14000, 14800,15600, 16500, 10500, 11000, 11500, 12000, 12600, 13200, 14000, 14800,15600, 16500, 23400, 20400, 10200, 10020]",
    "income_path": "[105000, 110000, 115000, 121000, 127000, 133000, 140000, 147000, 154000, 162000]",
    "expected_inflation": "0.025",
    "risk_aversion": "3.5",
    "hazard_rate": "[0.001, 0.0012, 0.0015, 0.0018, 0.0021, 0.0025, 0.003, 0.0035, 0.004, 0.005]",
    "liquidity_events": "[{'age': 45, 'cost': 2500, 'label': 'College'}, {'age': 32, 'cost': 100, 'label': 'House'}]",
    "tax_drag": "{'taxable': 0.4, 'roth': 0.3, 'traditional_401k': 0.3}",
    "economic_regime": "PESSIMISTIC",
    "debt_inventory": "[(15000, 0.07), (300000, 0.045)]"
    """
    assets = sim_data.get('asset_allocation', [])
    asset_desc = sim_data.get('asset_description', {})
    timeline = sim_data.get('timeline', {})
    contrib_timeline = sim_data.get('contribution_timeline', [])
    expected_inflation = sim_data.get('expected_inflation', 0.02)
    liquidity_events = sim_data.get('liquidity_events', [])
    tax_drag = sim_data.get('tax_drag', {})
    economic_regime = float(sim_data.get('economic_regime', 0.0))
    debt_inventory = sim_data.get('debt_inventory', [])
    # hazard_rates = sim_data.get('hazard_rate', []), removed the "early retirement" feature for simplicity
    
# 2) set up timeline
    start_age = timeline.get('current_age', 25) #default age is 25
    retirement_age = timeline.get('retirement_age', 65) #default age 65
    end_age = retirement_age  # We will only simulate the sim at retirement
    years = end_age - start_age
    
    steps_per_year = 252  # Monte Carlo for 252 trading days per year
    total_steps = years * steps_per_year
    ages = np.linspace(start_age, end_age, total_steps + 1) #timeline
    
    # Calculate GBM paramets: B(mu, sigma)
    mu, sigma = calculate_portfolio_params(assets, asset_desc, economic_regime, expected_inflation, tax_drag)
    dt = 1.0 / steps_per_year # dt one trading day, 1/252 years is a trading day
    
    # Calculate constant annual debt drag
    annual_debt_drag = sum(balance * apr for balance, apr in debt_inventory) #we need to deduct debt
    #payments per year
    
    # 4. Initialize Wealth Matrix and State Vectors
    wealth_paths = np.zeros((num_runs, total_steps + 1)) #
    # forced_retirement = np.zeros(num_runs, dtype=bool), feature removed for simplicity
# 5. run the sim loop
    for t in range(total_steps): 
        current_age_float = start_age + t * dt #age goes up
        year_idx = int(t * dt) #year incrementally goes up
        
# A. Apply Market Move (Euler-Maruyama approximation of GBM, impossible to compute full GBM)
        Z = np.random.standard_normal(num_runs)
        
        #the SDE is:
        #$dW_t=μW_tdt + σW_tdX_t$, essentially change in wealth is the internal growth rate (expected return times current wealth) plus any volatility
        growth_factors = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        wealth_paths[:, t+1] = wealth_paths[:, t] * growth_factors
        
# B. Add Contributions & Subtract Expenses (Scaled by dt)
        contribution = get_value_at_t(contrib_timeline, year_idx) * dt 
        # Apply cash flows to ALL paths based on age
        if current_age_float < retirement_age:
            wealth_paths[:, t+1] += contribution
        else:
            wealth_paths[:, t+1] -= (wealth_paths[:, t+1] * (0.04 * dt)) #previous wealth updated by wiener path
            
        #PROBLEM: debt serviced through personal income
        # D. Apply Debt Drag (Scaled by dt)
        # REMOVED: Debt is serviced via personal income, not portfolio liquidations
        # wealth_paths[:, t+1] -= (annual_debt_drag * dt)
        
        # E. Process Liquidity Events
        for event in liquidity_events:
            #Trigger event only on the specific step that hits the birthday
            if abs(current_age_float - event['age']) < (dt / 2.0): 
                wealth_paths[:, t+1] -= event['cost']
                
        # F. Bankruptcy Floor, cannot go below zero
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
    # p1 = np.percentile(wealth_paths, 1, axis=0)
    p25 = np.percentile(wealth_paths, 25, axis=0)
    p50 = np.percentile(wealth_paths, 50, axis=0)
    p75 = np.percentile(wealth_paths, 75, axis=0)
    # p99 = np.percentile(wealth_paths, 99, axis=0)
    
    # Plot Percentiles
    # plt.plot(ages, p99, color='green', linestyle=':', linewidth=1.5, label='99th Percentile')
    plt.plot(ages, p75, color='blue', linestyle='--', linewidth=2, label='75th Percentile (Optimistic)')
    plt.plot(ages, p50, color='black', linewidth=3, label='50th Percentile (Median)')
    plt.plot(ages, p25, color='orange', linestyle='--', linewidth=2, label='25th Percentile (Pessimistic)')
    # plt.plot(ages, p1, color='red', linestyle=':', linewidth=1.5, label='1st Percentile')
    
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
    # p1 = np.percentile(wealth_paths, 1, axis=0)
    p25 = np.percentile(wealth_paths, 25, axis=0)
    p50 = np.percentile(wealth_paths, 50, axis=0)
    p75 = np.percentile(wealth_paths, 75, axis=0)
    # p99 = np.percentile(wealth_paths, 99, axis=0)
    
    # Grab 50 random paths for the frontend background "noise"
    sample_size = min(50, wealth_paths.shape[0]) #THIS IS THE NUMBER OF MONTE CARLO PATHS THIS SIM WILL SHOW
    sample_indices = np.random.choice(wealth_paths.shape[0], sample_size, replace=False)
    
    payload = {
        "time_axis": ages.tolist(),
        "summary": {
            # "p99": p99.tolist(),
            "p75": p75.tolist(),
            "p50": p50.tolist(),
            "p25": p25.tolist(),
            # "p1": p1.tolist()
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