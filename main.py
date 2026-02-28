"""

THIS IS A JUNK FILE, DO NOT READ


INPUTS

Asset Allocation: List[Tuple(str, float)]: [('S&P500', 0.6), ('Agg_Bond', 0.3), ('BTC', 0.1)]

Timeline: Dict: {'current_age': 25, 'retirement_age': 65, 'end_age': 95}

Contribution Schedule: List[Dict]: [{'start_age': 25, 'end_age': 30, 'monthly_contribution': 2000}, ...]

Income path / drift: List[float]: [105000, 110000, 115000, ...]

Expected inflation: float: 0.025

Risk aversion: float: 3.5 (usually from 1 to 10)

Hazard rate (Health): List[float]: [0.001, 0.0012, ... 0.05]

Liquidity events: List[Dict]: [{'age': 45, 'cost': 250000, 'label': 'College'}]

Tax drag: Dict: {'taxable': 0.4, 'roth': 0.3, 'traditional_401k': 0.3}

Economic Situation (Pessimistic to Optimistic): Int: Regime.PESSIMISTIC (mapped to -0.02 drift adjustment)

Debt Inventory: List[Tuple(balance, APR)]: [(15000, 0.07), (300000, 0.045)]
"""

# from input_loader import load_simulation_inputs

# # 1. Load the data
# sim_data = load_simulation_inputs('input.json')

# # 2. Extract into local variables for your SDE
# # Using .get() prevents the script from crashing if a key is missing
# assets = sim_data.get('asset_allocation', [])
# inflation = sim_data.get('expected_inflation', 0.02)
# salary_path = sim_data.get('income_path', [])
# risk_aversion = sim_data.get('risk_aversion', 3.0)

# #setting up my wiener process:
# #variables that are in my wiener process

# #Portfolio drift (random variable with a median and variance)
# portfolio_drift = 
# #Portfolio volatility

# #Contributions

# # 3. Run your Monte Carlo Loop
# # for path in range(num_paths):
# #    wealth = initial_S0
# #    for year in range(T):
# #        ... apply Euler-Maruyama ...