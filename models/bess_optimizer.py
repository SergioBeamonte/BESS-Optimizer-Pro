import pulp

def run_bess_optimization(prices_kwh, cap, power, eff, soc_ini):
    n_steps = len(prices_kwh)
    dt = 1.0
    prob = pulp.LpProblem("BESS_Arbitrage", pulp.LpMaximize)
    indices = range(n_steps)
    p_c = pulp.LpVariable.dicts("P_charge", indices, 0, power)
    p_d = pulp.LpVariable.dicts("P_discharge", indices, 0, power)
    soc = pulp.LpVariable.dicts("SoC", indices, 0, cap)
    
    prob += pulp.lpSum([prices_kwh[t] * p_d[t] * dt - prices_kwh[t] * p_c[t] * dt for t in indices])
    
    initial_soc_kwh = soc_ini * cap
    for t in indices:
        if t == 0:
            prob += soc[t] == initial_soc_kwh + p_c[t]*eff*dt - (p_d[t]/eff)*dt
        else:
            prob += soc[t] == soc[t-1] + p_c[t]*eff*dt - (p_d[t]/eff)*dt
            
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    charge_vals = [p_c[t].varValue for t in indices]
    discharge_vals = [p_d[t].varValue for t in indices]
    soc_vals = [soc[t].varValue for t in indices]
    profit_val = pulp.value(prob.objective)
    
    return charge_vals, discharge_vals, soc_vals, profit_val
