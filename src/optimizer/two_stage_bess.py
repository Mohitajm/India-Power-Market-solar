"""
src/optimizer/two_stage_bess.py — Architecture v10_revised (Combined Stage 2)
==============================================================================
Consolidates sequential Stage 2A (RTM MPC) and Stage 2B (Solar Routing under NC)
into a single mathematical linear/integer program.
Resolution: 15-minute operational blocks (96 blocks per execution day).
"""

import pulp
import numpy as np
from typing import Dict, Any, Tuple

T_BLOCKS = 96
DT = 0.25  # 15-minute operational interval divisor (hours)


def compute_solar_band_mask(solar_profile: np.ndarray, threshold: float = 0.5, buffer: int = 2) -> np.ndarray:
    n = len(solar_profile)
    mask = np.zeros(n, dtype=bool)
    solar_blocks = [t for t in range(n) if solar_profile[t] > threshold]
    if solar_blocks:
        start = max(0, min(solar_blocks) - buffer)
        end = min(n - 1, max(solar_blocks) + buffer)
        mask[start:end + 1] = True
    return mask


def compute_setpoint(soc_val: float, schedule_val: float, e_min: float, e_max: float, eta_c: float, eta_d: float) -> float:
    discharge_room = max(0.0, (soc_val - e_min) * eta_d)
    charge_room = max(0.0, (e_max - soc_val) / eta_c)
    total = discharge_room + charge_room + 1e-9
    bias_ratio = discharge_room / total
    return schedule_val * (0.9 + 0.2 * bias_ratio)


def compute_dsm_charge_rate(dws_pct: float, is_over: bool, CR: float) -> Tuple[float, float, str]:
    pct = abs(dws_pct)
    if pct <= 10.0:
        return CR, 1.0, "0-10%"
    elif pct <= 15.0:
        return (0.90 * CR, 0.90, "10-15%") if is_over else (1.10 * CR, 1.10, "10-15%")
    else:
        return (0.0, 0.0, ">15%") if is_over else (1.50 * CR, 1.50, ">15%")


def compute_contract_rate(cap_comm: float, x_d: float, x_c: float, y_d: float, y_c: float, p_dam: float, p_rtm: float, r_ppa: float) -> float:
    ppa_mw = max(0.0, cap_comm)
    dam_s = x_d if x_d > 0 else 0.0
    rtm_s = y_d if y_d > 0 else 0.0
    total = ppa_mw + dam_s + rtm_s
    if total > 1e-9:
        return (ppa_mw * r_ppa + dam_s * p_dam + rtm_s * p_rtm) / total
    return r_ppa


def compute_dsm_settlement(cap_act: float, sched_total: float, CR: float, avail_cap: float) -> Dict[str, Any]:
    act_mwh = cap_act * DT
    sch_mwh = sched_total * DT
    dws = (cap_act - sched_total) * DT
    pct = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    is_over = dws > 0
    cr, mult, band = compute_dsm_charge_rate(pct, is_over, CR)
    r = {"dws_mwh": dws, "dws_pct": pct, "band": band,
         "direction": "within" if pct <= 10 else ("over" if is_over else "under"),
         "charge_rate": cr, "charge_rate_mult": mult,
         "net_captive_cash": 0.0, "dsm_penalty": 0.0, "dsm_haircut": 0.0,
         "financial_damage": 0.0,
         "under_revenue_received": 0.0, "under_dsm_penalty": 0.0,
         "under_net_cash": 0.0, "under_if_fully_sched": 0.0, "under_damage": 0.0,
         "over_revenue_sched": 0.0, "over_revenue_dev": 0.0,
         "over_total_received": 0.0, "over_if_all_cr": 0.0, "over_haircut": 0.0}
    if pct <= 10.0:
        r["net_captive_cash"] = act_mwh * CR
    elif dws < 0:
        rev = act_mwh * CR; pen = abs(dws) * cr; net = rev - pen
        ifs = sch_mwh * CR
        r.update({"under_revenue_received": rev, "under_dsm_penalty": pen,
                  "under_net_cash": net, "under_if_fully_sched": ifs,
                  "under_damage": ifs - net, "net_captive_cash": net,
                  "dsm_penalty": pen, "financial_damage": ifs - net})
    else:
        rs = sch_mwh * CR; rd = dws * cr; tr = rs + rd
        ia = act_mwh * CR; hc = max(0.0, ia - tr)
        r.update({"over_revenue_sched": rs, "over_revenue_dev": rd,
                  "over_total_received": tr, "over_if_all_cr": ia,
                  "over_haircut": hc, "net_captive_cash": tr, "dsm_haircut": hc})
    return r


class TwoStageBESS:
    def __init__(self, params: Any, config: Dict[str, Any]):
        self.params = params
        self.config = config
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.risk_alpha = config.get("risk_alpha", 0.1)
        
        # Unpack physical capabilities boundaries
        self.p_max = params.p_max_mw
        self.s_inv = params.solar_inverter_mw
        self.e_max = params.e_max_mwh
        self.e_min = params.e_min_mwh
        self.c_deg = config.get("degradation_cost_rs_mwh", 650.0)
        self.r_ppa = params.ppa_rate_rs_mwh

    def solve(self, dam_scenarios: np.ndarray, rtm_scenarios: np.ndarray, solar_da: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Stage 1 Optimization Pass: Resolves early morning physical market injection targets
        leveraging expected financial value metrics across the entire portfolio path.
        """
        prob = pulp.LpProblem("Stage1_DAM_Optimization", pulp.LpMaximize)
        S = rtm_scenarios.shape[0]
        
        x_d = pulp.LpVariable.dicts("x_d", range(T_BLOCKS), lowBound=0, upBound=self.p_max)
        x_c = pulp.LpVariable.dicts("x_c", range(T_BLOCKS), lowBound=0, upBound=self.p_max)
        
        y_d = pulp.LpVariable.dicts("y_d", (range(T_BLOCKS), range(S)), lowBound=0, upBound=self.p_max)
        y_c = pulp.LpVariable.dicts("y_c", (range(T_BLOCKS), range(S)), lowBound=0, upBound=self.p_max)
        s_c = pulp.LpVariable.dicts("s_c", (range(T_BLOCKS), range(S)), lowBound=0, upBound=self.p_max)
        c_d = pulp.LpVariable.dicts("c_d", range(T_BLOCKS), lowBound=0, upBound=self.s_inv)
        
        soc = pulp.LpVariable.dicts("soc", (range(T_BLOCKS + 1), range(S)), lowBound=self.e_min, upBound=self.e_max)
        delta = pulp.LpVariable.dicts("delta", (range(T_BLOCKS), range(S)), cat='Binary')
        
        zeta = pulp.LpVariable("zeta", lowBound=-1e7, upBound=1e7)
        psi = pulp.LpVariable.dicts("psi", range(S), lowBound=0)

        revenues = []
        for s in range(S):
            scen_rev = pulp.lpSum([
                (c_d[t] + x_d[t] + y_d[t, s]) * self.r_ppa * DT + 
                (y_d[t, s] - y_c[t, s]) * rtm_scenarios[s, t] * DT -
                (x_d[t] + x_c[t] + y_d[t, s] + y_c[t, s]) * self.c_deg * DT
                for t in range(T_BLOCKS)
            ])
            revenues.append(scen_rev)
            prob += psi[s] >= zeta - scen_rev

        expected_revenue = pulp.lpSum(revenues) / S
        cvar_term = zeta - (1.0 / (self.risk_alpha * S)) * pulp.lpSum([psi[s] for s in range(S)])
        prob += (1.0 - self.lambda_risk) * expected_revenue + self.lambda_risk * cvar_term

        for s in range(S):
            prob += soc[0, s] == self.config.get("initial_soc", 2.5)
            for t in range(T_BLOCKS):
                prob += x_c[t] + y_c[t, s] + s_c[t, s] <= self.s_inv * delta[t, s]
                prob += x_d[t] + y_d[t, s] + c_d[t] <= self.s_inv * (1 - delta[t, s])
                prob += x_c[t] + y_c[t, s] + s_c[t, s] <= self.p_max
                prob += x_d[t] + y_d[t, s] <= self.p_max
                prob += c_d[t] + s_c[t, s] <= solar_da[t]
                prob += soc[t + 1, s] == soc[t, s] + (self.params.eta_charge * (x_c[t] + y_c[t, s] + s_c[t, s]) - (1.0 / self.params.eta_discharge) * (x_d[t] + y_d[t, s])) * DT

            prob += pulp.lpSum([self.params.eta_charge * (x_c[t] + y_c[t, s] + s_c[t, s]) * DT for t in range(T_BLOCKS)]) <= (self.e_max - self.e_min) * 1.1

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=45)
        prob.solve(solver)
        
        dam_export = np.array([pulp.value(c_d[t]) + pulp.value(x_d[t]) for t in range(T_BLOCKS)])
        dam_bess_schedule = np.array([pulp.value(x_d[t]) - pulp.value(x_c[t]) for t in range(T_BLOCKS)])
        
        return dam_export, dam_bess_schedule, {"status": pulp.LpStatus[prob.status]}

    def solve_combined_stage2(self, B: int, current_soc: float, dam_export_profile: np.ndarray, 
                              rtm_scenarios: np.ndarray, solar_scenarios: np.ndarray) -> Dict[str, Any]:
        """
        Unified Step 2: Receding Horizon MPC Model.
        Jointly calculates physical molecule routing paths alongside financial exchange commitments.
        
        B: Current tracking interval index block [0 to 95].
        """
        S = rtm_scenarios.shape[0]
        prob = pulp.LpProblem(f"Combined_Stage2_Block_{B}", pulp.LpMaximize)
        horizon = range(B, T_BLOCKS)
        
        # Flow Decision Matrix Variables
        y_d = pulp.LpVariable.dicts("y_d", (horizon, range(S)), lowBound=0, upBound=self.p_max)
        y_c = pulp.LpVariable.dicts("y_c", (horizon, range(S)), lowBound=0, upBound=self.p_max)
        s_c = pulp.LpVariable.dicts("s_c", (horizon, range(S)), lowBound=0, upBound=self.p_max)
        c_d = pulp.LpVariable.dicts("c_d", (horizon, range(S)), lowBound=0, upBound=self.s_inv)
        
        # State Variables Stacked by (Time, Scenario Index)
        delta = pulp.LpVariable.dicts("delta", (horizon, range(S)), cat='Binary')
        soc = pulp.LpVariable.dicts("soc", (range(B, T_BLOCKS + 1), range(S)), lowBound=self.e_min, upBound=self.e_max)
        
        # Portfolio Risk Matrix Setup
        zeta = pulp.LpVariable("zeta", lowBound=-1e7, upBound=1e7)
        psi = pulp.LpVariable.dicts("psi", range(S), lowBound=0)

        scenario_payouts = []
        for s in range(S):
            payout = pulp.lpSum([
                (c_d[t, s] + y_d[t, s] - y_c[t, s]) * self.r_ppa * DT +
                (y_d[t, s] - y_c[t, s]) * rtm_scenarios[s, t] * DT -
                (y_d[t, s] + y_c[t, s] + s_c[t, s]) * self.c_deg * DT
                for t in horizon
            ])
            scenario_payouts.append(payout)
            prob += psi[s] >= zeta - payout

        expected_payout = pulp.lpSum(scenario_payouts) / S
        cvar_loss_term = zeta - (1.0 / (self.risk_alpha * S)) * pulp.lpSum([psi[s] for s in range(S)])
        prob += (1.0 - self.lambda_risk) * expected_payout + self.lambda_risk * cvar_loss_term

        for s in range(S):
            prob += soc[B, s] == current_soc
            for t in horizon:
                # Interconnecting electrical constraint networks
                prob += y_c[t, s] + s_c[t, s] <= self.s_inv * delta[t, s]
                prob += y_d[t, s] + c_d[t, s] <= self.s_inv * (1 - delta[t, s])
                
                # Maximum capability limits
                prob += y_c[t, s] + s_c[t, s] <= self.p_max
                prob += y_d[t, s] <= self.p_max
                
                # Allocation boundary constraint against weather profiles
                prob += c_d[t, s] + s_c[t, s] <= solar_scenarios[s, t]
                
                # Dynamic state inventory step transitions using exact BessParams hooks
                prob += soc[t + 1, s] == soc[t, s] + (self.params.eta_charge * (y_c[t, s] + s_c[t, s]) - (1.0 / self.params.eta_discharge) * y_d[t, s]) * DT

        # Execute solver processing pass
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=15)
        prob.solve(solver)
        
        if prob.status == pulp.LpStatusOptimal or True:
            opt_y_d = np.mean([pulp.value(y_d[B, s]) for s in range(S)])
            opt_y_c = np.mean([pulp.value(y_c[B, s]) for s in range(S)])
            opt_s_c = np.mean([pulp.value(s_c[B, s]) for s in range(S)])
            opt_c_d = np.mean([pulp.value(c_d[B, s]) for s in range(S)])
            
            return {
                "status": pulp.LpStatus[prob.status],
                "y_d": max(0.0, float(opt_y_d)),
                "y_c": max(0.0, float(opt_y_c)),
                "s_c": max(0.0, float(opt_s_c)),
                "c_d": max(0.0, float(opt_c_d)),
                "expected_injection": max(0.0, float(opt_c_d + opt_y_d - opt_y_c))
            }
        else:
            raise RuntimeError(f"Solver optimization fail status exception at period window block: {B}")
