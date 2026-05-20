"""
src/optimizer/two_stage_bess.py — Architecture v10_revised (Combined Stage 2)
==============================================================================
Stage 1:  D-1 MILP. Commits x_c, x_d, solar routing. No y_c/y_d.
Combined Stage 2: Receding-horizon MPC at every block B.
  Jointly optimizes solar routing (s_c_rt, s_cd_rt, c_d_rt) AND
  RTM bids (y_c, y_d) in a single LP. At NC trigger blocks {34,42,50,58},
  solar forecast is updated with 12-block NC nowcast.
  Only y_c[B+3], y_d[B+3] committed per solve.
Resolution: 15-minute blocks (96 per day).
"""

import pulp
import numpy as np
from typing import Dict, Any, Tuple, List

T_BLOCKS = 96
DT = 0.25
NC_TRIGGER_BLOCKS = [34, 42, 50, 58]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_solar_band_mask(solar_profile: np.ndarray,
                            threshold: float = 0.5, buffer: int = 2) -> np.ndarray:
    n = len(solar_profile)
    mask = np.zeros(n, dtype=bool)
    solar_blocks = [t for t in range(n) if solar_profile[t] > threshold]
    if solar_blocks:
        start = max(0, min(solar_blocks) - buffer)
        end = min(n - 1, max(solar_blocks) + buffer)
        mask[start:end + 1] = True
    return mask


def compute_setpoint(soc_val: float, schedule_val: float,
                     e_min: float, e_max: float,
                     eta_c: float, eta_d: float) -> float:
    discharge_room = max(0.0, (soc_val - e_min) * eta_d)
    charge_room = max(0.0, (e_max - soc_val) / eta_c)
    total = discharge_room + charge_room + 1e-9
    bias_ratio = discharge_room / total
    return schedule_val * (0.9 + 0.2 * bias_ratio)


def compute_dsm_charge_rate(dws_pct: float, is_over: bool,
                            CR: float) -> Tuple[float, float, str]:
    pct = abs(dws_pct)
    if pct <= 10.0:
        return CR, 1.0, "0-10%"
    elif pct <= 15.0:
        return (0.90 * CR, 0.90, "10-15%") if is_over \
            else (1.10 * CR, 1.10, "10-15%")
    else:
        return (0.0, 0.0, ">15%") if is_over \
            else (1.50 * CR, 1.50, ">15%")


def compute_contract_rate(cap_comm: float, x_d: float, x_c: float,
                          y_d: float, y_c: float,
                          p_dam: float, p_rtm: float, r_ppa: float) -> float:
    ppa_mw = max(0.0, cap_comm)
    dam_s = x_d if x_d > 0 else 0.0
    rtm_s = y_d if y_d > 0 else 0.0
    total = ppa_mw + dam_s + rtm_s
    if total > 1e-9:
        return (ppa_mw * r_ppa + dam_s * p_dam + rtm_s * p_rtm) / total
    return r_ppa


def compute_dsm_settlement(cap_act: float, sched_total: float,
                           CR: float, avail_cap: float) -> Dict[str, Any]:
    act_mwh = cap_act * DT
    sch_mwh = sched_total * DT
    dws = (cap_act - sched_total) * DT
    pct = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    is_over = dws > 0
    cr, mult, band = compute_dsm_charge_rate(pct, is_over, CR)
    r: Dict[str, Any] = {
        "dws_mwh": dws, "dws_pct": pct, "band": band,
        "direction": "within" if pct <= 10 else ("over" if is_over else "under"),
        "charge_rate": cr, "charge_rate_mult": mult,
        "net_captive_cash": 0.0, "dsm_penalty": 0.0, "dsm_haircut": 0.0,
        "financial_damage": 0.0,
        "under_revenue_received": 0.0, "under_dsm_penalty": 0.0,
        "under_net_cash": 0.0, "under_if_fully_sched": 0.0, "under_damage": 0.0,
        "over_revenue_sched": 0.0, "over_revenue_dev": 0.0,
        "over_total_received": 0.0, "over_if_all_cr": 0.0, "over_haircut": 0.0,
    }
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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Day-Ahead MILP
# ══════════════════════════════════════════════════════════════════════════════

class TwoStageBESS:
    def __init__(self, params: Any, config: Dict[str, Any]):
        self.params = params
        self.config = config
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.risk_alpha = config.get("risk_alpha", 0.1)

    def solve(self, dam_scenarios: np.ndarray, rtm_scenarios: np.ndarray,
              solar_da: np.ndarray) -> Dict[str, Any]:
        """Stage 1: D-1 MILP. Commits x_c, x_d, solar routing. No y_c/y_d."""
        S = dam_scenarios.shape[0]
        p = self.params
        p_max = p.p_max_mw
        S_inv = p.solar_inverter_mw
        r_ppa = p.ppa_rate_rs_mwh
        USABLE = p.e_max_mwh - p.e_min_mwh
        solar_da = np.clip(solar_da, 0.0, S_inv)
        solar_mask = compute_solar_band_mask(
            solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

        prob = pulp.LpProblem("Stage1", pulp.LpMaximize)
        x_c = pulp.LpVariable.dicts("xc", range(T_BLOCKS), 0, p_max)
        x_d = pulp.LpVariable.dicts("xd", range(T_BLOCKS), 0, p_max)
        s_c = pulp.LpVariable.dicts("sc", range(T_BLOCKS), 0, p_max)
        s_cd = pulp.LpVariable.dicts("scd", range(T_BLOCKS), 0, S_inv)
        c_d = pulp.LpVariable.dicts("cd", range(T_BLOCKS), 0, p_max)
        delta = pulp.LpVariable.dicts("d", range(T_BLOCKS), cat="Binary")
        soc = {si: pulp.LpVariable.dicts(
            f"soc{si}", range(T_BLOCKS + 1), p.e_min_mwh, p.e_max_mwh)
            for si in range(S)}
        zeta = pulp.LpVariable("zeta")
        u = pulp.LpVariable.dicts("u", range(S), 0)
        scen_revs = []

        for si in range(S):
            prob += soc[si][0] == p.soc_initial_mwh
            prob += soc[si][T_BLOCKS] == p.soc_terminal_min_mwh
            rev = 0
            for t in range(T_BLOCKS):
                pd_t = float(dam_scenarios[si, t])
                # C4: SoC dynamics
                prob += soc[si][t + 1] == (
                    soc[si][t]
                    + p.eta_charge * (s_c[t] + x_c[t]) * DT
                    - (1.0 / p.eta_discharge) * (x_d[t] + c_d[t]) * DT)
                # Objective terms
                rev += pd_t * x_d[t] * DT - pd_t * x_c[t] * DT
                rev += r_ppa * (s_cd[t] + c_d[t]) * DT
                rev -= p.iex_fee_rs_mwh * (x_c[t] + x_d[t]) * DT
                # C6: SoC solar band
                if solar_mask[t]:
                    prob += soc[si][t] >= p.soc_solar_low
                    prob += soc[si][t] <= p.soc_solar_high
            # C7: Max 1 cycle per day (DISCHARGE side)
            prob += pulp.lpSum(
                [(x_d[t] + c_d[t]) * DT / p.eta_discharge
                 for t in range(T_BLOCKS)]
            ) <= USABLE, f"cycle_{si}"
            prob += u[si] >= zeta - rev
            scen_revs.append(rev)

        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])
            # C1: Solar balance
            prob += s_c[t] + s_cd[t] == sol_t
            # C2: PCS discharge limit
            prob += x_d[t] + c_d[t] <= p_max
            # C3: Binary mutual exclusion
            prob += x_c[t] + s_c[t] <= p_max * delta[t]
            prob += x_d[t] + c_d[t] <= p_max * (1 - delta[t])
            prob += s_cd[t] <= S_inv * (1 - delta[t])

        avg_rev = pulp.lpSum(scen_revs) / S
        cvar = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(
            [u[si] for si in range(S)])
        prob.setObjective(avg_rev + self.lambda_risk * cvar)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            z = [0.0] * T_BLOCKS
            return {"status": "Infeasible", "x_c": z, "x_d": z,
                    "s_c_da": z, "s_cd_da": z, "c_d_da": z,
                    "captive_da": z, "schedule_da": z, "setpoint_da": z,
                    "dam_schedule": z, "expected_revenue": 0.0,
                    "solar_band_mask": [], "scenarios": []}

        def v(var, t):
            return max(0.0, pulp.value(var[t]) or 0.0)
        xc_v = [v(x_c, t) for t in range(T_BLOCKS)]
        xd_v = [v(x_d, t) for t in range(T_BLOCKS)]
        sc_v = [v(s_c, t) for t in range(T_BLOCKS)]
        scd_v = [v(s_cd, t) for t in range(T_BLOCKS)]
        cd_v = [v(c_d, t) for t in range(T_BLOCKS)]
        cap_da = [scd_v[t] + cd_v[t] for t in range(T_BLOCKS)]
        dam_net = [xd_v[t] - xc_v[t] for t in range(T_BLOCKS)]
        sched_da = [cap_da[t] + dam_net[t] for t in range(T_BLOCKS)]
        soc_mean = [float(np.mean([pulp.value(soc[si][t]) or 0
                    for si in range(S)])) for t in range(T_BLOCKS + 1)]
        sp_da = [compute_setpoint(soc_mean[t], sched_da[t],
                 p.e_min_mwh, p.e_max_mwh, p.eta_charge, p.eta_discharge)
                 for t in range(T_BLOCKS)]

        return {
            "status": "Optimal",
            "expected_revenue": float(pulp.value(avg_rev) or 0),
            "x_c": xc_v, "x_d": xd_v,
            "s_c_da": sc_v, "s_cd_da": scd_v, "c_d_da": cd_v,
            "captive_da": cap_da, "schedule_da": sched_da, "setpoint_da": sp_da,
            "dam_schedule": dam_net,
            "solar_band_mask": solar_mask.tolist(),
            "scenarios": [{"soc": [pulp.value(soc[si][t])
                          for t in range(T_BLOCKS + 1)]} for si in range(S)],
        }


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED STAGE 2: Joint Solar Routing + RTM (Receding-Horizon MPC)
# ══════════════════════════════════════════════════════════════════════════════

def solve_combined_stage2(params, block_B: int, soc_actual_B: float,
                          dam_actual: np.ndarray, rtm_q50: np.ndarray,
                          p_rtm_lag4: float,
                          s_c_rt: np.ndarray, s_cd_rt: np.ndarray,
                          c_d_rt: np.ndarray,
                          y_c_committed: np.ndarray,
                          y_d_committed: np.ndarray,
                          x_c_s1: np.ndarray, x_d_s1: np.ndarray,
                          solar_blend: np.ndarray,
                          captive_committed: np.ndarray,
                          cycle_used_so_far: float = 0.0) -> Dict[str, Any]:
    """
    Combined Stage 2: jointly optimize solar routing + RTM bids.
    Solves over blocks B..95. Commits y_c[B+3], y_d[B+3] only.
    Updates s_c_rt, s_cd_rt, c_d_rt for all remaining blocks.
    """
    p = params
    p_max = p.p_max_mw
    S_inv = p.solar_inverter_mw
    r_ppa = p.ppa_rate_rs_mwh
    USABLE = p.e_max_mwh - p.e_min_mwh
    RTM_LEAD = p.rtm_lead_blocks
    CAP_BUF = p.captive_buffer_blocks
    CAP_TOL = p.captive_buffer_tolerance_mw
    remaining = T_BLOCKS - block_B
    if remaining <= 0:
        return {"status": "Skip", "y_c_bid": 0.0, "y_d_bid": 0.0,
                "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
                "captive_rt": (s_cd_rt + c_d_rt).tolist()}

    # Lag-4 RTM price conditioning
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        bias = p_rtm_lag4 - float(rtm_q50[block_B - 4])
        for t in range(block_B, T_BLOCKS):
            rtm_adj[t] = max(0.0, rtm_adj[t] + bias * 0.85 ** (t - block_B))

    # Solar band mask for remaining blocks
    sol_rem = np.array([float(solar_blend[block_B + k])
                        for k in range(remaining)])
    solar_mask_r = compute_solar_band_mask(
        sol_rem, p.solar_threshold_mw, p.solar_buffer_blocks)

    # Locked IEX from Stage 1
    xc_r = np.array(x_c_s1[block_B:], dtype=float)
    xd_r = np.array(x_d_s1[block_B:], dtype=float)

    prob = pulp.LpProblem(f"CS2_B{block_B}", pulp.LpMaximize)

    # Decision variables
    sc = pulp.LpVariable.dicts("sc", range(remaining), 0, p_max)
    scd = pulp.LpVariable.dicts("scd", range(remaining), 0, S_inv)
    cd = pulp.LpVariable.dicts("cd", range(remaining), 0, p_max)
    y_c = pulp.LpVariable.dicts("yc", range(remaining), 0, p_max)
    y_d = pulp.LpVariable.dicts("yd", range(remaining), 0, p_max)
    soc_lp = pulp.LpVariable.dicts("soc", range(remaining + 1),
                                    p.e_min_mwh, p.e_max_mwh)
    dl = pulp.LpVariable.dicts("dl", range(remaining), cat="Binary")

    # SoC initial and terminal
    prob += soc_lp[0] == float(np.clip(soc_actual_B, p.e_min_mwh, p.e_max_mwh))
    prob += soc_lp[remaining] == p.soc_terminal_min_mwh

    # Lock y_c/y_d for blocks < RTM_LEAD (already committed)
    for k in range(min(RTM_LEAD, remaining)):
        t_abs = block_B + k
        prob += y_c[k] == float(y_c_committed[t_abs])
        prob += y_d[k] == float(y_d_committed[t_abs])

    rev = 0
    for k in range(remaining):
        t_abs = block_B + k
        xc_k = float(xc_r[k])
        xd_k = float(xd_r[k])
        sol_k = float(sol_rem[k])
        pr_k = float(rtm_adj[t_abs])

        # CS2-1: Solar balance
        prob += sc[k] + scd[k] == sol_k

        # CS2-2: PCS charge limit
        prob += sc[k] + xc_k + y_c[k] <= p_max

        # CS2-3: PCS discharge limit
        prob += cd[k] + xd_k + y_d[k] <= p_max

        # CS2-4: Binary mutual exclusion
        prob += xc_k + y_c[k] + sc[k] <= p_max * dl[k]
        prob += xd_k + y_d[k] + cd[k] <= p_max * (1 - dl[k])
        prob += scd[k] <= S_inv * (1 - dl[k])

        # Force delta if locked IEX commits direction
        if xc_k > 1e-6:
            prob += dl[k] >= 1
        elif xd_k > 1e-6:
            prob += dl[k] <= 0

        # CS2-5: Captive buffer (first 12 blocks frozen ±0.5 MW)
        if k < CAP_BUF and t_abs < len(captive_committed):
            ct = float(captive_committed[t_abs])
            prob += scd[k] + cd[k] >= ct - CAP_TOL
            prob += scd[k] + cd[k] <= ct + CAP_TOL

        # CS2-6: SoC dynamics
        prob += soc_lp[k + 1] == (soc_lp[k]
            + p.eta_charge * (sc[k] + xc_k + y_c[k]) * DT
            - (1.0 / p.eta_discharge) * (cd[k] + xd_k + y_d[k]) * DT)

        # CS2-8: SoC solar band
        if solar_mask_r[k]:
            prob += soc_lp[k] >= p.soc_solar_low
            prob += soc_lp[k] <= p.soc_solar_high

        # Objective: captive PPA + RTM + locked DAM - IEX fees (ALL throughput)
        rev += r_ppa * (scd[k] + cd[k]) * DT
        rev += pr_k * y_d[k] * DT - pr_k * y_c[k] * DT
        rev += pr_k * xd_k * DT - pr_k * xc_k * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k + y_c[k] + y_d[k]) * DT

    # CS2-9: Remaining cycle budget
    cycle_budget = max(0.0, USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(cd[k] + float(xd_r[k]) + y_d[k]) * DT / p.eta_discharge
         for k in range(remaining)]
    ) <= cycle_budget, "cycle_lim"

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    # Extract results
    sc_out = s_c_rt.copy()
    scd_out = s_cd_rt.copy()
    cd_out = c_d_rt.copy()
    y_c_bid = 0.0
    y_d_bid = 0.0

    if status == "Optimal":
        for k in range(remaining):
            t = block_B + k
            sc_out[t] = max(0.0, pulp.value(sc[k]) or 0.0)
            scd_out[t] = max(0.0, pulp.value(scd[k]) or 0.0)
            cd_out[t] = max(0.0, pulp.value(cd[k]) or 0.0)
        # Committed RTM bid for block B+3
        bid_idx = RTM_LEAD  # k index for block B+3
        if bid_idx < remaining:
            y_c_bid = max(0.0, pulp.value(y_c[bid_idx]) or 0.0)
            y_d_bid = max(0.0, pulp.value(y_d[bid_idx]) or 0.0)

    return {
        "status": status,
        "y_c_bid": y_c_bid, "y_d_bid": y_d_bid,
        "s_c_rt": sc_out, "s_cd_rt": scd_out, "c_d_rt": cd_out,
        "captive_rt": (scd_out + cd_out).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ACTUALS SETTLEMENT (Orchestration Loop)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_actuals_solar(params, stage1_result: Dict,
                           dam_actual: np.ndarray, rtm_actual: np.ndarray,
                           rtm_q50: np.ndarray,
                           solar_da: np.ndarray, solar_nc: np.ndarray,
                           solar_at: np.ndarray,
                           reschedule_blocks: List[int] = NC_TRIGGER_BLOCKS,
                           verbose: bool = False) -> Dict[str, Any]:
    p = params
    r_ppa = p.ppa_rate_rs_mwh
    RTM_LEAD = p.rtm_lead_blocks
    CAP_BUF = p.captive_buffer_blocks
    avail_cap = p.avail_cap_mwh

    x_c_s1 = np.array(stage1_result["x_c"])
    x_d_s1 = np.array(stage1_result["x_d"])
    sc_da = np.array(stage1_result["s_c_da"])
    scd_da = np.array(stage1_result["s_cd_da"])
    cd_da = np.array(stage1_result["c_d_da"])
    cap_da = np.array(stage1_result["captive_da"])

    # Initialize with DA plan
    s_c_rt = sc_da.copy()
    s_cd_rt = scd_da.copy()
    c_d_rt = cd_da.copy()
    y_c_committed = np.zeros(T_BLOCKS)
    y_d_committed = np.zeros(T_BLOCKS)
    captive_committed = cap_da.copy()
    soc_path = np.zeros(T_BLOCKS + 1)
    soc_path[0] = p.soc_initial_mwh

    # Solar blend: starts as DA, updated at NC triggers
    solar_blend = solar_da.copy().astype(float)

    # Output arrays
    s_c_actual_arr = np.zeros(T_BLOCKS)
    s_cd_actual_arr = np.zeros(T_BLOCKS)
    c_d_actual_arr = np.zeros(T_BLOCKS)
    captive_actual_arr = np.zeros(T_BLOCKS)
    setpoint_arr = np.zeros(T_BLOCKS)
    schedule_rt_arr = np.zeros(T_BLOCKS)
    dsm_results = []
    block_captive_net = np.zeros(T_BLOCKS)
    block_degradation = np.zeros(T_BLOCKS)
    block_net_arr = np.zeros(T_BLOCKS)
    no_bess_dsm_arr = np.zeros(T_BLOCKS)
    no_bess_rev_arr = np.zeros(T_BLOCKS)
    cum_discharge_mwh = 0.0

    for B in range(T_BLOCKS):
        lag4 = float(rtm_actual[B - 4]) if B >= 4 else np.nan

        # Update solar blend at NC trigger blocks
        if B in reschedule_blocks:
            nc_row = solar_nc[B] if B < len(solar_nc) else np.zeros(12)
            for k in range(min(12, T_BLOCKS - B)):
                if k < len(nc_row):
                    solar_blend[B + k] = float(nc_row[k])

        # Combined Stage 2: joint solar routing + RTM bids
        res2 = solve_combined_stage2(
            params=p, block_B=B, soc_actual_B=soc_path[B],
            dam_actual=dam_actual, rtm_q50=rtm_q50, p_rtm_lag4=lag4,
            s_c_rt=s_c_rt, s_cd_rt=s_cd_rt, c_d_rt=c_d_rt,
            y_c_committed=y_c_committed, y_d_committed=y_d_committed,
            x_c_s1=x_c_s1, x_d_s1=x_d_s1,
            solar_blend=solar_blend,
            captive_committed=captive_committed,
            cycle_used_so_far=cum_discharge_mwh)

        if res2["status"] == "Optimal":
            s_c_rt = res2["s_c_rt"]
            s_cd_rt = res2["s_cd_rt"]
            c_d_rt = res2["c_d_rt"]
            # Commit RTM bid for B+3
            bid_b = B + RTM_LEAD
            if bid_b < T_BLOCKS:
                y_c_committed[bid_b] = res2["y_c_bid"]
                y_d_committed[bid_b] = res2["y_d_bid"]
            # Update captive committed beyond buffer
            for k in range(CAP_BUF, T_BLOCKS - B):
                captive_committed[B + k] = float(
                    res2["s_cd_rt"][B + k] + res2["c_d_rt"][B + k])

        # ── ACTUALS SETTLEMENT for block B ──
        xc_B = float(x_c_s1[B])
        xd_B = float(x_d_s1[B])
        yc_B = float(y_c_committed[B])
        yd_B = float(y_d_committed[B])
        dam_net_B = xd_B - xc_B
        rtm_net_B = yd_B - yc_B
        cap_rt_B = float(s_cd_rt[B] + c_d_rt[B])
        schedule_rt_B = cap_rt_B + dam_net_B + rtm_net_B
        schedule_rt_arr[B] = schedule_rt_B

        # Setpoint from SoC
        setpoint_B = compute_setpoint(
            soc_path[B], schedule_rt_B,
            p.e_min_mwh, p.e_max_mwh, p.eta_charge, p.eta_discharge)
        setpoint_arr[B] = setpoint_B

        z_at = float(solar_at[B])

        # BESS capacity (c_d_actual = total PCS discharge)
        disch_cap = max(0.0, min(
            p.p_max_mw,
            (soc_path[B] - p.e_min_mwh) * p.eta_discharge / DT))
        charge_cap = max(0.0, min(
            p.p_max_mw - max(0.0, xc_B) - max(0.0, yc_B),
            (p.e_max_mwh - soc_path[B]) / (p.eta_charge * DT)))

        # Dispatch: z_at vs setpoint
        if z_at > setpoint_B + 1e-6:
            s_c_act = min(charge_cap, z_at - setpoint_B)
            s_cd_act = z_at - s_c_act
            c_d_act = 0.0
        elif z_at < setpoint_B - 1e-6:
            s_cd_act = z_at
            c_d_act = min(disch_cap, setpoint_B - z_at)
            s_c_act = 0.0
        else:
            s_cd_act = z_at
            s_c_act = 0.0
            c_d_act = 0.0

        cap_act = s_cd_act + c_d_act
        s_c_actual_arr[B] = s_c_act
        s_cd_actual_arr[B] = s_cd_act
        c_d_actual_arr[B] = c_d_act
        captive_actual_arr[B] = cap_act

        # Contract rate + DSM
        CR = compute_contract_rate(
            captive_committed[B], xd_B, xc_B, yd_B, yc_B,
            float(dam_actual[B]), float(rtm_actual[B]), r_ppa)
        dsm = compute_dsm_settlement(cap_act, schedule_rt_B, CR, avail_cap)
        dsm_results.append(dsm)
        block_captive_net[B] = dsm["net_captive_cash"]

        # Degradation on total PCS discharge
        block_degradation[B] = p.degradation_cost_rs_mwh * c_d_act * DT

        # Block P&L = captive_net - degradation (no separate iex_net)
        block_net_arr[B] = block_captive_net[B] - block_degradation[B]

        # SoC update: c_d_actual is total PCS discharge
        ch_e = p.eta_charge * (s_c_act + xc_B + yc_B) * DT
        dis_e = c_d_act / p.eta_discharge * DT
        soc_path[B + 1] = float(np.clip(
            soc_path[B] + ch_e - dis_e, p.e_min_mwh, p.e_max_mwh))

        # Track cycle budget
        cum_discharge_mwh += c_d_act * DT / p.eta_discharge

        # No-BESS counterfactual
        nb_dsm = compute_dsm_settlement(z_at, float(captive_committed[B]),
                                         r_ppa, avail_cap)
        no_bess_dsm_arr[B] = nb_dsm["dsm_penalty"] + nb_dsm["dsm_haircut"]
        no_bess_rev_arr[B] = nb_dsm["net_captive_cash"]

        if verbose and B % 24 == 0:
            print(f"  B={B:2d} soc={soc_path[B]:.2f}→{soc_path[B+1]:.2f} "
                  f"sp={setpoint_B:.2f} z={z_at:.2f} cap={cap_act:.2f} "
                  f"net=Rs{block_net_arr[B]:,.0f}")

    total_cap_net = float(np.sum(block_captive_net))
    total_deg = float(np.sum(block_degradation))

    return {
        "revenue": total_cap_net,
        "net_revenue": total_cap_net - total_deg,
        "y_c": y_c_committed, "y_d": y_d_committed,
        "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
        "s_c_actual": s_c_actual_arr, "s_cd_actual": s_cd_actual_arr,
        "c_d_actual": c_d_actual_arr, "captive_actual": captive_actual_arr,
        "captive_committed": captive_committed,
        "setpoint_rt": setpoint_arr, "schedule_rt": schedule_rt_arr,
        "soc_path": soc_path, "soc": soc_path.tolist(),
        "z_nc_blend": solar_blend,
        "block_captive_net": block_captive_net,
        "block_degradation": block_degradation,
        "block_net": block_net_arr,
        "dsm_results": dsm_results,
        "no_bess_dsm": no_bess_dsm_arr, "no_bess_revenue": no_bess_rev_arr,
        "total_dsm_mwh": 0.0,
        "fees_breakdown": {"captive_net": total_cap_net,
                           "total_degradation": total_deg},
    }
