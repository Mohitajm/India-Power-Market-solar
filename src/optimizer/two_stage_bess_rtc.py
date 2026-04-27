"""
src/optimizer/two_stage_bess_rtc.py — Architecture v10 RTC FINAL
=================================================================
Three-stage Solar+BESS optimizer with Round-the-Clock (RTC) captive contract.

HARDWARE : 25.4 MWp DC / 16.4 MW PCS / 80 MWh BESS
CONTRACT : 5 MW RTC ceiling | PPA Rs 5,000/MWh | fixed 4 MW penalty threshold
TOPOLOGY : DC-coupled — Solar SCB → DC-DC → DC Bus ← BESS → PCS → AC Bus → Grid

CHANGE LOG (from earlier versions)
───────────────────────────────────
FIX-1  Topology C3: s_c[t] IS restricted by delta (all flows through single PCS).
       Old wrong code: x_c[t] <= p_max×delta[t]  [s_c was free]
       Correct code:   x_c[t]+s_c[t] <= p_max×delta[t]

FIX-2  Setpoint formula: bias heuristic clamped to ±RTC_TOL_PCT (±5%).
       setpoint = schedule × (0.90 + 0.20×bias), clamp to [0.95, 1.05] × schedule
       This ensures EMS never dispatches outside the consumer's free band.

FIX-3  C7 cycle constraint added to Stage 1 (was missing; present in Stage 2B/2A).
       prob += Σ (x_d[t]+c_d[t])×DT/η_d ≤ USABLE = 72 MWh  (per scenario)

FIX-4  C_PSHORT uses fixed 4.0 MW threshold (= 0.80×rtc_mw ceiling),
       NOT dynamic 0.80×RTC_committed. PPA contract floor is vs ceiling, not day level.

FIX-5  SOD=EOD=40 MWh hard equality (C5). soc_terminal_value removed.

FIX-6  rtc_min_mw = 1.0 MW (allows LP to commit 1 MW on very-low-SoC days).

FIX-7  SoC solar band: 15%–85% of e_max = [12, 68] MWh (was 20%–80%).

FIX-8  ppa_rate_rs_mwh = 5000 (Rs 5/unit per contract).

FIX-9  bess_params_rtc.py: rtc_pshort_threshold_mw property is fixed (not dynamic).
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_solar_band_mask_rtc(solar: np.ndarray,
                                threshold: float = 0.5,
                                buffer: int = 2) -> np.ndarray:
    """Boolean mask True during solar-generation hours (±buffer blocks)."""
    mask         = np.zeros(len(solar), dtype=bool)
    solar_blocks = [t for t in range(len(solar)) if solar[t] > threshold]
    if solar_blocks:
        s = max(0,            min(solar_blocks) - buffer)
        e = min(len(solar)-1, max(solar_blocks) + buffer)
        mask[s:e+1] = True
    return mask


def compute_setpoint_rtc(soc: float, schedule: float,
                          e_min: float, e_max: float,
                          eta_c: float, eta_d: float,
                          rtc_tol_pct: float = 0.05) -> float:
    """
    Setpoint = schedule × (0.90 + 0.20 × bias_ratio), clamped to ±rtc_tol_pct.

    The SoC bias heuristic shapes where inside the ±5% free band the EMS targets:
      SoC = e_min (8):   raw = 0.90 × schedule → clamped UP   to 0.95 × schedule
      SoC = 40 MWh (mid):raw ≈ 1.00 × schedule → within band
      SoC = e_max (80):  raw = 1.10 × schedule → clamped DOWN to 1.05 × schedule

    The ±5% clamp implements RTC_TOL_PCT — the EMS NEVER dispatches outside the
    consumer's free band without triggering the advance-notice protocol.

    Parameters
    ----------
    soc          : current SoC (MWh)
    schedule     : schedule_da[t] or schedule_rt[t] (MW) filed with SLDC
    rtc_tol_pct  : free band (default 0.05 = ±5%)
    """
    dr  = max(0.0, (soc - e_min) * eta_d)
    cr  = max(0.0, (e_max - soc) / eta_c)
    br  = dr / (dr + cr + 1e-9)
    raw = schedule * (0.90 + 0.20 * br)
    lo  = schedule * (1.0 - rtc_tol_pct)
    hi  = schedule * (1.0 + rtc_tol_pct)
    return float(np.clip(raw, lo, hi))


def compute_contract_rate(rtc_committed: float, x_d: float, y_d: float,
                          p_dam: float, p_rtm: float, r_ppa: float) -> float:
    """Blended CR = (captive×PPA + DAM×p_dam + RTM×p_rtm) / total_sell."""
    ppa = max(0.0, rtc_committed)
    dam = max(0.0, x_d)
    rtm = max(0.0, y_d)
    tot = ppa + dam + rtm
    return (ppa*r_ppa + dam*p_dam + rtm*p_rtm) / tot if tot > 1e-9 else r_ppa


def compute_dsm_settlement(captive_actual: float, scheduled: float,
                           cr: float, avail_cap: float) -> dict:
    """CERC DSM 2024 three-band settlement for one 15-min block."""
    act_mwh = captive_actual * DT
    sch_mwh = scheduled * DT
    dws     = (captive_actual - scheduled) * DT
    pct     = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    over    = dws > 0

    if pct <= 10.0:
        rate, mult, band = cr,        1.0,  "0-10%"
    elif pct <= 15.0:
        rate, mult, band = (0.90*cr, 0.90, "10-15%") if over \
                      else (1.10*cr, 1.10, "10-15%")
    else:
        rate, mult, band = (0.0,    0.0,  ">15%") if over \
                      else (1.50*cr, 1.50, ">15%")

    direction = "within" if pct <= 10 else ("over" if over else "under")
    r = {
        "dws_mwh": dws, "dws_pct": pct, "band": band,
        "direction": direction, "charge_rate": rate, "charge_rate_mult": mult,
        "net_captive_cash": 0.0, "dsm_penalty": 0.0, "dsm_haircut": 0.0,
        "financial_damage": 0.0,
        "under_revenue_received": 0.0, "under_dsm_penalty": 0.0,
        "under_net_cash": 0.0, "under_if_fully_sched": 0.0, "under_damage": 0.0,
        "over_revenue_sched": 0.0, "over_revenue_dev": 0.0,
        "over_total_received": 0.0, "over_if_all_cr": 0.0, "over_haircut": 0.0,
    }
    if pct <= 10.0:
        r["net_captive_cash"] = act_mwh * cr
    elif dws < 0:
        rev = act_mwh*cr; pen = abs(dws)*rate; net = rev-pen; ifs = sch_mwh*cr
        r.update({"under_revenue_received": rev, "under_dsm_penalty": pen,
                  "under_net_cash": net, "under_if_fully_sched": ifs,
                  "under_damage": ifs-net, "net_captive_cash": net,
                  "dsm_penalty": pen, "financial_damage": ifs-net})
    else:
        rs = sch_mwh*cr; rd = dws*rate; tr = rs+rd; ia = act_mwh*cr; hc = max(0.0,ia-tr)
        r.update({"over_revenue_sched": rs, "over_revenue_dev": rd,
                  "over_total_received": tr, "over_if_all_cr": ia,
                  "over_haircut": hc, "net_captive_cash": tr, "dsm_haircut": hc})
    return r


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════════════════

class TwoStageBESSRTC:
    """
    Stage 1 MILP — D-1 10:00 IST.

    Selects RTC_committed ∈ [rtc_min_mw=1.0, rtc_mw=5.0] MW and the
    DAM IEX arbitrage schedule simultaneously.

    C_RTC (hard equality): s_cd[t]+c_d[t] == RTC_c ∀t  →  FLAT delivery
    C3 (DC-coupled topology): x_c[t]+s_c[t] ≤ p_max×delta[t]  (s_c through PCS)
    C5: soc[s][0]==40 AND soc[s][96]==40 MWh (hard SOD=EOD)
    C7: Σ(x_d+c_d)×DT/η_d ≤ USABLE=72 MWh  (1 cycle/day per scenario)
    C_PSHORT: p_short[t] ≥ 4.0 − (s_cd[t]+c_d[t])  [4.0 = 0.80×5.0 FIXED]
    """

    def __init__(self, params, config: Dict):
        self.p           = params
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.risk_alpha  = config.get("risk_alpha",  0.10)

    def solve(self, dam_scenarios: np.ndarray,
              rtm_scenarios: np.ndarray,
              solar_da:      np.ndarray) -> Dict:
        p      = self.p
        S      = dam_scenarios.shape[0]
        p_max  = p.p_max_mw
        S_inv  = p.solar_inverter_mw
        r_ppa  = p.ppa_rate_rs_mwh
        USABLE = p.usable_energy_mwh                 # = 72 MWh
        PSHORT = p.rtc_pshort_threshold_mw           # = 4.0 MW FIXED

        solar_da   = np.clip(solar_da, 0.0, S_inv)
        solar_mask = compute_solar_band_mask_rtc(
            solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

        prob  = pulp.LpProblem("Stage1_RTC", pulp.LpMaximize)

        # ── First-stage variables (shared across all scenarios) ───────────
        RTC_c   = pulp.LpVariable("RTC_c", lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        x_c     = pulp.LpVariable.dicts("xc",  range(T_BLOCKS), 0, p_max)
        x_d     = pulp.LpVariable.dicts("xd",  range(T_BLOCKS), 0, p_max)
        s_c     = pulp.LpVariable.dicts("sc",  range(T_BLOCKS), 0, p_max)
        s_cd    = pulp.LpVariable.dicts("scd", range(T_BLOCKS), 0, S_inv)
        c_d     = pulp.LpVariable.dicts("cd",  range(T_BLOCKS), 0, p_max)
        p_short = pulp.LpVariable.dicts("psh", range(T_BLOCKS), 0)
        delta   = pulp.LpVariable.dicts("dlt", range(T_BLOCKS), cat="Binary")

        # ── Scenario variables ────────────────────────────────────────────
        soc  = {si: pulp.LpVariable.dicts(f"soc{si}", range(T_BLOCKS+1),
                                           p.e_min_mwh, p.e_max_mwh)
                for si in range(S)}
        zeta = pulp.LpVariable("zeta")
        u    = pulp.LpVariable.dicts("u", range(S), lowBound=0)

        # ── C1, C_RTC, C_PSHORT, C2, C3 — added ONCE (not per scenario) ─
        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])

            # C1: solar balance — no curtailment
            prob += s_c[t] + s_cd[t] == sol_t,                      f"C1_{t}"

            # C_RTC: flat constant delivery — the HEART of RTC
            # Night: s_cd=0 ⟹ c_d == RTC_c (BESS alone)
            # Day:   solar covers some, c_d tops up exactly to RTC_c
            prob += s_cd[t] + c_d[t] == RTC_c,                      f"CRTC_{t}"

            # C_PSHORT: linearised penalty below FIXED 4 MW threshold
            # Since C_RTC forces delivery==RTC_c, p_short = max(0, 4-RTC_c)
            # which is 0 when RTC_c≥4 and positive on very-low-SoC days
            prob += p_short[t] >= PSHORT - (s_cd[t] + c_d[t]),      f"CPSH_{t}"

            # C2: PCS discharge limit
            prob += x_d[t] + c_d[t] <= p_max,                       f"C2_{t}"

            # C3: AC bus mutual exclusion (DC-COUPLED TOPOLOGY — FIX-1)
            # ALL flows go through the single PCS → s_c shares delta with x_c
            # Import mode (delta=1): x_c + s_c ≤ p_max  (PCS charges BESS)
            # Export mode (delta=0): x_d + c_d ≤ p_max  (PCS discharges)
            #                        s_cd ≤ S_inv        (solar → captive via AC)
            #                        s_c = 0             (no charging in export mode)
            prob += x_c[t] + s_c[t]   <= p_max * delta[t],          f"C3a_{t}"
            prob += x_d[t] + c_d[t]   <= p_max * (1 - delta[t]),    f"C3b_{t}"
            prob += s_cd[t]            <= S_inv * (1 - delta[t]),    f"C3c_{t}"

        # ── Per-scenario constraints ───────────────────────────────────────
        scen_revs = []
        for si in range(S):

            # C5: SOD = EOD = 40 MWh (hard equality)
            prob += soc[si][0]        == p.soc_initial_mwh,          f"SOD_{si}"
            prob += soc[si][T_BLOCKS] == p.soc_terminal_min_mwh,     f"EOD_{si}"

            # C7: max 1 cycle per day = USABLE = 72 MWh discharge throughput
            prob += pulp.lpSum(
                [(x_d[t] + c_d[t]) * DT / p.eta_discharge
                 for t in range(T_BLOCKS)]
            ) <= USABLE,                                              f"C7_{si}"

            rev = 0
            for t in range(T_BLOCKS):
                pd_t = float(dam_scenarios[si, t])

                # C4: SoC dynamics
                prob += soc[si][t+1] == (
                    soc[si][t]
                    + p.eta_charge    * (s_c[t] + x_c[t]) * DT
                    - (1.0/p.eta_discharge) * (x_d[t] + c_d[t]) * DT
                ),                                                    f"C4_{si}_{t}"

                # C6: SoC solar band
                if solar_mask[t]:
                    prob += soc[si][t] >= p.soc_solar_low,           f"C6lo_{si}_{t}"
                    prob += soc[si][t] <= p.soc_solar_high,          f"C6hi_{si}_{t}"
                # Outside solar band: no additional constraint — BESS free [e_min,e_max]

                # Revenue
                rev += pd_t  * x_d[t]              * DT  # DAM sell
                rev -= pd_t  * x_c[t]              * DT  # DAM buy
                rev += r_ppa * (s_cd[t] + c_d[t])  * DT  # PPA on flat RTC delivery
                rev -= p.iex_fee_rs_mwh*(x_c[t]+x_d[t])* DT  # IEX fees
                rev -= r_ppa * p_short[t]           * DT  # C_PSHORT penalty

            prob += u[si] >= zeta - rev
            scen_revs.append(rev)

        # ── Objective ─────────────────────────────────────────────────────
        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0/(S*self.risk_alpha)) * pulp.lpSum(u.values())
        prob.setObjective(avg_rev + self.lambda_risk * cvar)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        z96 = [0.0] * T_BLOCKS
        if pulp.LpStatus[prob.status] != "Optimal":
            return {"status": "Infeasible", "RTC_committed": p.rtc_min_mw,
                    "x_c": z96, "x_d": z96, "s_c_da": z96, "s_cd_da": z96,
                    "c_d_da": z96, "captive_da": z96, "dam_net": z96,
                    "schedule_da": z96, "setpoint_da": z96,
                    "solar_band_mask": [False]*T_BLOCKS, "expected_revenue": 0.0,
                    "scenarios": []}

        rtc_val = float(pulp.value(RTC_c) or p.rtc_min_mw)
        xc_v  = [max(0.0, pulp.value(x_c[t]) or 0.0) for t in range(T_BLOCKS)]
        xd_v  = [max(0.0, pulp.value(x_d[t]) or 0.0) for t in range(T_BLOCKS)]
        sc_v  = [max(0.0, pulp.value(s_c[t]) or 0.0) for t in range(T_BLOCKS)]
        scd_v = [max(0.0, pulp.value(s_cd[t])or 0.0) for t in range(T_BLOCKS)]
        cd_v  = [max(0.0, pulp.value(c_d[t]) or 0.0) for t in range(T_BLOCKS)]

        cap_da   = [scd_v[t] + cd_v[t]  for t in range(T_BLOCKS)]  # = RTC_c ∀t
        dam_net  = [xd_v[t]  - xc_v[t]  for t in range(T_BLOCKS)]
        sched_da = [rtc_val  + dam_net[t] for t in range(T_BLOCKS)]

        soc_mean = [float(np.mean([pulp.value(soc[si][t]) or 0.0 for si in range(S)]))
                    for t in range(T_BLOCKS+1)]

        # FIX-2: setpoint clamped to ±rtc_tol_pct (±5% free band)
        sp_da = [compute_setpoint_rtc(soc_mean[t], sched_da[t],
                                      p.e_min_mwh, p.e_max_mwh,
                                      p.eta_charge, p.eta_discharge,
                                      p.rtc_tol_pct)
                 for t in range(T_BLOCKS)]

        return {
            "status":           "Optimal",
            "expected_revenue": float(pulp.value(avg_rev) or 0.0),
            "RTC_committed":    rtc_val,
            "x_c": xc_v, "x_d": xd_v,
            "s_c_da": sc_v, "s_cd_da": scd_v, "c_d_da": cd_v,
            "captive_da": cap_da, "dam_net": dam_net,
            "schedule_da": sched_da, "setpoint_da": sp_da,
            "solar_band_mask": solar_mask.tolist(),
            "scenarios": [{"soc": [pulp.value(soc[si][t])
                                   for t in range(T_BLOCKS+1)]}
                          for si in range(S)],
        }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B
# ══════════════════════════════════════════════════════════════════════════════

def reschedule_captive_rtc(params,
                            trigger_block:          int,
                            soc_actual:             float,
                            solar_nc_row:           np.ndarray,
                            solar_da:               np.ndarray,
                            rtm_q50:                np.ndarray,
                            x_c_s1:                 np.ndarray,
                            x_d_s1:                 np.ndarray,
                            y_c_committed:          np.ndarray,
                            y_d_committed:          np.ndarray,
                            rtc_committed:          float,
                            captive_committed_prev: np.ndarray,
                            rtc_notice:             np.ndarray,
                            cycle_used_so_far:      float = 0.0) -> Dict:
    """
    Stage 2B: revise solar routing with NC nowcast.
    Runs at trigger blocks {34,42,50,58}. Before Stage 2A at each trigger.

    cap_rt[k] bounds:
      rtc_notice[t_abs] = True  →  [rtc_min_mw, rtc_mw]  (notice given 16 blk ago)
      otherwise                 →  [RTC_c×0.95, RTC_c×1.05]  (±5% free band)

    rtc_advance_blocks used: after solving, any cap_rt > ±5% issues
    rtc_notice[t_abs + rtc_advance_blocks] = True immediately.
    """
    p         = params
    B         = trigger_block
    remaining = T_BLOCKS - B
    p_max     = p.p_max_mw
    S_inv     = p.solar_inverter_mw
    r_ppa     = p.ppa_rate_rs_mwh
    USABLE    = p.usable_energy_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    PSHORT    = p.rtc_pshort_threshold_mw   # fixed 4 MW

    # Solar blend: NC for next 12 blocks, DA beyond
    solar_blend = np.zeros(remaining, dtype=float)
    for k in range(remaining):
        t_abs = B + k
        solar_blend[k] = (float(solar_nc_row[k]) if k < len(solar_nc_row)
                          else float(solar_da[t_abs]) if t_abs < T_BLOCKS else 0.0)
    solar_blend = np.clip(solar_blend, 0.0, S_inv)
    solar_mask  = compute_solar_band_mask_rtc(solar_da, p.solar_threshold_mw,
                                               p.solar_buffer_blocks)

    xc_r  = np.array(x_c_s1[B:], dtype=float)
    xd_r  = np.array(x_d_s1[B:], dtype=float)
    yc_r  = np.array(y_c_committed[B:], dtype=float)
    yd_r  = np.array(y_d_committed[B:], dtype=float)
    rtm_r = np.array(rtm_q50[B:], dtype=float)

    prob   = pulp.LpProblem(f"S2B_b{B}", pulp.LpMaximize)
    sc     = pulp.LpVariable.dicts("sc",  range(remaining), 0, p_max)
    scd    = pulp.LpVariable.dicts("scd", range(remaining), 0, S_inv)
    cd     = pulp.LpVariable.dicts("cd",  range(remaining), 0, p_max)
    psh    = pulp.LpVariable.dicts("psh", range(remaining), 0)
    soc_v  = pulp.LpVariable.dicts("soc", range(remaining+1), p.e_min_mwh, p.e_max_mwh)
    dl     = pulp.LpVariable.dicts("dl",  range(remaining), cat="Binary")

    cap_rt = {}
    for k in range(remaining):
        t_abs = B + k
        if rtc_notice[t_abs]:
            lo, hi = p.rtc_min_mw, p.rtc_mw
        else:
            lo, hi = p.rtc_band(rtc_committed)
        cap_rt[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_v[0] == float(np.clip(soc_actual, p.e_min_mwh, p.e_max_mwh))
    if p.soc_terminal_mode == "hard":
        prob += soc_v[remaining] == p.soc_terminal_min_mwh
    else:
        prob += soc_v[remaining] >= p.soc_terminal_min_mwh

    # Remaining cycle budget
    cycle_budget = max(0.0, USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(cd[k] + float(xd_r[k]) + (float(yd_r[k]) if k < RTM_LEAD else 0.0))
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "C7_2b"

    rtc_notice_out = rtc_notice.copy()
    rev = 0

    for k in range(remaining):
        t_abs = B + k
        xc_k  = float(xc_r[k])
        xd_k  = float(xd_r[k])
        yc_k  = float(yc_r[k]) if k < RTM_LEAD else 0.0
        yd_k  = float(yd_r[k]) if k < RTM_LEAD else 0.0
        pr_k  = float(rtm_r[k])
        sol_k = float(solar_blend[k])

        # C1
        prob += sc[k] + scd[k] == sol_k,                      f"C1_{k}"
        # C_RTC
        prob += scd[k] + cd[k] == cap_rt[k],                  f"CRTC_{k}"
        # C_PSHORT (fixed 4 MW)
        prob += psh[k] >= PSHORT - (scd[k] + cd[k]),          f"PSH_{k}"
        # C2
        prob += cd[k] + xd_k + yd_k <= p_max,                 f"C2_{k}"
        # C3 (DC-coupled — s_c through PCS, delta applies)
        prob += xc_k + yc_k + sc[k] <= p_max * dl[k],         f"C3a_{k}"
        prob += xd_k + yd_k + cd[k]  <= p_max*(1-dl[k]),      f"C3b_{k}"
        prob += scd[k]                <= S_inv*(1-dl[k]),      f"C3c_{k}"
        # Captive buffer (first 12 blocks)
        if k < p.captive_buffer_blocks:
            ct = float(captive_committed_prev[t_abs])
            prob += cap_rt[k] >= ct - p.captive_buffer_tolerance_mw, f"CBlo_{k}"
            prob += cap_rt[k] <= ct + p.captive_buffer_tolerance_mw, f"CBhi_{k}"
        # C4: SoC dynamics
        prob += soc_v[k+1] == (
            soc_v[k]
            + p.eta_charge    * (sc[k] + xc_k + yc_k) * DT
            - (1.0/p.eta_discharge) * (cd[k] + xd_k + yd_k) * DT
        ),                                                      f"C4_{k}"
        # C6: solar band
        if solar_mask[t_abs]:
            prob += soc_v[k] >= p.soc_solar_low,               f"C6lo_{k}"
            prob += soc_v[k] <= p.soc_solar_high,              f"C6hi_{k}"

        # Objective
        rev += r_ppa * (scd[k] + cd[k]) * DT
        rev += pr_k  * xd_k * DT - pr_k * xc_k * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= r_ppa * psh[k] * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    sc_out  = np.zeros(T_BLOCKS)
    scd_out = np.zeros(T_BLOCKS)
    cd_out  = np.zeros(T_BLOCKS)
    cap_out = np.full(T_BLOCKS, rtc_committed, dtype=float)

    if status == "Optimal":
        for k in range(remaining):
            t_abs = B + k
            sc_out[t_abs]  = max(0.0, pulp.value(sc[k])  or 0.0)
            scd_out[t_abs] = max(0.0, pulp.value(scd[k]) or 0.0)
            cd_out[t_abs]  = max(0.0, pulp.value(cd[k])  or 0.0)
            cap_val        = float(pulp.value(cap_rt[k]) or rtc_committed)
            cap_out[t_abs] = cap_val
            # Issue rtc_notice if >5% revision needed (rtc_advance_blocks used here)
            dev = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev > p.rtc_tol_pct:
                t_notice = t_abs + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True

    return {"status": status,
            "s_c_rt": sc_out, "s_cd_rt": scd_out, "c_d_rt": cd_out,
            "captive_rt": cap_out, "rtc_notice": rtc_notice_out}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A
# ══════════════════════════════════════════════════════════════════════════════

def solve_stage2a_rtc(params,
                       block_B:           int,
                       soc_actual_B:      float,
                       dam_actual:        np.ndarray,
                       rtm_q50:           np.ndarray,
                       p_rtm_lag4:        float,
                       s_c_rt:            np.ndarray,
                       c_d_rt:            np.ndarray,
                       y_c_committed:     np.ndarray,
                       y_d_committed:     np.ndarray,
                       x_c_s1:            np.ndarray,
                       x_d_s1:            np.ndarray,
                       solar_da:          np.ndarray,
                       rtc_committed:     float,
                       captive_committed: np.ndarray,
                       rtc_notice:        np.ndarray,
                       cycle_used_so_far: float = 0.0) -> Tuple[float, float, np.ndarray]:
    """
    Stage 2A: receding-horizon MPC — every block B.
    Bids y_c/y_d for block B+RTM_LEAD=B+3.
    Issues rtc_notice[B+k+rtc_advance_blocks] for any >5% revision needed.
    rtc_advance_blocks = 16 blocks = 4 hours (used here as stated in contract).
    """
    p         = params
    p_max     = p.p_max_mw
    r_ppa     = p.ppa_rate_rs_mwh
    USABLE    = p.usable_energy_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    PSHORT    = p.rtc_pshort_threshold_mw   # fixed 4 MW
    bid_block = block_B + RTM_LEAD
    rtc_notice_out = rtc_notice.copy()

    if bid_block >= T_BLOCKS:
        return 0.0, 0.0, rtc_notice_out

    # Lag-4 RTM price bias
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        bias = p_rtm_lag4 - float(rtm_q50[block_B-4])
        for t in range(bid_block, T_BLOCKS):
            rtm_adj[t] = max(0.0, rtm_adj[t] + bias * (0.85 ** (t - block_B)))

    solar_mask = compute_solar_band_mask_rtc(solar_da, p.solar_threshold_mw,
                                              p.solar_buffer_blocks)

    # Roll SoC forward B → bid_block using committed flows
    soc_rf = float(np.clip(soc_actual_B, p.e_min_mwh, p.e_max_mwh))
    for t in range(block_B, bid_block):
        xc_t = float(x_c_s1[t]); xd_t = float(x_d_s1[t])
        yc_t = float(y_c_committed[t]); yd_t = float(y_d_committed[t])
        sc_t = float(s_c_rt[t]); cd_t = float(c_d_rt[t])
        ch   = p.eta_charge * (sc_t + xc_t + yc_t) * DT
        di   = (cd_t + xd_t + yd_t) / p.eta_discharge * DT
        soc_rf = float(np.clip(soc_rf + ch - di, p.e_min_mwh, p.e_max_mwh))

    remaining = T_BLOCKS - bid_block
    if remaining <= 0:
        return 0.0, 0.0, rtc_notice_out

    prob   = pulp.LpProblem(f"S2A_b{block_B}", pulp.LpMaximize)
    y_c    = pulp.LpVariable.dicts("yc",  range(remaining), 0, p_max)
    y_d    = pulp.LpVariable.dicts("yd",  range(remaining), 0, p_max)
    psh    = pulp.LpVariable.dicts("psh", range(remaining), 0)
    soc_lp = pulp.LpVariable.dicts("soc", range(remaining+1), p.e_min_mwh, p.e_max_mwh)
    dl     = pulp.LpVariable.dicts("dl",  range(remaining), cat="Binary")

    cap_rt_lp = {}
    for k in range(remaining):
        t_abs = bid_block + k
        if t_abs < T_BLOCKS and rtc_notice[t_abs]:
            lo, hi = p.rtc_min_mw, p.rtc_mw
        else:
            lo, hi = p.rtc_band(rtc_committed)
        cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_lp[0] == soc_rf
    if p.soc_terminal_mode == "hard":
        prob += soc_lp[remaining] == p.soc_terminal_min_mwh
    else:
        prob += soc_lp[remaining] >= p.soc_terminal_min_mwh

    # Remaining cycle budget
    cycle_budget = max(0.0, USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(float(c_d_rt[bid_block+k]) + float(x_d_s1[bid_block+k]) + y_d[k])
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "C7_2a"

    rev = 0
    for k in range(remaining):
        ta   = bid_block + k
        xc_t = float(x_c_s1[ta]); xd_t = float(x_d_s1[ta])
        sc_t = float(s_c_rt[ta]); cd_t = float(c_d_rt[ta])
        pr   = float(rtm_adj[ta])

        # PCS limits
        prob += sc_t + xc_t + y_c[k] <= p_max,              f"PCS_c_{k}"
        prob += cd_t + xd_t + y_d[k] <= p_max,              f"PCS_d_{k}"
        # C_RTC at RT level
        prob += sc_t + cd_t == cap_rt_lp[k],                 f"CRTC_{k}"
        # C_PSHORT (fixed 4 MW)
        prob += psh[k] >= PSHORT - (sc_t + cd_t),            f"PSH_{k}"
        # C3 (DC-coupled: s_c through PCS, delta applies)
        prob += xc_t + y_c[k] + sc_t <= p_max * dl[k],      f"C3a_{k}"
        prob += xd_t + y_d[k] + cd_t <= p_max*(1-dl[k]),    f"C3b_{k}"
        # C4: SoC dynamics
        prob += soc_lp[k+1] == (
            soc_lp[k]
            + p.eta_charge    * (sc_t + xc_t + y_c[k]) * DT
            - (1.0/p.eta_discharge) * (cd_t + xd_t + y_d[k]) * DT
        ),                                                    f"C4_{k}"
        # C6: solar band
        if solar_mask[ta]:
            prob += soc_lp[k] >= p.soc_solar_low,            f"C6lo_{k}"
            prob += soc_lp[k] <= p.soc_solar_high,           f"C6hi_{k}"

        # Objective
        rev += pr * y_d[k] * DT - pr * y_c[k] * DT
        rev += r_ppa * cap_rt_lp[k] * DT
        rev -= p.iex_fee_rs_mwh * (y_c[k] + y_d[k]) * DT
        rev -= r_ppa * psh[k] * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_bid = max(0.0, pulp.value(y_c[0]) or 0.0)
        y_d_bid = max(0.0, pulp.value(y_d[0]) or 0.0)
        # Issue rtc_notice for any block needing >5% revision (rtc_advance_blocks used)
        for k in range(remaining):
            cap_val = float(pulp.value(cap_rt_lp[k]) or rtc_committed)
            dev = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev > p.rtc_tol_pct:
                t_notice = bid_block + k + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True
        return y_c_bid, y_d_bid, rtc_notice_out

    return 0.0, 0.0, rtc_notice_out


# ══════════════════════════════════════════════════════════════════════════════
# ACTUALS SETTLEMENT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_actuals_rtc(params,
                          stage1_result:    Dict,
                          dam_actual:       np.ndarray,
                          rtm_actual:       np.ndarray,
                          rtm_q50:          np.ndarray,
                          solar_da:         np.ndarray,
                          solar_nc:         np.ndarray,
                          solar_at:         np.ndarray,
                          reschedule_blocks: List[int] = RESCHEDULE_BLOCKS,
                          verbose:          bool = False) -> Dict:
    """
    Block-by-block settlement loop.

    Per block steps:
      1. Stage 2B (at reschedule blocks)
      2. Stage 2A (every block, bids for B+3)
      3. Setpoint derivation (schedule × bias_ratio, clamped ±5%)
      4. Dispatch (Case A surplus / B deficit / C match)
      5. DSM settlement (CERC 2024 three-band)
      6. RTC shortfall penalty: max(0, 4.0 − captive_actual) × DT × r_ppa
      7. IEX P&L
      8. SoC update
      9. Block P&L = captive_net − rtc_penalty + iex_net − degradation
     10. No-BESS counterfactual
    """
    p             = params
    r_ppa         = p.ppa_rate_rs_mwh
    RTM_LEAD      = p.rtm_lead_blocks
    avail_cap     = p.avail_cap_mwh
    rtc_committed = float(stage1_result["RTC_committed"])
    PSHORT        = p.rtc_pshort_threshold_mw  # fixed 4 MW

    x_c_s1  = np.array(stage1_result["x_c"],    dtype=float)
    x_d_s1  = np.array(stage1_result["x_d"],    dtype=float)
    s_c_rt  = np.array(stage1_result["s_c_da"], dtype=float)
    s_cd_rt = np.array(stage1_result["s_cd_da"],dtype=float)
    c_d_rt  = np.array(stage1_result["c_d_da"], dtype=float)

    y_c_committed     = np.zeros(T_BLOCKS)
    y_d_committed     = np.zeros(T_BLOCKS)
    captive_committed = np.full(T_BLOCKS, rtc_committed, dtype=float)
    rtc_notice        = np.zeros(T_BLOCKS, dtype=bool)
    soc_path          = np.zeros(T_BLOCKS + 1)
    soc_path[0]       = p.soc_initial_mwh

    s_c_act   = np.zeros(T_BLOCKS); s_cd_act  = np.zeros(T_BLOCKS)
    c_d_act   = np.zeros(T_BLOCKS); cap_act   = np.zeros(T_BLOCKS)
    setpt     = np.zeros(T_BLOCKS); sch_rt    = np.zeros(T_BLOCKS)
    bl_capnet = np.zeros(T_BLOCKS); bl_cappen = np.zeros(T_BLOCKS)
    bl_iex    = np.zeros(T_BLOCKS); bl_deg    = np.zeros(T_BLOCKS)
    bl_net    = np.zeros(T_BLOCKS)
    nb_dsm    = np.zeros(T_BLOCKS); nb_rev    = np.zeros(T_BLOCKS)
    nb_pen    = np.zeros(T_BLOCKS)
    dsm_res   = []
    rtc_notice_issued = np.zeros(T_BLOCKS, dtype=bool)
    rtc_notice_target = np.full(T_BLOCKS, -1, dtype=int)
    cum_disch = 0.0

    for B in range(T_BLOCKS):
        lag4 = float(rtm_actual[B-4]) if B >= 4 else np.nan

        # ── Stage 2B ───────────────────────────────────────────────────────
        if B in reschedule_blocks:
            nc_row = solar_nc[B] if B < len(solar_nc) else np.zeros(12)
            r2b = reschedule_captive_rtc(
                params=p, trigger_block=B, soc_actual=soc_path[B],
                solar_nc_row=nc_row, solar_da=solar_da, rtm_q50=rtm_q50,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                rtc_committed=rtc_committed,
                captive_committed_prev=captive_committed.copy(),
                rtc_notice=rtc_notice, cycle_used_so_far=cum_disch,
            )
            if r2b["status"] == "Optimal":
                s_c_rt[B:]            = r2b["s_c_rt"][B:]
                s_cd_rt[B:]           = r2b["s_cd_rt"][B:]
                c_d_rt[B:]            = r2b["c_d_rt"][B:]
                old_notice            = rtc_notice.copy()
                rtc_notice            = r2b["rtc_notice"]
                captive_committed[B:] = r2b["captive_rt"][B:]
                for tb in range(B, T_BLOCKS):
                    if rtc_notice[tb] and not old_notice[tb]:
                        rtc_notice_issued[B] = True
                        rtc_notice_target[B] = tb

        # ── Stage 2A ───────────────────────────────────────────────────────
        bid_b = B + RTM_LEAD
        if bid_b < T_BLOCKS:
            yc_b, yd_b, rtc_notice = solve_stage2a_rtc(
                params=p, block_B=B, soc_actual_B=soc_path[B],
                dam_actual=dam_actual, rtm_q50=rtm_q50, p_rtm_lag4=lag4,
                s_c_rt=s_c_rt, c_d_rt=c_d_rt,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1, solar_da=solar_da,
                rtc_committed=rtc_committed, captive_committed=captive_committed,
                rtc_notice=rtc_notice, cycle_used_so_far=cum_disch,
            )
            y_c_committed[bid_b] = yc_b
            y_d_committed[bid_b] = yd_b

        # ── Step 1: Inputs ─────────────────────────────────────────────────
        xc_B = float(x_c_s1[B]); xd_B = float(x_d_s1[B])
        yc_B = float(y_c_committed[B]); yd_B = float(y_d_committed[B])
        cap_rt_B  = float(s_cd_rt[B] + c_d_rt[B])
        sch_rt_B  = cap_rt_B + (xd_B - xc_B) + (yd_B - yc_B)
        sch_rt[B] = sch_rt_B
        z_at      = float(solar_at[B])

        # Setpoint clamped to ±5% (FIX-2)
        sp_B = compute_setpoint_rtc(soc_path[B], sch_rt_B,
                                    p.e_min_mwh, p.e_max_mwh,
                                    p.eta_charge, p.eta_discharge,
                                    p.rtc_tol_pct)
        setpt[B] = sp_B

        # SoC reserve guard: cap x_d so BESS keeps enough for remaining RTC night
        blocks_left     = T_BLOCKS - B
        rtc_reserve_soc = rtc_committed * blocks_left * DT / p.eta_discharge
        xd_avail        = max(0.0, (soc_path[B] - rtc_reserve_soc - p.e_min_mwh)
                              * p.eta_discharge / DT)
        xd_B            = min(xd_B, xd_avail)  # cap to reserve

        disch_cap = max(0.0, min(
            p.p_max_mw - xd_B - yd_B,
            (soc_path[B] - p.e_min_mwh) * p.eta_discharge / DT))
        charg_cap = max(0.0, min(
            p.p_max_mw - xc_B - yc_B,
            (p.e_max_mwh - soc_path[B]) / (p.eta_charge * DT)))

        # ── Step 2: Dispatch (Case A / B / C) ─────────────────────────────
        if z_at > sp_B + 1e-6:          # Case A: solar surplus → charge BESS
            sc_a  = min(charg_cap, z_at - sp_B)
            scd_a = z_at - sc_a
            cd_a  = 0.0
        elif z_at < sp_B - 1e-6:        # Case B: deficit → BESS discharges
            scd_a = z_at
            cd_a  = min(disch_cap, sp_B - z_at)
            sc_a  = 0.0
        else:                            # Case C: exact match
            scd_a = z_at; sc_a = 0.0; cd_a = 0.0

        cap_a = scd_a + cd_a
        s_c_act[B] = sc_a; s_cd_act[B] = scd_a
        c_d_act[B] = cd_a; cap_act[B]  = cap_a

        # ── Step 3: DSM settlement ─────────────────────────────────────────
        cr  = compute_contract_rate(rtc_committed, xd_B, yd_B,
                                    float(dam_actual[B]), float(rtm_actual[B]), r_ppa)
        dsm = compute_dsm_settlement(cap_a, sch_rt_B, cr, avail_cap)
        dsm_res.append(dsm)

        # ── Step 4: RTC shortfall penalty (FIXED 4 MW threshold) ──────────
        short_mw  = max(0.0, PSHORT - cap_a)           # 4.0 - captive_actual
        short_mwh = short_mw * DT
        rtc_pen   = short_mwh * r_ppa                  # Rs
        bl_cappen[B] = rtc_pen
        bl_capnet[B] = dsm["net_captive_cash"] - rtc_pen

        # ── Step 5: IEX ────────────────────────────────────────────────────
        iex_dam  = float(dam_actual[B]) * (xd_B - xc_B) * DT
        iex_rtm  = float(rtm_actual[B]) * (yd_B - yc_B) * DT
        iex_fees = p.iex_fee_rs_mwh * (xc_B + xd_B + yc_B + yd_B) * DT
        bl_iex[B] = iex_dam + iex_rtm - iex_fees

        # ── Step 6: SoC update ─────────────────────────────────────────────
        tot_d = xd_B + cd_a + yd_B
        ch_e  = p.eta_charge * (sc_a + xc_B + yc_B) * DT
        di_e  = tot_d / p.eta_discharge * DT
        soc_path[B+1] = float(np.clip(
            soc_path[B] + ch_e - di_e, p.e_min_mwh, p.e_max_mwh))
        cum_disch += tot_d * DT / p.eta_discharge

        # ── Step 7: Block P&L ──────────────────────────────────────────────
        deg       = p.degradation_cost_rs_mwh * tot_d * DT
        bl_deg[B] = deg
        bl_net[B] = bl_capnet[B] + bl_iex[B] - deg

        # ── Step 8: No-BESS counterfactual ────────────────────────────────
        nb_d   = compute_dsm_settlement(z_at, float(captive_committed[B]), r_ppa, avail_cap)
        nb_pen[B] = max(0.0, PSHORT - z_at) * DT * r_ppa
        nb_dsm[B] = nb_d["dsm_penalty"] + nb_d["dsm_haircut"]
        nb_rev[B] = z_at * DT * r_ppa - nb_dsm[B] - nb_pen[B]

        if verbose and B % 16 == 0:
            print(f"  B={B:02d} soc={soc_path[B]:.1f}→{soc_path[B+1]:.1f} "
                  f"sol={z_at:.2f} cap={cap_a:.2f} pen=₹{rtc_pen:,.0f} "
                  f"iex=₹{bl_iex[B]:,.0f} net=₹{bl_net[B]:,.0f}")

    with_dsm      = sum(d["dsm_penalty"] + d["dsm_haircut"] for d in dsm_res)
    bess_dsm_sav  = float(np.sum(nb_dsm)) - with_dsm
    bess_pen_sav  = float(np.sum(nb_pen)) - float(np.sum(bl_cappen))

    return {
        "net_revenue":           float(np.sum(bl_net)),
        "captive_net_total":     float(np.sum(bl_capnet)),
        "iex_net_total":         float(np.sum(bl_iex)),
        "rtc_penalty_total":     float(np.sum(bl_cappen)),
        "degradation_total":     float(np.sum(bl_deg)),
        "no_bess_revenue_total": float(np.sum(nb_rev)),
        "bess_dsm_savings":      bess_dsm_sav,
        "bess_rtc_pen_savings":  bess_pen_sav,
        "bess_total_value":      bess_dsm_sav + bess_pen_sav
                                 + float(np.sum(bl_iex))
                                 - float(np.sum(bl_deg)),
        "soc_path":              soc_path,
        "s_c_actual":  s_c_act,    "s_cd_actual":  s_cd_act,
        "c_d_actual":  c_d_act,    "captive_actual": cap_act,
        "setpoint":    setpt,       "schedule_rt":   sch_rt,
        "captive_committed": captive_committed,
        "block_captive_net":     bl_capnet,
        "block_captive_penalty": bl_cappen,
        "block_iex_net":         bl_iex,
        "block_degradation":     bl_deg,
        "block_net":             bl_net,
        "no_bess_dsm":           nb_dsm,
        "no_bess_rtc_penalty":   nb_pen,
        "no_bess_revenue":       nb_rev,
        "dsm_results":           dsm_res,
        "rtc_notice_issued":     rtc_notice_issued,
        "rtc_notice_target":     rtc_notice_target,
        "x_c":  x_c_s1,  "x_d":  x_d_s1,
        "y_c":  y_c_committed, "y_d": y_d_committed,
        "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
    }
