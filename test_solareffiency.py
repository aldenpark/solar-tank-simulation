"""Unit tests for SolarThermalSystem: equations & integration checks."""

from __future__ import annotations

import numpy as np
import pytest

from solar_thermal_system import SolarThermalSystem

# Common tolerances
REL_TOL = 1e-9
ABS_TOL = 1e-9
DAY_REL_TOL = 1e-3
DAY_ABS_TOL = 1e-2
NIGHT_ATOL = 1e-2


def test_collector_efficiency_formula_and_bounds() -> None:
    """Verify η formula matches spec and clamps to [0, 1]; G<=0 returns 0."""
    s = SolarThermalSystem()
    t_in, t_amb, irr = 40.0, 20.0, 800.0

    eta = s.collector_efficiency(t_in, t_amb, irr)

    d_t = t_in - t_amb
    expected = s.eta_optical - (s.a1 * d_t + s.a2 * d_t**2) / irr
    expected = max(0.0, min(expected, 1.0))

    assert eta == pytest.approx(expected, rel=REL_TOL, abs=ABS_TOL)
    assert s.collector_efficiency(30.0, 20.0, 0.0) == 0.0
    assert 0.0 <= eta <= 1.0


def test_collector_heat_gain_consistency() -> None:
    """Confirm Q_useful == η * G * A for given inputs."""
    s = SolarThermalSystem()
    t_in, t_amb, irr = 35.0, 20.0, 700.0

    eta = s.collector_efficiency(t_in, t_amb, irr)
    q = s.collector_heat_gain(t_in, t_amb, irr)

    assert q == pytest.approx(eta * irr * s.A_collector, rel=REL_TOL, abs=ABS_TOL)


def test_heat_exchanger_transfer_caps_and_effectiveness() -> None:
    """Check HX transfer caps by ΔT capacity, available heat, and clamps negatives."""
    s = SolarThermalSystem()
    t_hot_in, t_cold_in = 60.0, 40.0

    c_hot = s.flow_rate * s.cp_fluid
    c_cold = s.flow_rate * s.cp_water
    c_min = min(c_hot, c_cold)
    q_max_temp = c_min * (t_hot_in - t_cold_in)

    # Capped by temperature capacity
    q1 = s.heat_exchanger_transfer(t_hot_in, t_cold_in, q_max_temp * 2.0)
    assert q1 == pytest.approx(s.epsilon_hx * q_max_temp, rel=REL_TOL, abs=ABS_TOL)

    # Limited by available heat
    q2 = s.heat_exchanger_transfer(t_hot_in, t_cold_in, q_max_temp * 0.25)
    assert q2 == pytest.approx(
        s.epsilon_hx * q_max_temp * 0.25, rel=REL_TOL, abs=ABS_TOL
    )

    # Negative → zero
    q3 = s.heat_exchanger_transfer(t_hot_in, t_cold_in, -100.0)
    assert q3 == 0.0


def test_simulate_outputs_and_night_zero_input() -> None:
    """Ensure simulate() returns expected keys/shapes and zero transfer at night."""
    s = SolarThermalSystem(U_tank=3.0)
    res = s.simulate(duration_hours=48, dt=0.2)

    # Keys present
    required = (
        "time",
        "T_collector",
        "T_tank",
        "solar_irradiance",
        "heat_gain",
        "efficiency",
        "Q_to_tank",
        "energy_collected_MJ",
        "energy_stored_MJ",
        "final_tank_temp",
    )
    for key in required:
        assert key in res

    # Shapes match
    n = len(res["time"])
    for key in (
        "T_collector",
        "T_tank",
        "solar_irradiance",
        "heat_gain",
        "efficiency",
        "Q_to_tank",
    ):
        assert len(res[key]) == n

    # Non-negativity
    assert np.all(res["solar_irradiance"] >= -1e-9)
    assert np.all(res["heat_gain"] >= -1e-6)
    assert np.all(res["Q_to_tank"] >= -1e-6)

    # Nighttime transfer ~ 0 (outside 06–18)
    t = np.asarray(res["time"])
    night = (t % 24 < 6) | (t % 24 > 18)
    assert np.allclose(np.asarray(res["Q_to_tank"])[night], 0.0, atol=NIGHT_ATOL)


def test_energy_accounting_signs_and_consistency() -> None:
    """Validate energy_collected_MJ equals ∫Q_useful dt and >= energy_stored."""
    s = SolarThermalSystem(U_tank=3.0)
    res = s.simulate(duration_hours=24, dt=0.1)

    energy_collected = res["energy_collected_MJ"]
    assert energy_collected >= 0.0

    expected = np.trapezoid(res["heat_gain"], res["time"]) * 3600.0 / 1e6
    assert energy_collected == pytest.approx(expected, rel=REL_TOL, abs=ABS_TOL)

    assert energy_collected + 1e-6 >= res["energy_stored_MJ"]


def test_stagnation_upper_bound() -> None:
    """Bound max collector temp below simple stagnation estimate for G=800."""
    s = SolarThermalSystem()
    res = s.simulate(duration_hours=24, dt=0.1)

    max_theoretical = s.T_ambient + (800.0 * s.eta_optical) / s.a1
    assert np.max(res["T_collector"]) < max_theoretical + 1e-6


def test_daily_energy_integral_consistency() -> None:
    """Assert sum of per-day integrals equals total integral (within tolerance)."""
    s = SolarThermalSystem()
    res = s.simulate(duration_hours=48, dt=0.1)

    t = np.asarray(res["time"])
    q = np.asarray(res["Q_to_tank"])

    total = np.trapezoid(q, t)

    # Half-open day bins [start, end) to avoid double counting edges.
    n_days = int(np.ceil((t[-1] - t[0]) / 24.0))
    sum_days = 0.0
    for day in range(n_days):
        start, end = day * 24.0, (day + 1) * 24.0
        mask = (t >= start) & (t < end)
        if np.any(mask):
            sum_days += np.trapezoid(q[mask], t[mask])

    assert sum_days == pytest.approx(total, rel=DAY_REL_TOL, abs=DAY_ABS_TOL)


@pytest.mark.parametrize("duration, dt", [(48, 0.1), (48, 0.2), (24, 0.1)])
def test_daily_energy_by_masks_equals_chunked_indexing(
    duration: float, dt: float
) -> None:
    """Compare mask-based vs index-based daily integrals for multiple grids."""
    s = SolarThermalSystem()
    res = s.simulate(duration_hours=duration, dt=dt)

    t = np.asarray(res["time"])
    q = np.asarray(res["Q_to_tank"])

    # Reference: mask-based, half-open bins.
    n_days = int(np.ceil((t[-1] - t[0]) / 24.0))
    sum_masks = 0.0
    for day in range(n_days):
        start, end = day * 24.0, (day + 1) * 24.0
        m = (t >= start) & (t < end)
        if np.any(m):
            sum_masks += np.trapezoid(q[m], t[m])

    # Index-based, mirroring chunking logic used in plotting.
    if len(t) > 1:
        dt_hours = t[1] - t[0]
    else:
        dt_hours = 0.1
    total_days_index = int(t[-1] / 24.0) + 1

    sum_index = 0.0
    for day in range(total_days_index):
        start_idx = int(day * 24.0 / dt_hours)
        end_idx = int((day + 1) * 24.0 / dt_hours)
        if start_idx < len(q):
            end_idx = min(end_idx, len(q))
            day_heat = q[start_idx:end_idx]
            day_time = t[start_idx:end_idx] - day * 24.0
            sum_index += np.trapezoid(day_heat, day_time)

    assert sum_masks == pytest.approx(sum_index, rel=DAY_REL_TOL, abs=DAY_ABS_TOL)
