"""Solar thermal system simulation and visualization."""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Exact conversion: 1 MJ = 0.277777... kWh
MJ_TO_KWH = 1.0 / 3.6


class SolarThermalSystem:
    """Thermodynamically accurate solar thermal system with tank + HX."""

    def __init__(self, U_tank: float = 1.0) -> None:
        """Initialize system constants and state."""
        # Solar panel properties
        self.A_collector = 4.0  # m²
        self.eta_optical = 0.75  # -
        self.a1 = 3.5  # W/(m²·K)
        self.a2 = 0.015  # W/(m²·K²)

        # Storage tank properties
        self.V_tank = 200.0  # L
        # 200 L × 1.0 kg/L = 200 kg (water)
        self.m_tank = self.V_tank * 1.0  # kg
        # Overall tank heat loss coefficient (W/K); 0 → perfect insulation
        self.U_tank = U_tank

        # Heat transfer fluid properties
        self.cp_fluid = 3800.0  # J/(kg·K)
        self.rho_fluid = 1030.0  # kg/m³
        self.flow_rate = 0.05  # kg/s

        # System properties
        self.cp_water = 4186.0  # J/(kg·K)
        self.T_ambient = 20.0  # °C
        self.epsilon_hx = 0.8  # heat exchanger effectiveness
        self.mj_to_kwh = MJ_TO_KWH  # keep attribute for compatibility

    def solar_irradiance(self, t: float) -> float:
        """
        Return solar irradiance G(t) in W/m² as a simple daily cosine² model.

        Peak at local noon; zero at sunrise/sunset. Daylight window: 06:00–18:00.
        """
        hour = t % 24.0
        if 6.0 <= hour <= 18.0:
            hour_angle = np.pi * (hour - 12.0) / 12.0
            return float(max(0.0, 800.0 * np.cos(hour_angle) ** 2))
        return 0.0

    def collector_efficiency(self, T_in: float, T_ambient: float, G: float) -> float:
        """
        Compute collector efficiency η = η₀ − (a₁ΔT + a₂ΔT²)/G, clamped to [0, 1].

        Args:
            T_in: Collector inlet temperature (°C).
            T_ambient: Ambient temperature (°C).
            G: Solar irradiance (W/m²).

        Returns:
            Efficiency (dimensionless in [0, 1]).
        """
        if G <= 0.0:
            return 0.0
        dT = T_in - T_ambient
        eta = self.eta_optical - (self.a1 * dT + self.a2 * dT**2) / G
        return float(max(0.0, min(eta, 1.0)))

    def collector_heat_gain(self, T_in: float, T_ambient: float, G: float) -> float:
        """
        Compute useful collector heat gain (W).

        Q_useful = η(T_in, T_ambient, G) · G · A_collector
        """
        eta = self.collector_efficiency(T_in, T_ambient, G)
        return float(eta * G * self.A_collector)

    def heat_exchanger_transfer(
        self, T_hot_in: float, T_cold_in: float, Q_available: float
    ) -> float:
        """
        Compute HX heat transfer to the tank (W), capped by ΔT capacity and availability.

        Args:
            T_hot_in: Hot-side inlet (collector outlet) temperature (°C).
            T_cold_in: Cold-side inlet (tank) temperature (°C).
            Q_available: Available heat from collector (W).

        Returns:
            Actual heat delivered to tank (W), ≥ 0.
        """
        # Heat capacity rates (W/K)
        C_hot = self.flow_rate * self.cp_fluid
        C_cold = self.flow_rate * self.cp_water
        C_min = min(C_hot, C_cold)

        # Max transfer from temperature difference (W)
        Q_max_temp = C_min * (T_hot_in - T_cold_in)

        # Limit by both availability and ΔT capacity
        Q_max = min(Q_available, Q_max_temp)

        # Apply HX effectiveness and clamp non-negative
        return float(max(0.0, self.epsilon_hx * Q_max))

    def system_equations(self, state: np.ndarray, t: float) -> list[float]:
        """
        ODEs for collector outlet temperature and tank temperature.

        State vector:
            state[0] = T_collector_out (°C)
            state[1] = T_tank (°C)
        """
        T_collector_out, T_tank = state

        # Current solar conditions
        G = self.solar_irradiance(t)

        # Collector inlet temperature (return from tank)
        T_collector_in = T_tank

        # Collector heat gain (W)
        Q_useful = self.collector_heat_gain(T_collector_in, self.T_ambient, G)

        # HX analysis
        if Q_useful > 0.0 and self.flow_rate > 0.0:
            # Temperature rise in collector loop (K)
            dT_collector = Q_useful / (self.flow_rate * self.cp_fluid)
            T_collector_out_new = T_collector_in + dT_collector

            # Actual heat delivered to tank (W)
            Q_to_tank = self.heat_exchanger_transfer(
                T_collector_out_new, T_tank, Q_useful
            )
        else:
            T_collector_out_new = T_collector_in
            Q_to_tank = 0.0

        # Tank losses (W)
        Q_tank_loss = self.U_tank * (T_tank - self.T_ambient)

        # Fast collector time constant in hours (~1–2 minutes)
        tau_h = 0.02  # hr ≈ 72 s

        # ODEs: K/hr. Convert W → K/hr using 3600 s/hr.
        dT_collector_dt = (T_collector_out_new - T_collector_out) / tau_h
        dT_tank_dt = 3600.0 * (Q_to_tank - Q_tank_loss) / (self.m_tank * self.cp_water)

        return [float(dT_collector_dt), float(dT_tank_dt)]

    def simulate(self, duration_hours: float = 24.0, dt: float = 0.1) -> Dict[str, Any]:
        """
        Run the simulation and compute derived series/energies.

        Args:
            duration_hours: Total simulated hours.
            dt: Time step (hours).

        Returns:
            Dict of time series and summary scalars.
        """
        # Time array (endpoint excluded like np.arange)
        t = np.arange(0.0, duration_hours, dt)

        # Initial state [T_collector_out, T_tank] in °C
        initial_state = [self.T_ambient, self.T_ambient]

        # Solve ODEs
        solution = odeint(self.system_equations, initial_state, t)

        # Extract results
        T_collector = solution[:, 0]
        T_tank = solution[:, 1]

        # Build Q_to_tank series (W)
        q_to_tank: list[float] = []
        for i, t_hr in enumerate(t):
            G_now = self.solar_irradiance(t_hr)
            T_in = T_tank[i]
            Q_use = self.collector_heat_gain(T_in, self.T_ambient, G_now)
            if Q_use > 0.0 and self.flow_rate > 0.0:
                dT_col = Q_use / (self.flow_rate * self.cp_fluid)
                T_hot = T_in + dT_col
                Q_hx = self.heat_exchanger_transfer(T_hot, T_in, Q_use)
            else:
                Q_hx = 0.0
            q_to_tank.append(Q_hx)
        Q_to_tank_series = np.asarray(q_to_tank)

        # Additional series
        G_solar = np.asarray([self.solar_irradiance(t_hr) for t_hr in t])
        Q_useful = np.asarray(
            [self.collector_heat_gain(T_tank[i], self.T_ambient, G_solar[i]) for i in range(len(t))]
        )
        efficiency = np.asarray(
            [self.collector_efficiency(T_tank[i], self.T_ambient, G_solar[i]) for i in range(len(t))]
        )

        # Energies (MJ)
        energy_collected = np.trapezoid(Q_useful, t) * 3600.0 / 1e6
        energy_stored = (T_tank[-1] - T_tank[0]) * self.m_tank * self.cp_water / 1e6

        return {
            "time": t,
            "T_collector": T_collector,
            "T_tank": T_tank,
            "solar_irradiance": G_solar,
            "heat_gain": Q_useful,
            "efficiency": efficiency,
            "Q_to_tank": Q_to_tank_series,
            "energy_collected_MJ": energy_collected,
            "energy_stored_MJ": energy_stored,
            "final_tank_temp": float(T_tank[-1]),
        }

    def plot_results(self, results: Dict[str, Any]):
        """
        Create 2×2 plots: temperatures, inputs vs. heat, cumulative energy, daily kWh.

        This merges your previous “key metrics” daily-energy plot into the bottom-right
        axis to avoid duplication.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Temperature profiles
        ax1 = axes[0, 0]
        ax1.plot(results["time"], results["T_collector"], "r-", label="Collector Outlet", linewidth=2)
        ax1.plot(results["time"], results["T_tank"], "b-", label="Tank Temperature", linewidth=2)
        ax1.axhline(y=self.T_ambient, color="k", linestyle="--", alpha=0.5, label="Ambient")
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title("System Temperatures")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Solar irradiance and heat gain
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(
            results["time"], results["solar_irradiance"], "y-", label="Solar Irradiance", linewidth=2
        )
        line2 = ax2_twin.plot(
            results["time"], results["heat_gain"] / 1000.0, "g-", label="Heat Gain", linewidth=2
        )
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Solar Irradiance (W/m²)", color="orange")
        ax2_twin.set_ylabel("Heat Gain (kW)", color="green")
        ax2.set_title("Solar Input and Heat Collection")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Cumulative energy balance (MJ)
        ax3 = axes[1, 0]
        dt_hours = results["time"][1] - results["time"][0] if len(results["time"]) > 1 else 0.1
        cumulative_collected = np.cumsum(results["heat_gain"]) * dt_hours * 3600.0 / 1e6
        tank_energy = (
            (results["T_tank"] - results["T_tank"][0]) * self.m_tank * self.cp_water / 1e6
        )
        ax3.plot(results["time"], cumulative_collected, "r-", label="Energy Collected", linewidth=2)
        ax3.plot(results["time"], tank_energy, "b-", label="Energy Stored in Tank", linewidth=2)
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Energy (MJ)")
        ax3.set_title("Cumulative Energy Balance")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Useful Energy Delivered per Day (kWh)
        ax4 = axes[1, 1]
        time_hours = results["time"]
        heat_to_tank = results["Q_to_tank"]

        dt = time_hours[1] - time_hours[0] if len(time_hours) > 1 else 0.1
        total_days = int(time_hours[-1] / 24.0) + 1

        daily_energy_kwh: list[float] = []
        for day in range(total_days):
            start_idx = int(day * 24.0 / dt)
            end_idx = int((day + 1) * 24.0 / dt)
            if start_idx < len(heat_to_tank):
                end_idx = min(end_idx, len(heat_to_tank))
                day_heat = heat_to_tank[start_idx:end_idx]
                day_time = time_hours[start_idx:end_idx] - day * 24.0
                # Integrate W over hours → J (×3600), then J→MJ (÷1e6), then MJ→kWh
                day_mj = np.trapezoid(day_heat, day_time) * 3600.0 / 1e6
                daily_energy_kwh.append(day_mj * self.mj_to_kwh)

        days = range(1, len(daily_energy_kwh) + 1)
        ax4.plot(days, daily_energy_kwh, "bo-", linewidth=2, markersize=6)
        ax4.set_xlabel("Day")
        ax4.set_ylabel("Useful Energy Delivered (kWh)")
        ax4.set_title("Useful Energy Delivered per Day")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, None)

        plt.tight_layout()
        plt.show()
        return fig

    def print_key_metrics(self, results: Dict[str, Any]) -> None:
        """Print key metrics as text for quick inspection."""
        print("\n=== KEY METRICS OUTPUT ===")

        # Print tank temperature at 24-hour marks (first 10 days)
        print("\nTank Temperature (°C) - Every 24 hours:")
        time_hours = results["time"]
        T_tank = results["T_tank"]
        for i, t_hr in enumerate(time_hours):
            if abs(t_hr % 24.0) < 0.1 and t_hr <= 240.0:
                print(f"  Hour {t_hr:6.1f} (Day {t_hr/24.0:4.1f}): {T_tank[i]:6.2f}°C")

        # Collector useful power at noon (first 10 days)
        print("\nCollector Useful Power (kW) - At noon each day:")
        heat_gain = results["heat_gain"]
        for i, t_hr in enumerate(time_hours):
            if abs((t_hr % 24.0) - 12.0) < 0.1 and t_hr <= 240.0:
                print(f"  Day {t_hr/24.0:4.1f} noon: {heat_gain[i]/1000.0:6.2f} kW")

        # Daily energy delivered (kWh) for first 15 days
        print("\nUseful Energy Delivered per Day (kWh):")
        heat_to_tank = results.get("Q_to_tank")
        dt_hours = time_hours[1] - time_hours[0] if len(time_hours) > 1 else 0.1
        total_days = min(15, int(time_hours[-1] / 24.0) + 1)

        for day in range(total_days):
            start_idx = int(day * 24.0 / dt_hours)
            end_idx = int((day + 1) * 24.0 / dt_hours)
            if start_idx < len(heat_to_tank):
                end_idx = min(end_idx, len(heat_to_tank))
                day_heat = heat_to_tank[start_idx:end_idx]
                day_time = time_hours[start_idx:end_idx] - day * 24.0
                day_mj = np.trapezoid(day_heat, day_time) * 3600.0 / 1e6
                day_kwh = day_mj * self.mj_to_kwh
                print(f"  Day {day + 1:2d}: {day_kwh:6.2f} kWh")

        # Summary statistics
        print("\nSUMMARY:")
        print(f"  Initial Tank Temp: {T_tank[0]:6.2f}°C")
        print(f"  Final Tank Temp:   {T_tank[-1]:6.2f}°C")
        print(f"  Max Tank Temp:     {np.max(T_tank):6.2f}°C")
        print(f"  Min Tank Temp:     {np.min(T_tank):6.2f}°C")
        print(f"  Max Heat Gain:     {np.max(heat_gain)/1000.0:6.2f} kW")
        print(f"  Tank oscillates:   {(np.max(T_tank) - np.min(T_tank)) > 1.0}")

    def summary_dataframe(self, results: Dict[str, Any]) -> None:
        """Print a pandas DataFrame with summary performance metrics."""
        # Precompute available solar energy (MJ)
        available_mj = (
            np.trapezoid(results["solar_irradiance"], results["time"])
            * self.A_collector
            * 3600.0
            / 1e6
        )

        summary = {
            "Collector Area (m²)": [self.A_collector],
            "Tank Volume (L)": [self.V_tank],
            "Flow Rate (L/hr)": [self.flow_rate * 3600.0],
            "Initial Tank Temp (°C)": [results["T_tank"][0]],
            "Ambient Temp (°C)": [self.T_ambient],
            "Final Tank Temp (°C)": [results["final_tank_temp"]],
            "Temperature Rise (°C)": [
                results["final_tank_temp"] - results["T_tank"][0]
            ],
            "Peak Collector Temp (°C)": [np.max(results["T_collector"])],
            "Max Heat Collection Rate (kW)": [np.max(results["heat_gain"]) / 1000.0],
            "Peak Solar Irradiance (W/m²)": [np.max(results["solar_irradiance"])],
            "Avg Collector Efficiency (%)": [
                np.mean(results["efficiency"][results["solar_irradiance"] > 100.0]) * 100.0
            ],
            "Total Energy Collected (MJ)": [results["energy_collected_MJ"]],
            "Energy Stored in Tank (MJ)": [results["energy_stored_MJ"]],
            "System Energy Efficiency (%)": [
                (results["energy_stored_MJ"] / results["energy_collected_MJ"] * 100.0)
                if results["energy_collected_MJ"] > 0.0
                else 0.0
            ],
            "Available Solar Energy (MJ)": [available_mj],
            "Collection Efficiency (%)": [
                (results["energy_collected_MJ"] / available_mj * 100.0) if available_mj > 0.0 else 0.0
            ],
        }

        summary_df = pd.DataFrame(summary).T
        summary_df.columns = ["Value"]
        print(summary_df)

    def thermodynamic_verification(self, results: Dict[str, Any]) -> None:
        """Print basic thermodynamic checks (energy balance and stagnation bound)."""
        print("\n=== THERMODYNAMIC VERIFICATION ===")
        print("Energy Conservation Check:")
        diff_mj = results["energy_collected_MJ"] - results["energy_stored_MJ"]
        print(f"  Energy In - Energy Stored = {diff_mj:.3f} MJ")
        print("  (Difference is system loss; should be positive.)")

        # Collector stagnation temperature (theoretical max for G = 800 W/m²)
        max_theoretical = self.T_ambient + (800.0 * self.eta_optical) / self.a1
        print("\nTemperature Validation:")
        print(f"  Max theoretical stagnation: {max_theoretical:.1f} °C")
        print(f"  Max collector temperature:  {np.max(results['T_collector']):.1f} °C")
        print(
            f"  ✓ Constraint satisfied:     {np.max(results['T_collector']) < max_theoretical}"
        )
