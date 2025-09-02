# Solar Thermal System — Simulation & Validation

A compact simulation of a **solar thermal collector + storage tank** with a heat exchanger and ambient heat loss. Includes plotting, tabular summaries, and a pytest suite. Run the Jupyter notebook locally, or launch it on **Binder** / **Google Colab**.

---

## Repo contents

- `solar_thermal_system.py` — `SolarThermalSystem` class (simulation, plots, summaries)
- `SolarThermalSystem.ipynb` — interactive notebook walkthrough
- `test_solareffiency.py` — unit tests (efficiency, HX transfer, energy balance, daily integration)
- `requirements.txt` — Python dependencies (used locally and by Binder)
- `Makefile` — optional helper targets (e.g., `make test`)
- `SolarThermalSystem.pdf` — optional reference/export

---

## Binderly / Google Colab

[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aldenpark/solar-tank-simulation/HEAD?urlpath=%2Fdoc%2Ftree%2FSolarThermalSystem.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldenpark/solar-tank-simulation/blob/main/SolarThermalSystem.ipynb)

---

## Virtual Environment (local)

```bash
# Create & activate a venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Open the notebook locally

```bash
jupyter notebook SolarThermalSystem.ipynb
```

---

## Makefile

```bash
# run tests:
make test
# Or run pytest directly:
pytest -q

# clean up build files (before/after tests, optional):
make clean
```

---
