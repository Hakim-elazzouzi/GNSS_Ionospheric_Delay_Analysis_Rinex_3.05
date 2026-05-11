# 🌐 Project 6 — Ionospheric Delay: Geometry-Free Combination

> **L4 = Φ₁ − Φ₂ | Dual-Frequency | ROTI | Multi-Constellation | Auckland, NZ**

---

## 📌 Overview

The ionosphere introduces a range error of **1 to 50+ metres** depending on solar
activity, satellite elevation, and geographic location. This project extracts the
ionospheric delay signal directly from dual-frequency carrier-phase observations
using the **Geometry-Free (L4) combination** — no satellite positions, no clocks,
no troposphere model needed.

| Plot | What It Shows |
|------|---------------|
| 📡 GPS L4 time series | Ionospheric variation per GPS satellite over 24 hours |
| 🌡️ GPS L4 heatmap | All GPS satellites — delay direction (red/blue diverging) |
| 🌍 Multi-constellation L4 | GPS, GLONASS, Galileo, BeiDou, QZSS on the same plot |
| 📊 ROTI histogram | Ionospheric activity index + per-satellite variability |

---

## 📐 Key Equations

### Geometry-Free combination:
```
L4 = Φ₁ − Φ₂  [metres]
   = (I₂ − I₁) + (λ₁N₁ − λ₂N₂) + noise
```

All geometry cancels (ρ, clock, troposphere) — only ionosphere + ambiguity remain.

### Detrended (removes the constant ambiguity term):
```
ΔL4(t) = L4(t) − L4(t₀)  ≈  ionospheric variation during arc
```

### Rate of TEC Index (ROTI):
```
ROTI = σ(ΔL4 / Δt)   [m / epoch  or  TECU / min]
```
ROTI < 0.5 TECU/min → quiet ionosphere  
ROTI > 1.0 TECU/min → active / scintillation

### Dual-frequency pairs used:

| System | L1 code | L2 code | f₁ [MHz] | f₂ [MHz] |
|--------|---------|---------|----------|----------|
| GPS | `L1C` | `L2W` | 1575.42 | 1227.60 |
| GLONASS | `L1C` | `L2C` | 1602.00 | 1246.00 |
| Galileo | `L1X` | `L5X` | 1575.42 | 1176.45 |
| BeiDou | `L1X` | `L2I` | 1561.10 | 1207.14 |
| QZSS | `L1C` | `L2X` | 1575.42 | 1227.60 |

---

## 🖼️ Output Plots

### Plot 1 — GPS L4 Time Series
All GPS satellites, each detrended at arc start. Smooth curves = calm ionosphere.
Abrupt jumps = cycle slips (phase discontinuities).

### Plot 2 — GPS L4 Heatmap
Diverging red/blue colourmap:
- **Red** = L4 increasing (ionospheric delay growing)
- **White** = near zero (stable)
- **Blue** = L4 decreasing (delay reducing)
- **Dark grey** = satellite not tracked

### Plot 3 — Multi-Constellation L4
Best satellite per constellation on the same axes — demonstrates that all systems
observe the same ionospheric layer above Auckland, with similar overall trends.

### Plot 4 — ROTI Histogram + Per-Satellite Variability
- Left: epoch-to-epoch ΔL4 distribution with Gaussian fit
- Right: σ(ΔL4) per GPS satellite — higher bars indicate more dynamic arcs or cycle slips

---

## 📂 File Structure

```
project6-ionospheric-delay/
├── Outputs/
│   ├── plot1_gps_L4_timeseries.png
│   ├── plot2_gps_L4_heatmap.png
│   ├── plot3_multiconst_L4.png
│   └── plot4_iono_histogram.png
├── src/
│   ├── project6_ionospheric_delay__geometry_free_combination.py    
├── requirements.txt                    ← Python dependencies
├── LICENSE                             ← MIT License
└── README.md                           ← This file
```

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your RINEX file path

Update **Step 2** of the notebook:

```python
obs_path = "/path/to/your/file.rnx"
```

**Requirement:** your RINEX file must contain **dual-frequency carrier-phase observations**.
This project uses `L1C + L2W` for GPS. If your receiver only logs single-frequency,
L4 cannot be computed.

### 3. Run all cells

```bash
jupyter notebook project6_ionospheric_delay.ipynb
```

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `georinex` | Parse RINEX 3 observation files |
| `xarray` | N-dimensional labelled arrays |
| `pandas` | Time series manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Publication-quality plotting |

---

## 💡 Why Does the Ionosphere Affect GNSS?

GNSS signals travel through the ionosphere at ~300 km altitude. Free electrons
in this layer slow the signal down (code) and speed it up (phase) by the same
amount — a dispersive effect that depends on frequency:

```
Ionospheric delay on L1 ≈ 40.3 · STEC / f₁²
```

Because the delay is **frequency-dependent**, two-frequency receivers can measure
and remove it. Single-frequency receivers must use an ionospheric model (e.g.
Klobuchar for GPS, NeQuick for Galileo).

Auckland at ~37°S is a mid-latitude station with a generally **quiet ionosphere**
— ROTI values well below 1 TECU/min are expected.

---

## 👤 Author

**Hakim El Azzouzi**  
MSc Global Navigation Satellite Systems  
Mohammed First University, Oujda, Morocco  
📧 elazzouzihakim10@gmail.com  
🔗 [linkedin.com/in/Hakim-El-Azzouzi](https://linkedin.com/in/Hakim-El-Azzouzi)  
📍 Luxembourg 🇱🇺

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 Part of the GNSS RINEX Analysis Series

| # | Project |
|---|---------|
| 1 | Single GPS Satellite — Pseudorange & SNR Heatmap |
| 2 | All GPS Satellites — Fleet Pseudorange & SNR Heatmap |
| 3 | Multi-Constellation GNSS — One Satellite per System |
| 4 | Pseudorange vs Carrier-Phase Comparison |
| 5 | Constellation Summary — Pie Chart & Histograms |
| **6** | **Ionospheric Delay — Geometry-Free Combination** ← You are here |
| 7 | Data Quality Report |
