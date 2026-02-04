# IS-6950 Independent Study - Spring 2026

**Author:** Weldon T. Antoine III

An independent study project exploring computational finance, agent-based economics, and market microstructure through a series of focused mini-projects.

---

## Project Philosophy

> *"Arbitrage is not a theorem — it is a trade."* — Figlewski (2017)

This project applies rigorous quantitative methods to understand how markets actually work, combining:
- **Austrian Economics** (praxeology, methodological individualism)
- **Bayesian Statistics** (uncertainty quantification, belief updating)
- **Agent-Based Modeling** (emergent behavior, market microstructure)

---

## Mini-Projects

### 1. Figlewski Limits-to-Arbitrage Lab ✓ **COMPLETE**

Replication and extension of Figlewski (1989) "Options Arbitrage in Imperfect Markets."

| Benchmark | Paper | Achieved |
|-----------|-------|----------|
| Residual Risk | ~6-7% | 6.64% ✓ |
| Shares/Share Hedged | ~5 | 6.1 ✓ |
| Break-even Spread | >30% | 9.6% (2× better) |

**Key Innovation:** Unified agent with 6 OHMC upgrades (event-driven rebalancing, EMA smoothing, gamma correction, Huberized loss, state-dependent costs, control variates) that dominates the Figlewski baseline.

📁 `src/figlewski/` | 📓 `notebooks/Figlewski_Limits_to_Arbitrage.ipynb`

---

### 2. Prezo.jl Package 🔄 **IN PROGRESS**

Julia package for derivatives pricing, Greeks, volatility models, and Monte Carlo methods.

**Features:**
- Black-Scholes pricing and Greeks
- GJR-GARCH volatility forecasting
- Optimal Hedged Monte Carlo (OHMC)
- ABC-SMC for Bayesian inference
- American options (LSM, OHMC)

📁 `Prezo.jl/`

---

### 3. Tesfatsion Sequential Games ⏳ **PLANNED**

Implementation of Tesfatsion (2017) "Modeling Economic Systems as Locally Constructive Sequential Games."

> Tesfatsion, L. (2017). Modeling economic systems as locally constructive sequential games. *Journal of Economic Methodology*, 24(4), 384-409. DOI: 10.1080/1350178X.2017.1382068

**Concepts:**
- Locally constructive agents
- Sequential game structure
- Emergent market dynamics
- Bounded rationality

---

### 4. Praxeology Software System ⏳ **PLANNED**

A unified framework integrating Austrian economic methodology with computational tools.

**Goals:**
- Action-based modeling (praxeological foundations)
- Subjective value and preference revelation
- Market process as discovery procedure
- Integration with Bayesian decision theory

📁 `src/Praxeology.jl`

---

### 5. ERCOT Market Simulation ⏳ **PLANNED**

Agent-based simulation of the Texas electricity market (ERCOT).

**Topics:**
- Nodal pricing and congestion
- Generator bidding strategies
- Demand response
- Market power and manipulation detection

---

## Repository Structure

```
FinTech 2.0/
├── src/
│   ├── Praxeology.jl       # Main module
│   └── figlewski/          # Figlewski lab (complete)
├── notebooks/
│   └── Figlewski_Limits_to_Arbitrage.ipynb
├── scripts/
│   └── run_figlewski.jl
├── notes/
│   └── pdf/                # Rendered reports
├── study/
│   └── Plan.md
└── README.md
```

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| **Julia** | High-performance scientific computing |
| **Prezo.jl** | Derivatives pricing, GARCH, OHMC |
| **Quarto** | Document rendering |
| **Jupyter** | Interactive analysis |

---

## References

### Completed
- Figlewski, S. (1989). Options arbitrage in imperfect markets. *Journal of Finance*.
- Figlewski, S. (2017). Derivatives valuation based on arbitrage: The trade is crucial.

### Upcoming
- Tesfatsion, L. (2017). Modeling economic systems as locally constructive sequential games. *Journal of Economic Methodology*, 24(4), 384-409.
- Mises, L. (1949). *Human Action: A Treatise on Economics*.
- Hayek, F.A. (1945). The use of knowledge in society. *American Economic Review*.

---

## Getting Started

```bash
# Activate Julia environment
cd "FinTech 2.0"
julia --project=.

# Run Figlewski experiment
julia --project=. scripts/run_figlewski.jl

# Open notebook
jupyter notebook notebooks/Figlewski_Limits_to_Arbitrage.ipynb
```

---

*"The curious task of economics is to demonstrate to men how little they really know about what they imagine they can design."* — F.A. Hayek
