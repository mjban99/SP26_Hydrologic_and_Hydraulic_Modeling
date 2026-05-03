# SP26_Hydrologic_and_Hydraulic_Modeling
Final Project: WATER SYS MDLNG AND SIMULATION
Please download everything and start with the notebooks.


## Repository structure
    UHF_SIR/
    ├── README.md
    ├── requirements.txt
    ├── data/                                # Pre-processed storm data
    ├── uhf_sir/                             # Python package (Class library)
    │   ├── __init__.py
    │   ├── model.py
    │   ├── sir.py
    │   ├── metrics.py
    │   └── visualization.py
    ├── notebooks/                           # Step-by-step walkthroughs
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_model_demo.ipynb              # Forward Model unrolled
    │   ├── 03_prior_monte_carlo.ipynb
    │   ├── 04_sir_sweep.ipynb               # SIR algorithm unrolled
    │   └── 05_results_and_figures.ipynb
    ├── scripts/                             # Batch entry points (call the classes)
    │   ├── run_prior_mc.py
    │   ├── run_sir_sweep.py
    │   └── reproduce_appendix_C.py
    └── outputs/                             # Generated results

## uhf_sir file descriptions:
| Component  | File                       | Purpose                                          |
|------------|----------------------------|--------------------------------------------------|
| Model      | `uhf_sir/model.py`         | Forward simulation: hillslope + channel + tracer |
| SIR        | `uhf_sir/sir.py`           | Particle filter for posterior parameter inference|
| Metrics    | `uhf_sir/metrics.py`       | NSE, KGE, IQR reduction, tracer gain             |
| Plotting   | `uhf_sir/visualization.py` | Manuscript figures                               |


## How to read this repository
Three files have three roles:
- The **classes** in `uhf_sir/` are the production library. They encapsulate the forward model and the SIR particle filter and are imported by the scripts and by other notebooks.
- The **scripts** in `scripts/` are batch entry points. They call the classes to run heavy jobs (Monte Carlo simulation, SIR sweep, Appendix C reproduction).
- The **notebooks** in `notebooks/` are step-by-step walkthroughs that unroll the algorithms cell-by-cell. They expose the same logic as the classes, but spread out as code cells with equations and intermediate plots, for reading rather than for batch use.

If you want to **read** the algorithms, please start with the notebooks (`02_model_demo.ipynb` for the forward model, `04_sir_sweep.ipynb` for the SIR particle filter). The notebooks are self-contained: they do not require the classes for the algorithm logic, only for the final cross-check at the end. If you want to **run** the workflow end-to-end, use the scripts.


## Reproducing the report
Stage 1: build the prior MC cache (LHS 2000 -> behavioural filter -> 500 working ensemble):
    python scripts/run_prior_mc.py --storm-id 33 --n-lhs 2000 --n-workers 9
Stage 2: run the SIR sweep (14 windows x 2 scenarios):
    python scripts/run_sir_sweep.py --storm-id 33 --n-workers 9
Reproduce Appendix C (3-layer vs simplified model):
    python scripts/reproduce_appendix_C.py --storm-id 33 --n-lhs 2000 --n-workers 9
Then open `notebooks/05_results_and_figures.ipynb` to regenerate Figures 3, 4, D, E and Tables B1 from the manuscript.
