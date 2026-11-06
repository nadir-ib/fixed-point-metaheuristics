# Fractional-order (finite-memory) metaheuristics: code and data

Code, experiment scripts and result data for the article

> N. İbrahimoğlu, M. U. Demirezen, A. Ghaffari, "A Banach Fixed-Point Framework for Fractional-Order Iterative Schemes
> in Population-Based Optimization", revised manuscript submitted to *The Journal of Supercomputing* (2026).

Every table, figure and number of the revised manuscript can be regenerated with the commands below.

## Layout
- `src/algorithms/`  PSO, Jaya, WFO, ACOR and their fractional-order variants (FO-PSO, FO-JAYA, FO-WFO, FO-ACOR).
- `src/fractional/`  Grünwald–Letnikov coefficients.  `src/benchmarks/functions.py`  Sphere, Rastrigin, Ackley, Rosenbrock, Griewank.
- `experiments/`     scripts that produce the results (see below).  `math/`  numerical checks of Section 4.
- `results/`         the data files and text outputs used in the manuscript.  `CHANGES.md`  code changes made during the revision.

## Reproducing the results (run every command from the repository root; Python 3.9+, `pip install -r requirements.txt`)
| Manuscript item | Command | Output |
|---|---|---|
| Tables 5-9, 11-13; Sections 5.2.7-5.2.10 | `OMP_NUM_THREADS=1 python3 experiments/exp18_r1_benchmark.py` then `python3 experiments/exp18_tables.py <npz> results/data/exp21_timing.json` and `python3 experiments/exp18_r1_analysis.py <npz>` | `results/data/exp18_r1_*.npz`, `results/tables/R1_tables.*`, `results/exp18_r1_analysis.txt` |
| Conventional domains (Section 5.2.10) | `python3 experiments/exp18_r1_benchmark.py --suites convdomain` | `results/data/exp18_r1_convdomain_*.npz` |
| Centre-bias probe (Section 5.2.10) | `python3 experiments/exp19_centre_bias_probe.py` | `results/exp19_centre_bias_probe.txt` |
| FO-PSO weight ablation | `python3 experiments/exp20_fopso_weight_ablation.py` | `results/exp20_fopso_weight_ablation.txt` |
| Table 10, Section 6.2 (run serially on an idle machine) | `python3 experiments/exp21_timing.py` | `results/exp21_timing.txt` |
| Section 5.3, Table 14, Observation 1 | `python3 experiments/exp22_alpha_sweep.py` | `results/exp22_alpha_sweep.txt` |
| Other numbers in Sections 5-7 | `python3 experiments/exp23_text_numbers.py` | `results/exp23_text_numbers.txt` |
| FO-PSO ablation summary (Sec. 4.1.2), function-block Friedman (Sec. 5.2.7, 5.2.10), WFO equal-component shift (Sec. 5.2.10) | `python3 experiments/exp24_r1_extra_checks.py` | `results/exp24_r1_extra_checks.txt` |
| Figures 1 and 2 | `python3 experiments/generate_figure1_gl_weights.py`, `python3 experiments/generate_figure2_convergence_R1.py <npz>` | `results/figures/` |
| Section 4 constants (Lemmas A-C, M; Theorems 2-5) | `python3 math/verify_math.py`; `cd math && python3 pso_mean_square_check.py && python3 pso_thresholds.py && python3 mean_coupling_lower_bounds.py && python3 memory_strength_shape.py` | `math/*_output.txt` |
| Theorem 2 checks | `python3 experiments/exp16_pso_validation.py` | `results/exp16_pso_validation_R1_output.txt` |

Protocol: 30 runs per setting with seed 42 + run, 500 iterations, population 30, box [-100,100]^n (Table 4 of the
manuscript). Numbers may differ in the last digits across machines and NumPy versions (floating-point summation order).
