# Code changes made during the revision (version used for all revised results)

C001 ACOR and FO-ACOR: the archive is held fixed during an iteration; all ants sample from it and the archive is then
updated once (merge the k archive members with the new solutions, sort, keep the best k), as in ACO_R. In the earlier
version the archive was re-sorted after every ant while the standard deviations were computed once per iteration, so the
kernel mean and the standard deviation could belong to different archive members. An unused random draw was removed and
the standard deviations are computed with vectorised code (identical values).

C002 JAYA and FO-JAYA: the population is evaluated once at the start; afterwards the stored fitness values are kept
current by the greedy selection, giving 15,030 objective evaluations per run (the earlier version re-evaluated the whole
population every iteration, 30,000 evaluations). The best solution is refreshed at the end of every iteration. The
search dynamics are bit-identical to the earlier version.

C003 experiments/exp16_pso_validation.py: the mean-square rate is computed from the exact second-moment recursion
instead of a Monte Carlo slope (which is biased low for heavy-tailed products); the almost-sure rate is reported separately.

C004 FO-PSO: optional `memory_weights` ('uniform' = default, as published; 'gl_def6', 'gl_abs') for the weight ablation.
The default is bit-identical to the earlier version.

C005, C006: figure scripts (Figure 1 with the symbol omega_j and j = 0..M-1; Figure 2 from the revised benchmark).

Note on the recorded hashes: the metadata of `results/data/exp18_r1_20261106_090957.npz` stores the sha256 of each source
file at the time of the run. The value for `src/algorithms/fo_pso.py` differs from the released file because C004 (the
optional `memory_weights` argument and its docstring) was added after the benchmark run; the default path used in the
benchmark is bit-identical: the released code reproduces the stored best values and convergence curves of FO-PSO(0.9)
exactly (checked on 07 Nov 2026 for the canonical and the shifted and rotated suites, n = 10, all five functions,
runs 0-2).

C008 experiments/generate_figure2_convergence_R1.py: the panel titles of the top row read "standard form" (the earlier
"optimum at the origin" was wrong for Rosenbrock, whose minimizer is (1, ..., 1)). Data and curves are unchanged.

C009 new scripts, no change to `src/`: `experiments/exp25_vectorised_timing.py` (end-to-end runtime of FO-JAYA and FO-WFO
with the memory terms computed once per iteration; the vectorised variants are subclasses defined in the script and are
checked to be bit-identical to `src/` in every timed run) and `experiments/exp26_r1_council2_checks.py` (checks quoted in
Sections 4.1.1, 4.3, 5.1.3, 5.2.8 and 5.2.10).
