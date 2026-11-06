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
