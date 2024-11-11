# How to use the simulator
Example command line: 
```
python simulator.py --job-trace workloads/40kjobs_24hrs_pai/workload-1.csv --round-duration=20 --solver-name=CBC --policy=sia-lp-relaxed --solver-rtol=1e-2 --cluster-scale=40 --output-log=/tmp/workload-2.pkl --warm-start-solver --verbose-solver
```
- `--job-trace` --> Choose a trace file from `workloads/*/*`. Folder name inside `workloads` is the name of the workload and describes how many jobs are submitted in the trace over a period of time
- `--round-duration` --> Duration of each round in seconds. The simulator will invoke the solver every `round-duration` seconds
-- `--policy` --> Choose a policy. `sia-ilp` runs the ILP version of the policy and `sia-lp-relaxed` runs the LP relaxation of the Sia ILP policy. `sia-lp-relaxed-pjadmm` is our version of the LP relaxation with the PJADMM solver
- `--solver-name` --> Choose a solver for the given policy. Look inside `utils/solver_params.py` for available solvers. For the Sia ILP policy, use solvers with a `_MI` suffix (MI = Mixed Integer), and for the LP relaxation, use any solver (except PJADMM). For `sia-lp-relaxed-pjadmm` policy, set `--solver-name=PJADMM`
- `--solver-rtol` --> Relative tolerance for the solver. Recommend setting this to `1e-2` or `1e-3`. ADMM-based solvers may not converge even with `1e-3` tolerance (some constraints will be violated in the returned solution)
- `--cluster-scale` --> Scale the cluster size by this factor. Consider setting `--cluster-scale=2` for `2kjobs_24hrs_pai`, and `--cluster-scale=40` for `40kjobs_24hrs_pai` workloads
- `--output-log` --> Output log file (pkl format) to save the results of the simulation throughout execution. Simulator automatically checkpoints the state of simulation every `--checkpoint-frequency=10` rounds. Can continue the simulation from the last checkpoint by setting the `--load-checkpoint` flag
- `--warm-start-solver` --> Warm start the solver with the previous solution. Might help some solvers
- `--verbose-solver` --> Control the verbosity of the solver. Set this flag to see the solver intermediate output
- `--simulator-timeout` --> Stops simulation after a certain number of rounds. Don't need to set this if you want to run the trace to completion
- `--debug` --> Set this flag to run the simulator in debug mode. This will print additional information about the simulation and will stop and wait for user-input after each round. Use to inspect round-by-round behavior of the simulator
- `--disable-status` --> Disables printing detailed per-job information in each round. Instead, will only print aggregate info about number of jobs and resources in the cluster. Use to suppress clutter for large experiments

For other options, run `python simulator.py --help`.