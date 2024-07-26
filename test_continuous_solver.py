import pickle
from argparse import ArgumentParser
from problem_utils import read_sia_problems
from rich import print as rprint

from cvxpy_continuous_solver import SiaCvxpyContinuousSolver

# instantiate arg parser
parser = ArgumentParser()
parser.add_argument("--program-dump", type=str, help="path to pkl containing program dumps", required=True)
parser.add_argument("--solver", type=str, help="solver to use for CVXPY", choices=['ECOS', 'SCS', 'GLPK', 'PROXQP', 'PIQP'], default='GLPK')
parser.add_argument("--rtol", type=float, help="relative tolerance for solver", default=1e-4)
parser.add_argument("--start-program-id", type=int, help="where to start solving from", default=0)
parser.add_argument("--num-programs", type=int, help="how many programs to solve (in a continuous manner) [default= None: solve all problems AFTER start-program-id]", default=None)
parser.add_argument("--output-logs", type=str, default=None, help="path to output per-problem logs (solution, solver stats) to (in a pkl format)")

# parse args
args = parser.parse_args()
dump_file = args.program_dump
start_idx = args.start_program_id
num_problems = args.num_programs
output_logs = args.output_logs
rtol = args.rtol

# read problems from disk
problems = read_sia_problems(dump_file, start_idx, num_problems)
solver_params = {'G_mat' : problems['G_mat'], 'g_vec' : problems['g_vec'], 'solver_name' : args.solver}
solver_params['rtol'] = rtol

# instantiate solver
solver = SiaCvxpyContinuousSolver(solver_params)

# run through all problems
solutions = []
cur_jobs = set()
for idx, cost_dict in enumerate(problems['costs']):
  # get changes from previous problem
  new_jobset = set(cost_dict.keys())
  added_jobs = new_jobset - cur_jobs
  removed_jobs = cur_jobs - new_jobset
  updated_jobs = cur_jobs.intersection(new_jobset)

  # update current jobset
  cur_jobs = new_jobset

  # update costs
  updated_costs = {jobname: cost_dict[jobname] for jobname in updated_jobs}
  solver.update_costs(updated_costs)

  # update jobs
  added_jobs = {jobname: cost_dict[jobname] for jobname in added_jobs}
  removed_jobs = list(removed_jobs)
  solver.update_jobs(added_jobs, removed_jobs)

  # solve problem
  ret = solver.solve()
  print(f"Problem {idx} solved :: return={ret}")

  # get solution
  sol = solver.get_solution()
  solutions.append((ret, sol))

# output logs containing solutions
if output_logs is not None:
  with open(output_logs, 'wb') as f:
    pickle.dump(solutions, f)