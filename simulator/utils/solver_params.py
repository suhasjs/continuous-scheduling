import cvxpy as cp

# time_limit: time limit for solver (in seconds, default=1200s/20min)
# rtol: relative tolerance for solver (default: 1e-4)
# mipgap: sub-optimality of integer solution for MIP solver (default: 1e-3/0.1%)
def get_solver_params(solver_name, time_limit=None, rtol=None, mipgap=None):
  if mipgap is None:
    mipgap = 1e-3
  if rtol is None:
    rtol = 1e-4
  if time_limit is None:
    time_limit = 1200
  # for any solvers that can use multiple threads
  num_threads = 8
  if solver_name == 'GLPK_MI':
    options = {'msg_lev': 'GLP_MSG_OFF', 'tm_lim' : time_limit*1000, 'mip_gap' : mipgap}
  elif solver_name == "GLPK":
    options = {'msg_lev': 'GLP_MSG_OFF', 'tm_lim' : time_limit*1000}
  elif solver_name == "CBC_MI":
    options = {'maximumSeconds': time_limit, 'numberThreads': num_threads, 
               'allowablePercentageGap': rtol*1e2}
  elif solver_name == "CBC":
    options = {'maximumSeconds': time_limit, 'numberThreads': num_threads, 
               'allowablePercentageGap': rtol*1e2}
  elif solver_name == "ECOS_BB":
    options = {'reltol': rtol}
  elif solver_name == "OSQP":
    options = {'eps_rel': rtol}
  elif solver_name == "SCS":
    options = {'eps': rtol, 'use_indirect': False}
  elif solver_name == "PROXQP":
    options = {'eps_rel': rtol, 'backend': 'sparse'}
  elif solver_name == "PIQP":
    options = {'eps_rel': rtol}
  else:
    raise ValueError(f"Solver {solver_name} not supported")
  return options