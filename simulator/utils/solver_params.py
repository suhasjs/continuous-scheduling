import cvxpy as cp

# time_limit: time limit for solver (in seconds, default=1200s/20min)
# rtol: relative tolerance for solver (default: 1e-4)
# mipgap: sub-optimality of integer solution for MIP solver (default: 1e-3/0.1%)
def get_solver_params(solver_name, time_limit=None, rtol=None, mipgap=None, verbose=False):
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
  elif solver_name == "SCIPY_MI":
    options = {'scipy_options': {'method': 'highs', 'disp': verbose, 'tol': rtol}}
  elif solver_name == "SCIPY_DS":
    options = {'scipy_options': {'method': 'highs-ds', 'disp': verbose, 'tol': rtol}}
  elif solver_name == "SCIPY_IPM":
    options = {'scipy_options': {'method': 'highs-ipm', 'disp': verbose, 'tol': rtol}}
  elif solver_name == "OSQP":
    # Known good parameters for OSQP: https://github.com/cvxpy/cvxpy/issues/898#issuecomment-589861097
    options = {'eps_rel': rtol, 'max_iter': 100000, 'rho': 1, 'alpha': 1, 
               'adaptive_rho': False, 'linsys_solver': 'mkl pardiso'}
  elif solver_name == "SCS":
    options = {'eps_rel': rtol, 'use_indirect': False}
  elif solver_name == "CVXOPT":
    options = {'reltol': rtol, 'kktsolver': 'ldl2', 'feastol': 1e-3}
  elif solver_name == "PROXQP":
    options = {'eps_rel': rtol, 'backend': 'sparse', 'rho': 1e-5, 'mu_in': 1e-1}
  elif solver_name == "PIQP":
    options = {'eps_rel': rtol}
  elif solver_name == "PJADMM":
    options = {'tol': rtol}
  elif solver_name == "ALCD":
    options = {'tol': rtol}
  else:
    raise ValueError(f"Solver {solver_name} not supported")
  return options