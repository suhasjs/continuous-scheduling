import cvxpy as cp
import numpy as np
from rich import print as rprint
import time

# solves the Sia linear relaxation using cvxpy
# separable form: min sum_i c_i^T x_i,
#            s.t. G sum_i x_i <= g
#                 0 <= x_i <= 1
# standard form: min c^T x, 
#           s.t. Ax <= b
class SiaCvxpyContinuousSolver:
  # initialize solver
  def __init__(self, params):
    self.G_mat = params['G_mat']
    self.g_vec = params['g_vec']
    self.solver_name = params['solver_name']
    self.rtol = params.get('rtol', 1e-6)
    self.solver_map = {'GLPK': cp.GLPK, 'CBC': cp.CBC, 'SCS': cp.SCS, 'PROXQP': cp.PROXQP, 'PIQP' : cp.PIQP}
    self.solver_params = {'GLPK': {'verbose':False, 
                                   'opts':{'msg_lev': 'GLP_MSG_OFF', 'tol_bnd': self.rtol}},
                          'CBC': {'verbose':False, 'allowableGap': self.rtol*100},
                          'SCS': {'verbose':False, 'eps': self.rtol},
                          'PROXQP': {'verbose':False, 'eps_rel': self.rtol},
                          'PIQP': {'verbose':False, 'eps_rel': self.rtol}}

    # problem parameters
    self.num_configs = self.G_mat.shape[1]
    self.num_jobs = 0
    self.num_gpu_types = self.G_mat.shape[0]
    self.num_variables = self.num_jobs * self.num_configs
    
    # jobname -> idx mapping
    self.job_to_idx = dict()
    
    # state for the problem
    self.cost_vector = None
    self.A_mat = None
    self.b_vec = None
    self.variables = None
    self.problem = None
    self.constraints = None
    self.solution = None
    self.results = None
    self.is_solved = False

  def __reconstruct_standard_form(self):
    rprint(f"Reconstructing standard form for {self.num_jobs} jobs")
    self.num_variables = self.num_jobs * self.num_configs
    num_rows = self.num_gpu_types + self.num_jobs + (2 * self.num_variables)
    num_cols = self.num_jobs * self.num_configs
    new_A_mat = np.zeros(shape=(num_rows, num_cols))
    new_b_vec = np.zeros(shape=(num_rows, ))
    # Gx <= g constraints
    new_A_mat[:self.num_gpu_types, :] = np.tile(self.G_mat, (1, self.num_jobs))
    new_b_vec[:self.num_gpu_types] = self.g_vec
    # sum-to-1 constraints
    start_idx = self.num_gpu_types
    for job_idx in range(self.num_jobs):
      new_A_mat[start_idx, (job_idx*self.num_configs) : (job_idx + 1)*self.num_configs] = 1
      new_b_vec[start_idx] = 1
      start_idx += 1
    # 0 <= x_i constraints ==> represented as (-x_i <= 0)
    new_A_mat[start_idx: start_idx + self.num_variables, :] = -np.eye(self.num_variables)
    new_b_vec[start_idx: start_idx + self.num_variables] = 0
    start_idx += self.num_variables
    # x_i <= 1 constraints
    new_A_mat[start_idx: start_idx + self.num_variables, :] = np.eye(self.num_variables)
    new_b_vec[start_idx: start_idx + self.num_variables] = 1
    # update the standard form matrices
    self.A_mat = new_A_mat
    self.b_vec = new_b_vec
    # reconstruct the CVXPY problem
    self.variables = cp.Variable(self.num_variables)
    self.variables.value = self.solution
    self.constraints = [self.A_mat @ self.variables <= self.b_vec]
    self.problem = cp.Problem(cp.Minimize(self.cost_vector @ self.variables), self.constraints)

  def update_costs(self, cost_updates):
    self.is_solved = False
    for jobname in cost_updates:
      job_idx = self.job_to_idx.get(jobname, None)
      if job_idx is None:
        rprint(f"Job {jobname} not found in job_to_idx mapping")
        continue
      cost_vec_start_idx = job_idx * self.num_configs
      self.cost_vector[cost_vec_start_idx: cost_vec_start_idx+self.num_configs] = cost_updates[jobname]

  # added_jobs = {new_jobname: cost_vector}
  # removed_jobs = [jobname]
  def update_jobs(self, added_jobs, removed_jobs):
    # no changes to set of jobs
    if len(added_jobs) == 0 and len(removed_jobs) == 0:
      rprint(f"No changes to job list")
      return
    # needs to be solved again
    self.is_solved = False

    # remove all removed jobs
    removed_jobs = [x for x in removed_jobs if x in self.job_to_idx]
    for jobname in removed_jobs:
      # remove job from job_to_idx mapping
      self.job_to_idx.pop(jobname)
    
    # reassign indices for jobs
    new_num_jobs = self.num_jobs + len(added_jobs) - len(removed_jobs)
    must_reconstruct_standard_form = (new_num_jobs != self.num_jobs)
    new_num_variables = new_num_jobs * self.num_configs
    new_cost_vector = np.zeros(shape=(new_num_variables, ))
    new_solution_vector = np.zeros(shape=(new_num_variables, ))

    new_job_to_idx = dict()
    cur_idx = 0
    for jobname, idx in self.job_to_idx.items():
      new_job_to_idx[jobname] = cur_idx
      # copy over cost vector
      src_start_idx = idx * self.num_configs
      dest_start_idx = cur_idx * self.num_configs
      new_cost_vector[dest_start_idx: dest_start_idx+self.num_configs] = self.cost_vector[src_start_idx: src_start_idx+self.num_configs]
      # copy over solution vector
      new_solution_vector[dest_start_idx: dest_start_idx+self.num_configs] = self.solution[src_start_idx: src_start_idx+self.num_configs]
      # bump up index
      cur_idx += 1

    # add new jobs
    for jobname, cost_vector in added_jobs.items():
      new_job_to_idx[jobname] = cur_idx
      dest_start_idx = cur_idx * self.num_configs
      new_cost_vector[dest_start_idx: dest_start_idx+self.num_configs] = cost_vector
      # bump up index
      cur_idx += 1

    # update job_to_idx mapping
    self.job_to_idx = new_job_to_idx
    # update cost vector, solution vectors (initialized to 0 for new jobs)
    self.cost_vector = new_cost_vector
    self.solution = new_solution_vector

    # update number of jobs and variables; reconstruct standard form if needed
    self.num_jobs = new_num_jobs
    self.num_variables = new_num_variables
    if must_reconstruct_standard_form:
      self.__reconstruct_standard_form()

  # solve the existing problem
  def solve(self):
    solver = self.solver_map.get(self.solver_name, None)
    addnl_params = self.solver_params.get(self.solver_name, {})
    # return if already solved
    if self.is_solved:
      return self.results
    else:
      start_t = time.time()
      self.problem.solve(solver=solver, **addnl_params)
      end_t = time.time()
      self.is_solved = True
      self.results = {'status': self.problem.status, 
                      'optimal_value': self.problem.value,
                      'solver_time_ms': (end_t - start_t)*1000.0}
      self.solution = self.variables.value
      return self.results

  # returns optimal solution to last solved problem
  def get_solution(self, jobname=None):
    if not self.is_solved:
      rprint(f"Problem not solved yet")
      return None
    chosen_jobs = list(self.job_to_idx.keys()) if jobname is None else [jobname]
    solutions = dict()
    for jobname in chosen_jobs:
      job_idx = self.job_to_idx.get(jobname, None)
      start_idx = job_idx * self.num_configs
      solutions[jobname] = self.solution[start_idx: start_idx+self.num_configs]
    return solutions