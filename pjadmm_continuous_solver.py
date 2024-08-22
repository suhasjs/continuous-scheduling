import cvxpy as cp
import numpy as np
from rich import print as rprint
from pjadmm import eval_augmented_lagrangian, iter_prox_jacobi_admm
from jaxopt import LBFGSB
import time

# solves the Sia linear relaxation using Proximal Jacobi ADMM
# separable form: min sum_i c_i^T x_i,
#            s.t. G sum_i x_i <= g
#                 0 <= x_i <= 1
class PJADMMContinuousSolver:
  # initialize solver
  def __init__(self, params):
    self.G_mat = params['G_mat']
    self.g_vec = params['g_vec']
    self.solver_name = params['solver_name']
    self.block_size = params['block_size']
    self.rtol = params.get('rtol', 1e-6)
    self.solver_params = None

    # problem parameters
    self.num_configs = self.G_mat.shape[1]
    self.num_jobs = 0
    self.num_gpu_types = self.G_mat.shape[0]
    self.num_variables = self.num_jobs * self.num_configs
    self.is_solved = False
    
    # jobname -> idx mapping
    self.job_to_idx = dict()
    # job ID -> block ID mapping
    self.idx_to_block_map = dict()
    # block ID -> [job ID] mapping
    self.block_to_idxs_map = dict()
    # valid IDxs in each block
    self.block_valid_idxs = dict()

    # other params
    lbfgs_max_iters = 15
    jax_xproblem = LBFGSB(fun=eval_augmented_lagrangian, jit=True, unroll=True, implicit_diff=False,
                          maxiter=lbfgs_max_iters, tol=1e-4, stepsize=0.8, history_size=10, use_gamma=True)
    
    ############# PROBLEM STATE #############
    # cost vector for each job in each block
    # shape: num_blocks x block_size x num_configs
    self.vmapped_cost_matrices = None
    ## Variables for k, (k+1) iterations
    # solution for each job in each block
    # shape: num_blocks x block_size x num_configs
    self.vmapped_x_ks = None
    self.vmapped_x_kp1s = None
    # slack variables for each GPU type
    self.s_k = None
    self.s_kp1 = None
    # dual variables for each GPU constraint
    self.u_k = None
    self.u_kp1 = None
    # slack variables for sum-to-1 constraints: one per job
    # shape: num_blocks x block_size
    self.f_k = None
    self.f_kp1 = None
    # dual variables for sum-to-1 constraints: one per job
    # shape: num_blocks x block_size
    self.v_k = None
    self.v_kp1 = None
    # additional variables for kth iteration
    # r_ks = residual for GPU constraints per block
    # shape: num_blocks x num_gpu_types
    self.r_ks = None
    # y_ks = residual for sum-to-1 constraints per block
    # shape: num_blocks x block_size
    self.y_ks = None

  def update_costs(self, cost_updates):
    self.is_solved = False
    for jobname in cost_updates:
      job_idx = self.job_to_idx.get(jobname, None)
      if job_idx is None:
        rprint(f"Job {jobname} not found in job_to_idx mapping")
        continue
      block_id = self.idx_to_block_map[job_idx]
    # TODO :: finish impl

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