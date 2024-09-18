from .policy import AbstractPolicy
from .sia_ilp import SiaILP
import cvxpy as cp
import numpy as np
from rich import print as rprint
import time

class SiaLPRelaxed(SiaILP):
  def __init__(self, num_nodes, ngpus_per_node, policy_options, solver_options):
    # cluster configuration
    self.num_nodes = num_nodes
    self.ngpus_per_node = ngpus_per_node
    self.cluster_ordering = sorted(list(num_nodes.keys()))
    self.num_gputypes = len(self.cluster_ordering)
    self.cluster_gpus = {cluster: num_nodes[cluster] * ngpus_per_node[cluster] for cluster in self.cluster_ordering}
    self.max_ngpus = np.asarray([self.cluster_gpus[cluster] for cluster in self.cluster_ordering])
    self.total_num_gpus = sum(self.cluster_gpus.values())

    # populate configurations
    self.configs, configs_cnstrs = self._get_configurations()
    self.config_cnstr_matrix, self.config_cnstr_vec = configs_cnstrs

    # policy parameters
    self.lambda_no_alloc = policy_options.get('lambda_no_alloc', 1.1)
    self.p_value = policy_options.get('p_value', 0.5)

    # solver options
    self.solver_options = solver_options
    self.solver_name = solver_options.get('solver', 'GLPK')
    self.solver_maps = {'GLPK': cp.GLPK, 'ECOS': cp.ECOS, 'CBC': cp.CBC, "SCS": cp.SCS,
                        "OSQP": cp.OSQP, "PROXQP": cp.PROXQP, "CVXOPT": cp.CVXOPT}
    self.solver_options.pop('solver', None)
    self.warm_start = solver_options.get('warm_start', False)
    self.solver_options.pop('warm_start', None)

    # cluster state
    self.active_jobs = {}
    self.allocations = {}
    self.job_utilities = {}
    self.current_time = 0

    # stats
    self.solver_stats = []

    # rounding functions to convert fractional allocations to integer allocations
    self.round_allocations = self.round_allocations_largest

  def get_save_state(self):
    state = super().get_save_state()
    return state
  
  def load_saved_state(self, state, jobs):
    super().load_saved_state(state, jobs)

  def round_allocations_largest(self, partial_allocations, cluster_free_gpus):
    # allocate the largest possible config to each job
    rounded_allocs = {}
    for jobname, partial_alloc in partial_allocations.items():
      alloced_gpus = 0
      for config, weight in partial_alloc:
        _, ngpus, cluster = config
        if cluster_free_gpus[cluster] >= ngpus:
          rounded_allocs[jobname] = config
          cluster_free_gpus[cluster] -= ngpus
          alloced_gpus = ngpus
          break
      if alloced_gpus == 0:
        rounded_allocs[jobname] = None
    return rounded_allocs

  # override optimize_allocations to use LP relaxation of ILP + rounding
  def optimize_allocations(self):
    # start setup time 
    setup_start = time.time()
    # create inputs to the ILP
    num_jobs = len(self.active_jobs)
    if num_jobs == 0:
      stat = {"time": self.current_time, "num_jobs": 0, "num_vars": 0, "setup_time_ms": 0, "solve_time_ms": 0,
              "solver_status": "optimal", "objective_val": 0}
      self.solver_stats.append(stat)
      return
    num_configs = len(self.configs)
    allocX = cp.Variable((num_jobs, num_configs))
    cost_matrix = np.zeros((num_jobs, num_configs))
    job_ordering = sorted(list(self.active_jobs.keys()))
    for i, jobname in enumerate(job_ordering):
      utilities = np.asarray(self.job_utilities[jobname])
      cost_matrix[i, :] = utilities
    # rprint(f"Cost matrix: {cost_matrix}")
    # rprint(f"Utilities: {self.job_utilities}")
    # raise cost_matrix to the power of p_value
    # add a small value to cost_matrix to avoid division by zero
    if self.p_value < 0:
      cost_matrix[cost_matrix == 0] = 1e-3
    cost_matrix = np.power(cost_matrix, self.p_value)
    cost_matrix[cost_matrix < 1e-2] = -1

    #### Construct optimization problem ####
    objective = cp.sum(cp.multiply(allocX, cost_matrix))
    objective -= self.lambda_no_alloc * cp.sum(1 - cp.sum(allocX, axis=1)) # penalize no allocation
    if self.p_value < 0:
      objective = cp.Minimize(objective)
    else:
      objective = cp.Maximize(objective)
    ## add constraints ##
    # 1. Each job is allocated at-most one configuration
    constraints = []
    for i in range(num_jobs):
      constraints.append(cp.sum(allocX[i, :]) <= 1)
    # 2. Sum of GPUs allocated to all jobs is less than total number of GPUs
    alloced_gpus = cp.matmul(self.config_cnstr_matrix, cp.sum(allocX, axis=0).T)
    constraints.append(alloced_gpus <= self.max_ngpus)
    # 3. X >= 0
    constraints.append(allocX >= 0)

    # Solve problem
    prob = cp.Problem(objective, constraints)
    setup_end = time.time()
    solve_start = time.time()
    cp_solver = self.solver_maps[self.solver_name]
    # print(f"Problem: {prob}")
    if self.warm_start:
      rprint(f"[yellow]Warm starting ILP solver with previous timestep solution...[/yellow]")
      warm_start_allocs = super().get_warm_start_guess(job_ordering)
      allocX.value = warm_start_allocs
    prob.solve(solver=cp_solver, warm_start=self.warm_start, **self.solver_options)
    solve_end = time.time()
    if prob.status != cp.OPTIMAL:
      rprint(f"ERROR :: LP did not converge to optimal solution; returning previous solution")
      rprint(f"Solver status: {prob.status}, exited after {(solve_end - solve_start):.2f} seconds")
      return self.allocations
    else:
      rprint(f"Problem size: {num_jobs}x{num_configs}={num_jobs*num_configs/1000:.1f}k vars, solver time: {(solve_end - setup_start)*1000:.2f} ms, optimal value: {prob.value:.2f}")
      rprint(f"\t Setup: {(setup_end - setup_start)*1000:.2f} ms, Solve: {(solve_end - solve_start)*1000:.2f} ms")
    
    # rprint(f"Solver stats: {stat}")

    # extract allocations
    allocs = allocX.value.round(3)
    alloced_gpus = np.matmul(self.config_cnstr_matrix, np.sum(allocs, axis=0))
    # violations = np.where(alloced_gpus > self.max_ngpus)[0]
    # add a 0.1 buffer for floating point errors
    # assert np.all(alloced_gpus <= (self.max_ngpus + 0.8)), f"GPU allocation exceeds available GPUs: {alloced_gpus} >= {self.max_ngpus}: {alloced_gpus[violations]} > {self.max_ngpus[violations]}"
    # rprint(f"Allocated GPUs: {alloced_gpus}")
    cluster_free_gpus = {cluster: cluster_max_gpus for cluster, cluster_max_gpus in zip(self.cluster_ordering, self.max_ngpus)}
    partial_allocs = {}
    partial_allocs_obj_val = 0
    for i, jobname in enumerate(job_ordering):
      job_alloc = allocs[i, :]
      # no allocation for this job
      if np.sum(job_alloc) == 0:
        self.allocations[jobname] = None
      # some allocation for this job
      elif np.abs(np.sum(job_alloc) - 1) < 0.05:
        # check how many non-zeros in job_alloc
        nnz_job_alloc = np.count_nonzero(job_alloc)
        # exactly one config allocated to this job
        if nnz_job_alloc == 1:
          job_alloc_idx = np.argmax(job_alloc)
          alloc_config = self.configs[job_alloc_idx]
          _, ngpus, cluster = alloc_config
          cluster_free_gpus[cluster] -= ngpus
          self.allocations[jobname] = alloc_config
        else:
          # partial alloc with >1 configs selected (fractionally)
          partial_allocs_obj_val += np.dot(job_alloc, cost_matrix[i, :])
          valid_idxs = np.where(job_alloc > 0)[0]
          config_weights = job_alloc[job_alloc > 0]
          config_choices = [self.configs[idx] for idx in valid_idxs]
          partial_allocs[jobname] = sorted([(x,y) for x,y in zip(config_choices, config_weights)], key=lambda x: x[1], reverse=True)
      else:
        rprint(f"[red]ERROR :: Job {jobname} has invalid allocation: {np.sum(job_alloc)}[/red]")
        self.allocations[jobname] = None
    if len(partial_allocs) > 0:
      rprint(f"[yellow]#Partial allocations: {len(partial_allocs)}/{num_jobs}[/yellow]")
      rounded_allocs = self.round_allocations(partial_allocs, cluster_free_gpus)
      for k in rounded_allocs.keys():
        # rprint(f"\tJob: {k}, partial allocation: {partial_allocs[k]} -> rounded allocation: {rounded_allocs[k]}")
        pass
      self.allocations.update(rounded_allocs)
    
    stat = {"time": self.current_time, "num_jobs": num_jobs, "num_vars": num_jobs*num_configs, 
            "setup_time_ms": (setup_end - setup_start)*1000, "solve_time_ms": (solve_end - solve_start)*1000, 
            "solver_status": str(prob.status), "objective_val": prob.value, "num_partial_allocs": len(partial_allocs),
            "partial_allocs_obj_val": partial_allocs_obj_val}
    self.solver_stats.append(stat)
    
    '''
    rprint(f"Cluster GPU usage:")
    for cluster, ngpus in cluster_alloced_gpus.items():
      assert ngpus <= self.cluster_gpus[cluster], f"GPU type: {cluster} overallocated: allocated={ngpus} > available={self.cluster_gpus[cluster]}"
      rprint(f"\t{cluster} = {ngpus} / {self.cluster_gpus[cluster]} GPUs ({round(ngpus / self.cluster_gpus[cluster] * 100, 2)}%)")
    '''