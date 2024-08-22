from policy import AbstractPolicy
import cvxpy as cp
import numpy as np
from rich import print as rprint
import time

class SiaILP(AbstractPolicy):
  def __init__(self, num_nodes, ngpus_per_node, policy_options, solver_options):
    # cluster configuration
    self.num_nodes = num_nodes
    self.ngpus_per_node = ngpus_per_node
    self.cluster_ordering = sorted(list(num_nodes.keys()))
    self.num_gputypes = len(self.cluster_ordering)
    self.cluster_gpus = {cluster: num_nodes[cluster] * ngpus_per_node for cluster in self.cluster_ordering}
    self.total_num_gpus = sum(self.cluster_gpus.values())

    # populate configurations
    self.configs, configs_cnstrs = self.__get_configurations()
    self.config_cnstr_matrix, self.config_cnstr_vec = configs_cnstrs

    # policy parameters
    self.lambda_no_alloc = policy_options.get('lambda_no_alloc', 1.1)
    self.p_value = policy_options.get('p_value', 0.5)

    # solver options
    self.solver_options = solver_options
    self.solver_name = solver_options.get('solver', 'GLPK_MI')
    self.solver_maps = {'GLPK_MI': cp.GLPK_MI, 'ECOS_BB': cp.ECOS_BB, 'CBC_MI': cp.CBC}
    self.solver_options.pop('solver', None)

    # cluster state
    self.active_jobs = {}
    self.allocations = {}
    self.job_utilities = {}
    self.current_time = 0
    pass

  # Configurations are tuples of (num_nodes, num_gpus, cluster)
  # Each configuration is a candidate allocation in a given heterogeneous GPU cluster
  def __get_configurations(self):
    configs_to_idx = {}
    configs = []
    config_ngpus = {}
    idx_to_config = {}
    idx = 0
    for cluster in self.cluster_ordering:
      ngpus = 1
      nnodes = 1
      max_ngpus = self.cluster_gpus[cluster]
      config_ngpus[cluster] = []
      while nnodes <= self.num_nodes[cluster]:
        new_config = (nnodes, ngpus, cluster)
        configs_to_idx[new_config] = idx
        idx_to_config[idx] = new_config
        config_ngpus[cluster].append(ngpus)
        configs.append(new_config)
        idx += 1
        # double ngpus for allocs within node
        if ngpus < self.ngpus_per_node[cluster]:
          ngpus *= 2
        else:
        # grab whole nodes for distributed allocs
          nnodes += 1
          ngpus = nnodes * self.ngpus_per_node[cluster]
      rprint(f"GPU type: {cluster}, max_ngpus: {max_ngpus}, configs: {configs}")
    # construct constraint matrix
    num_configs = len(configs)
    config_cnstr_matrix = np.zeros(shape=(self.num_gputypes, num_configs))
    start_idx = 0
    max_ngpus = np.asarray([self.cluster_gpus[cluster] for cluster in self.cluster_ordering])
    for i, cluster in enumerate(self.cluster_ordering):
      cluster_nconfigs = len(config_ngpus[cluster])
      config_cnstr_matrix[i, start_idx : start_idx + cluster_nconfigs] = np.asarray(config_ngpus[cluster])
    return configs, (config_cnstr_matrix, max_ngpus)
  
  def update_failed_nodes(self, failed_nodes):
    pass

  def update_job_utilities(self, new_job_utilities):
    for jobname, utilities in new_job_utilities.items():
      self.job_utilities[jobname] = utilities

  def add_new_jobs(self, new_jobs):
    for jobname, job in new_jobs.items():
      if jobname in self.active_jobs:
        rprint(f"ERROR :: Job {jobname} already exists in active_jobs")
        continue
      self.active_jobs[jobname] = job
      self.allocations[jobname] = None
      self.job_utilities[jobname] = None

  def remove_completed_jobs(self, completed_jobs):
    for jobname in completed_jobs:
      if jobname not in self.active_jobs:
        continue
      else:
        self.active_jobs.pop(jobname)
        self.allocations.pop(jobname)
        self.job_utilities.pop(jobname)


  def step(self, seconds):
    self.time += seconds

  def optimize_allocations(self):
    # start setup time 
    setup_start = time.time()
    # create inputs to the ILP
    num_jobs = len(self.active_jobs)
    num_configs = len(self.configs)
    allocX = cp.Variable((num_jobs, num_configs), boolean=True)
    cost_matrix = np.zeros((num_jobs, num_configs))
    job_ordering = sorted(list(self.active_jobs.keys()))
    for i, jobname in enumerate(job_ordering):
      utilities = np.asarray(self.job_utilities[jobname])
      cost_matrix[i, :] = utilities
    # raise cost_matrix to the power of p_value
    # add a small value to cost_matrix to avoid division by zero
    if self.p_value < 0:
      cost_matrix[cost_matrix == 0] = 1e-3
    cost_matrix = np.pow(cost_matrix, self.p_value)

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
      constraints.append(cp.sum(allocX[i, :]) == 1)
    # 2. Sum of GPUs allocated to all jobs is less than total number of GPUs
    alloced_gpus = cp.matmul(cp.sum(allocX, axis=0), self.config_cnstr_matrix)
    constraints.append(cp.sum(alloced_gpus) <= self.total_num_gpus)
    # 3. X >= 0
    constraints.append(allocX >= 0)

    # Solve problem
    prob = cp.Problem(objective, constraints)
    setup_end = time.time()
    solve_start = time.time()
    cp_solver = self.solver_maps[self.solver_name]
    prob.solve(solver=cp_solver, verbose=False, **self.solver_options)
    solve_end = time.time()
    if prob.status != cp.OPTIMAL:
      rprint(f"ERROR :: ILP did not converge to optimal solution")
      rprint(f"Solver status: {prob.status}")
    else:
      rprint(f"Solver finished in {(solve_end - setup_start):.2f} seconds")
      rprint(f"\t Setup: {(setup_end - setup_start):.2f} seconds, Solve: {(solve_end - solve_start):.2f} seconds")

    # extract allocations
    allocs = allocX.value
    for i, jobname in enumerate(job_ordering):
      job_alloc = allocs[i, :]
      if np.sum(job_alloc) == 0:
        self.allocations[jobname] = None
      elif np.sum(job_alloc) == 1:
        job_alloc_idx = np.argmax(job_alloc)
        alloc_config = self.configs[job_alloc_idx]
        self.allocations[jobname] = alloc_config
      else:
        rprint(f"ERROR :: Job {jobname} has multiple allocations")
        self.allocations[jobname] = None
    
  def get_allocations(self):
    return self.allocations