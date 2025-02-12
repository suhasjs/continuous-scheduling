from .policy import AbstractPolicy
from .sia_ilp import SiaILP
from .policy_utils import round_allocations_largest
import pylpsparse as lps
import cvxpy as cp
import numpy as np
from rich import print as rprint
import time


class SiaLPRelaxedALCD(SiaILP):
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
    self.solver_tol = self.solver_options.get('tol', 1e-2)
    lpcfg = lps.LP_Param()
    lpcfg.solve_from_dual = self.solver_options.get("solve_from_dual", True)
    lpcfg.eta = self.solver_options.get("alcd_eta", 1)
    lpcfg.verbose = self.solver_options.get("verbose", False)
    lpcfg.tol = self.solver_tol
    self.verbose_solver = lpcfg.verbose
    lpcfg.tol_sub = self.solver_options.get("alcd_tol_sub", 1e-1)
    lpcfg.tol_trans = self.solver_options.get("alcd_tol_trans", 0.1)
    lpcfg.use_CG = not self.solver_options.get("disable_CG", False)
    lpcfg.inner_max_iter = self.solver_options.get("alcd_inner_max_iters", 1)
    self.lpcfg = lpcfg
    self.warm_start = solver_options.get('warm_start', False)
    self.warm_start_duals = solver_options.get('warm_start_duals', True)
    self.solver_options.pop('warm_start', None)
    self.differential_update = solver_options.get('differential_update', False)
    self.alcd_job_cnstr_reweight = solver_options.get('alcd_job_cnstr_reweight', 1.0)
    self.lpobj = None

    # cluster state
    self.active_jobs = {}
    self.job_ordering = []
    self.allocations = {}
    self.raw_allocations = {} # store raw fractional allocations output by ALCD solver
    self.gpu_duals = {}
    self.job_duals = {}
    self.job_utilities = {}
    self.current_time = 0

    # stats
    self.solver_stats = []

    # rounding functions to convert fractional allocations to integer allocations
    self.round_allocations = round_allocations_largest

    # record programs for offline playback (external)
    # use standard form:
    # min c^T x
    # s.t. Ax <= b
    #      x >= 0
    # OLD -> store: (A, b, c, x_opt, obj_val, status, time, num_jobs, num_configs, job ordering)
    # NEW -> store: (c, x_opt, obj_val, status, time, num_jobs, job_ordering)
    #     -> meta: (self.config_cnstr_matrix, self.max_ngpus, self.cluster_ordering)
    self.record_programs = solver_options.get('record_programs', False)
    self.recorded_programs = []
  
  def get_program_dump(self):
    program_meta = {"config_cnstr_matrix": self.config_cnstr_matrix, 
                    "max_ngpus": self.max_ngpus, 
                    "cluster_ordering": self.cluster_ordering
    }
    dump = {
      "meta" : program_meta,
      "programs": self.recorded_programs,
      "solver_stats": self.solver_stats
    }
    return dump

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
  
  # Ensures consistent ordering across time steps
  # IF jobA and jobB are present in timesteps t and (t+1), then:
  #    index(jobA) < index(jobB) at time t => index(jobA) < index(jobB) at time (t+1)
  # WARNING:: This is required because we do not permute constraint matrix rows in LP solver
  def get_job_ordering(self):
    new_job_set = set(self.active_jobs.keys())
    old_job_ordering = self.job_ordering
    new_job_ordering = []
    # preserve ordering of old jobs --> don't shuffle them around
    for jobname in old_job_ordering:
      if jobname in new_job_set:
        new_job_ordering.append(jobname)
        new_job_set.remove(jobname)
    # add new jobs to the end of the ordering in sorted order
    new_job_ordering.extend(sorted(list(new_job_set)))
    return new_job_ordering
  
  # override get_warm_start_guess to use raw fractional allocations
  def get_warm_start_guess(self, job_ordering):
    num_jobs, num_configs = len(job_ordering), len(self.configs)
    warm_start_allocs = np.zeros((num_jobs, num_configs))
    for i, jobname in enumerate(job_ordering):
      # get current allocation
      cur_alloc = self.raw_allocations.get(jobname, None)
      # zero allocation if current allocation is None
      if cur_alloc is None:
        continue
      else:
        warm_start_allocs[i, :] = cur_alloc

    warm_start_dual = np.zeros(num_jobs + self.num_gputypes)
    if self.warm_start_duals:
      for i, cluster in enumerate(self.cluster_ordering):
        warm_start_dual[i] = self.gpu_duals.get(cluster, 0)
      for i, jobname in enumerate(job_ordering):
        warm_start_dual[self.num_gputypes + i] = self.job_duals.get(jobname, 0)
    return warm_start_allocs, warm_start_dual
  
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
    job_ordering = self.get_job_ordering()
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
    stdc = -1 * (cost_matrix.flatten() + self.lambda_no_alloc)

    ### Construct ALCD problem
    program_start_time = time.time()
    program = {
      "num_jobs" : num_jobs, "num_configs" : num_configs, "c" : stdc,
      "job_ordering" : job_ordering, "time" : self.current_time,
      "cnstrA" : self.config_cnstr_matrix, "cnstrb" : self.config_cnstr_vec,
      "is_compressed" : True, "job_cnstr_reweight" : self.alcd_job_cnstr_reweight,
    }
    if self.differential_update:
      # update existing LP object if exists; else create new LP object
      if self.lpobj is None:
        self.lpobj = lps.SiaSparseLPFormat(program)
      else:
        self.lpobj.update_program(program, verbose=False)
      lpobj = self.lpobj
    else:
      lpobj = lps.SiaSparseLPFormat(program)
    primal_args = lpobj.get_primal_alcd_format()
    dual_lpargs = lpobj.get_dual_alcd_format()
    program_load_time = time.time() - program_start_time

    A, b, c, nb, nf, m, me = primal_args
    At = dual_lpargs[0]
    assert nf == 0, "Free variables not allowed"
    assert me == 0, "Equality constraints not allowed"
    x0 = np.zeros(len(c))
    # w0 = np.zeros(len(b))
    w0 = -b
    init_start_time = time.time()
    if self.lpcfg.solve_from_dual is False:
      h2jj = np.zeros(nb + nf)
      hjj_ubound = np.zeros(nb + nf)
      lps.init_state(x0, w0, h2jj, hjj_ubound, nb, nf, m, me, A, b, c, self.lpcfg.eta) 
    else:
      h2jj = np.zeros(m + me)
      hjj_ubound = np.zeros(m + me)
      lps.init_state(w0, x0, h2jj, hjj_ubound, m, me, nb, nf, At, c, b, self.lpcfg.eta)
    # warm-start x0 ?
    # if self.warm_start:
    #   x0[:] = self.get_warm_start_guess(job_ordering).flatten()
    if self.warm_start:
      x0_ws, w0_ws = self.get_warm_start_guess(job_ordering)
      x0[:] = x0_ws.flatten()
      if self.warm_start_duals:
        w0[:] = w0_ws
    x0copy = np.copy(x0)
    w0copy = np.copy(w0)
    program_init_time = time.time() - init_start_time

    ### 2. Solve the ALCD problem
    lpinfo = lps.LP_Info()
    # make a copy of LP_Param object since it is modified in-place inside cpp solver code
    new_lpcfg = lps.LP_Param()
    new_lpcfg.solve_from_dual = self.lpcfg.solve_from_dual
    new_lpcfg.eta = self.lpcfg.eta
    new_lpcfg.verbose = self.lpcfg.verbose
    new_lpcfg.tol = self.lpcfg.tol
    new_lpcfg.tol_sub = self.lpcfg.tol_sub
    new_lpcfg.tol_trans = self.lpcfg.tol_trans
    new_lpcfg.use_CG = self.lpcfg.use_CG
    new_lpcfg.inner_max_iter = self.lpcfg.inner_max_iter

    solve_start_time = time.time()
    rprint(f"[yellow]Solving ALCD problem with eta={new_lpcfg.eta}[/yellow]")
    lps.solve_alcd_corrector(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, new_lpcfg, lpinfo)
    program_solve_time = time.time() - solve_start_time
    total_time = time.time() - program_start_time

    ### 3. Extract solution
    allocX = x0.reshape((num_jobs, num_configs))
    # print w in structured_form
    self.gpu_duals = {cluster: w for cluster, w in zip(self.cluster_ordering, w0[:self.num_gputypes])}
    self.job_duals = {jobname: w for jobname, w in zip(job_ordering, w0[self.num_gputypes:])}
    # store raw fractional allocations for warm-start
    self.raw_allocations.clear()
    for i, jobname in enumerate(job_ordering):
      self.raw_allocations[jobname] = np.array(allocX[i, :])
    ret_info = lps.lpinfo_to_dict(lpinfo)
    program_status = "OPTIMAL" if ret_info["final_primal_inf"] <= self.solver_tol else "INACCURATE"
    path_length_taken_rel = np.linalg.norm(x0 - x0copy, ord=1) / np.linalg.norm(x0, ord=1)
    dual_path_length_taken_rel = np.linalg.norm(w0 - w0copy, ord=1) / np.linalg.norm(w0, ord=1)
    ret_info["path_length_taken_rel"] = path_length_taken_rel
    ret_info["dual_path_length_taken_rel"] = dual_path_length_taken_rel
    # check if program needs to be recorded
    if self.record_programs:
      rprint(f"[yellow]Recording LP program for offline playback...[/yellow]")
      # create inputs to standard form (except A, b --> constructed in sparse form when needed)
      ret_x = allocX.flatten()
      ret_obj_val = ret_info["final_primal_obj"]
      ret_status = program_status
      solver_time = total_time
      gpu_duals_array = np.asarray([self.gpu_duals[cluster] for cluster in self.cluster_ordering])
      job_duals_array = np.asarray([self.job_duals[jobname] for jobname in job_ordering])
      stdform_inputs = {
        "c": stdc, "num_jobs": num_jobs, "num_configs": num_configs,
        "job_ordering": job_ordering, "time": self.current_time, "solver": "ALCD", 
        "solver_options": self.solver_options, "x_opt": ret_x, "obj_opt": ret_obj_val,
        "job_duals" : job_duals_array, "gpu_duals" : gpu_duals_array,
        "solver_status": ret_status, "solver_time_ms": solver_time * 1000, "info": ret_info,
        "load_time_ms" : program_load_time * 1000, "init_time_ms": program_init_time * 1000,
        "solve_time_ms": program_solve_time * 1000
      }
      self.recorded_programs.append(stdform_inputs)
    
    # commit new job ordering
    self.job_ordering = job_ordering

    if program_status != "OPTIMAL":
      rprint(f"ERROR :: LP did not converge to optimal solution; returning previous solution")
      rprint(f"Solver status: {program_status}, exited after {program_solve_time:.2f} seconds.")
      rprint(f"ALCD Solver returned info: {lpinfo}")
      return self.allocations
    else:
      rprint(f"Problem size: {num_jobs}x{num_configs}={num_jobs*num_configs/1000:.1f}k vars, solver time: {total_time*1000:.2f} ms, optimal value: {ret_info['final_primal_obj']:.2f}")
      rprint(f"\t Load/Compile: {program_load_time*1000:.2f}ms, Init: {program_init_time*1000:.2f} ms, Solve: {program_solve_time*1000:.2f} ms, Path Length (rel): {path_length_taken_rel:.2f}, Dual Path Length (rel): {dual_path_length_taken_rel:.2f}")

    # extract allocations
    # allocs = allocX.round(3)
    allocs = allocX
    # allocs[np.abs(allocs) < 1e-2] = 0
    # allocs[np.abs(allocs - 1) < 1e-2] = 1
    alloced_gpus = np.matmul(self.config_cnstr_matrix, np.sum(allocs, axis=0))
    violations = np.where(alloced_gpus > self.max_ngpus + 0.5)[0]
    # add a 0.1 buffer for floating point errors
    assert np.all(alloced_gpus <= self.max_ngpus + 0.5), f"GPU allocation exceeds available GPUs: {alloced_gpus} >= {self.max_ngpus}: {alloced_gpus[violations]} > {self.max_ngpus[violations] + 0.5}"
    rprint(f"Allocated GPUs: {alloced_gpus}")
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
        # at-most one config allocated to this job
        if nnz_job_alloc == 1:
          job_alloc_idx = np.argmax(job_alloc)
          alloc_config = self.configs[job_alloc_idx]
          _, ngpus, cluster = alloc_config
          solver_ngpus = np.matmul(self.config_cnstr_matrix, job_alloc)
          solver_ngpus = np.round(solver_ngpus[solver_ngpus > 0][0], 0)
          if solver_ngpus != ngpus:
            # alloced fewer than ngpus, but only one config
            # fix by finding the largest config smaller than ngpus
            cluster_id = self.cluster_ordering.index(cluster)
            while (job_alloc_idx > 0 and self.config_cnstr_matrix[cluster_id, job_alloc_idx] > 0):
              job_alloc_idx -= 1
              _, new_ngpus, cluster = self.configs[job_alloc_idx]
              if (new_ngpus <= ngpus):
                rprint(f"[yellow]WARNING :: Invalid allocation: {solver_ngpus} != {ngpus} --> fixed to {self.configs[job_alloc_idx]}[/yellow]")
                alloc_config = self.configs[job_alloc_idx]
                ngpus = new_ngpus
                break
            if ngpus > solver_ngpus:
              rprint(f"[red]ERROR :: Job {jobname} has invalid allocation: {solver_ngpus} != {ngpus}[/red]")
              ngpus = 0
              alloc_config = None
          if cluster_free_gpus[cluster] >= ngpus:
            cluster_free_gpus[cluster] -= ngpus
            self.allocations[jobname] = alloc_config
          else:
            rprint(f"No enough free GPUs in cluster {cluster} for job {jobname}: {cluster_free_gpus[cluster]} < {ngpus}")
            self.allocations[jobname] = None
          # rprint(f"Job: {jobname}, non-zero allocs: {job_alloc[job_alloc > 0]}, alloc:{alloc_config}, expected_alloc: {solver_ngpus[solver_ngpus > 0]}")
        else:
          # partial alloc with >1 configs selected (fractionally)
          partial_allocs_obj_val += np.dot(job_alloc, c[i * num_configs:(i+1)*num_configs])
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
    rprint(f"Cluster free GPUs: {cluster_free_gpus}")
    
    stat = {"time": self.current_time, "num_jobs": num_jobs, "num_vars": num_jobs*num_configs, 
            "setup_time_ms": (program_init_time)*1000, "solve_time_ms": (program_solve_time)*1000, 
            "solver_status": program_status, "objective_val": ret_info["final_primal_obj"], 
            "num_partial_allocs": len(partial_allocs), "partial_allocs_obj_val": partial_allocs_obj_val}
    self.solver_stats.append(stat)
    
    '''
    rprint(f"Cluster GPU usage:")
    for cluster, ngpus in cluster_alloced_gpus.items():
      assert ngpus <= self.cluster_gpus[cluster], f"GPU type: {cluster} overallocated: allocated={ngpus} > available={self.cluster_gpus[cluster]}"
      rprint(f"\t{cluster} = {ngpus} / {self.cluster_gpus[cluster]} GPUs ({round(ngpus / self.cluster_gpus[cluster] * 100, 2)}%)")
    '''
