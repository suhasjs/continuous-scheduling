import jaxopt.tree_util
from .sia_ilp import SiaILP
from .policy_utils import round_allocations_largest, initialize_subproblem_solver, pjadmm_iter_fun
import cvxpy as cp
import numpy as np
import jax.numpy as jnp
import jax
import jaxopt
from rich import print as rprint
import time
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16'
# os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
# os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
# jax.config.update("jax_explain_cache_misses", True)

class SiaLPRelaxedPJADMM(SiaILP):
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
    self.num_configs = len(self.configs)
    self.config_cnstr_matrix, self.config_cnstr_vec = configs_cnstrs

    # policy parameters
    self.lambda_no_alloc = policy_options.get('lambda_no_alloc', 1.1)
    self.p_value = policy_options.get('p_value', 0.5)

    # rounding functions to convert fractional allocations to integer allocations
    self.round_allocations = round_allocations_largest

    # cluster state
    self.active_jobs = {}
    self.allocations = {}
    self.job_utilities = {}
    self.current_time = 0

    # stats
    self.solver_stats = []

    # solver options
    self.solver_options = solver_options
    self.solver_name = solver_options.pop('solver', 'PJADMM')
    assert self.solver_name == "PJADMM", f"Invalid solver: {self.solver_name}"
    self.warm_start = solver_options.pop('warm_start', False)
    self.solver_block_size = solver_options.pop('block_size', 5)
    self.solver_iters_per_sync = solver_options.pop('iters_per_sync', 20)
    self.solver_max_iters = solver_options.pop('max_iters', 5000)
    self.solver_prox_mu = solver_options.pop('prox_mu', 10)
    self.solver_viol_beta = solver_options.pop('viol_beta', 0.1)
    self.solver_dual_tau = solver_options.pop('dual_tau', 0.2)
    self.solver_tol = solver_options.pop('tol', 1e-5)
    self.solver_backend = solver_options.pop('backend', 'cpu')
    self.solver_normalize_cnstrs = solver_options.pop('normalize_cnstrs', True)
    self.solver_normalize_obj = solver_options.pop('normalize_obj', True)
    self.require_binary_solutions = solver_options.pop('require_binary_solutions', False)
    self.lbfgs_max_iters = 8
    self.lbfgs_history_size = 6
    self.print_compilation_stats = False
    # key: (num_blocks, block_size, num_configs)
    self.cached_solver_loop_fns = {}

    # saved solver state between two timesteps
    self.gpu_duals = np.zeros(self.num_gputypes)
    self.gpu_slacks = np.zeros(self.num_gputypes)
    self.job_primals = dict()
    self.job_duals = dict()
    self.job_slacks = dict()

    # block info for solver
    self.num_blocks = None # num partitions of blocks
    self.block_to_jobs_map = dict() # {block_id: [jobid1, jobid2, ...]}
    self.num_valid_jobs = dict() # number of jobs with non-zero utility in each block
    self.job_ordering = None # sorted list of jobnames
    # NOTE: job -> block map is simply obtained by dividing jobid by block_size
    self.job_id_map = dict() # {jobname: jobid}
    
  '''
  fn to Populate block info for the solver
  assigns each job to a block and a job_idx_in_block
  '''
  def __populate_block_info(self):
    if len(self.active_jobs) == 0:
      rprint(f"[yellow]No active jobs to populate block info[/yellow]")
      return
    self.num_jobs = len(self.active_jobs)
    self.num_blocks = int(np.ceil(self.num_jobs / self.solver_block_size))
    self.max_num_jobs = self.num_blocks * self.solver_block_size
    self.block_to_jobs_map.clear()
    self.num_valid_jobs.clear()
    self.job_id_map.clear()
    self.job_ordering = sorted(list(self.active_jobs.keys()))
    block_idx = -1
    job_idx_in_block = 0
    for i in range(len(self.job_ordering)):
      # new block
      if job_idx_in_block == 0:
        block_idx += 1
        self.block_to_jobs_map[block_idx] = []
        self.num_valid_jobs[block_idx] = 0
      # add job to block
      self.block_to_jobs_map[block_idx].append(i)
      self.job_id_map[self.job_ordering[i]] = i
      # update job_idx_in_block
      job_idx_in_block = (job_idx_in_block + 1) % self.solver_block_size
      self.num_valid_jobs[block_idx] += 1
    rprint(f"Populated block info: {self.num_blocks} blocks, {self.num_jobs} jobs, block size: {self.solver_block_size}")
  
  def __get_warm_start_guess(self):
    job_primals_k = jnp.zeros((self.num_blocks, self.solver_block_size, self.num_configs)) # job allocations
    job_slacks_k = jnp.zeros((self.num_blocks, self.solver_block_size)) # job slacks
    job_duals_k = jnp.zeros((self.num_blocks, self.solver_block_size)) # job slacks
    gpu_duals_k = jnp.array(self.gpu_duals) # gpu duals
    gpu_slacks_k = jnp.array(self.gpu_slacks) # gpu slacks

    # set values from recorded state if any
    for i, jobname in enumerate(self.job_ordering):
      block_id, job_idx_in_block = (i // self.solver_block_size), i % self.solver_block_size
      if jobname not in self.job_primals:
        # zero allocation for this job
        # non-zero duals/slacks for this job
        job_slacks_k = job_slacks_k.at[block_id, job_idx_in_block].set(0.0)
        job_primals_k = job_primals_k.at[block_id, job_idx_in_block, :].set(0.0)
        job_duals_k = job_duals_k.at[block_id, job_idx_in_block].set(0.0)
        rprint(f"[yellow]Job {jobname} not found in saved state[/yellow]")
        continue
      else:
        rprint(f"[green] Warm-starting job {jobname} from saved state[/green]")
      job_primals_k = job_primals_k.at[block_id, job_idx_in_block, :].set(self.job_primals[jobname])
      job_slacks_k = job_slacks_k.at[block_id, job_idx_in_block].set(self.job_slacks[jobname])
      job_duals_k = job_duals_k.at[block_id, job_idx_in_block].set(self.job_duals[jobname])
    
    return job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k

  # save state for warm-starting the next timestep
  # Args: (job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k)
  # job_primals_k: (num_jobs, num_configs)
  # job_slacks_k: (num_jobs)
  # job_duals_k: (num_jobs)
  # gpu_duals_k: (num_gputypes)
  # gpu_slacks_k: (num_gputypes)
  def __save_warm_start_state(self, job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k):
    rprint(f"SAVE WARM START STATE :: Primals shape: {job_primals_k.shape}, Slacks shape: {job_slacks_k.shape}, Duals shape: {job_duals_k.shape}")
    for i, jobname in enumerate(self.job_ordering):
      self.job_primals[jobname] = job_primals_k[i, :]
      self.job_slacks[jobname] = job_slacks_k[i]
      self.job_duals[jobname] = job_duals_k[i]
    self.gpu_duals = gpu_duals_k
    self.gpu_slacks = gpu_slacks_k
  
  # early-binds some jobs, saves their state and updates; also resets sharding to Amat.sharding for all variables
  # opt_state = (job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k)
  # constants = (vmapped_cmat, Amat, bvec, primal_bounds)
  def __early_bind_jobs(self, opt_state, params, early_bound_ids):
    rprint(f"Early binding {len(early_bound_ids)} jobs: {early_bound_ids}")
    # unpack args
    job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k = opt_state
    vmapped_cmat, Amat, bvec, primal_bounds = params
    # if no early bound jobs, return early
    if early_bound_ids is None or len(early_bound_ids) == 0:
      return job_primals_k, job_slacks_k, job_duals_k, vmapped_cmat
    # 1. Save warm start state for early bound job ids
    job_primals_partial_sum = jnp.zeros((self.num_configs,))
    for i in early_bound_ids:
      jobname = self.job_ordering[i]
      block_id, job_idx_in_block = (i // self.solver_block_size), i % self.solver_block_size
      job_primals_partial_sum += job_primals_k[block_id, job_idx_in_block, :]
      self.job_primals[jobname] = np.array(job_primals_k[block_id, job_idx_in_block, :])
      self.job_slacks[jobname] = np.array(job_slacks_k[block_id, i])
      self.job_duals[jobname] = np.array(job_duals_k[block_id, i])
    
    # 2. Create new jobname -> job_id mapping
    new_njobs = self.num_jobs - len(early_bound_ids)
    keep_idxs = [i for i in range(self.num_jobs) if i not in early_bound_ids]
    new_job_ordering = [self.job_ordering[i] for i in keep_idxs]
    new_num_blocks = int(np.ceil(new_njobs / self.solver_block_size))
    new_job_id_map, new_block_to_jobs_map, new_num_valid_jobs = dict(), dict(), dict()
    for i in range(new_njobs):
      block_id, job_idx_in_block = (i // self.solver_block_size), i % self.solver_block_size
      jobname = new_job_ordering[i]
      new_job_id_map[jobname] = i
      if job_idx_in_block == 0:
        new_block_to_jobs_map[block_id] = []
        new_num_valid_jobs[block_id] = 0
      new_block_to_jobs_map[block_id].append(i)
      new_num_valid_jobs[block_id] += 1
    self.num_jobs = new_njobs
    self.num_blocks = new_num_blocks
    self.max_num_jobs = self.num_blocks * self.solver_block_size
    self.job_ordering = new_job_ordering
    self.job_id_map = new_job_id_map
    self.block_to_jobs_map = new_block_to_jobs_map
    self.num_valid_jobs = new_num_valid_jobs

    # 3. Copy over job_ordering, job_primals_k, job_slacks_k, job_duals_k, vmapped_cmat
    new_job_primals_k = jnp.zeros((self.num_blocks, self.solver_block_size, self.num_configs))
    new_job_slacks_k = jnp.ones((self.num_blocks, self.solver_block_size))
    new_job_duals_k = jnp.zeros((self.num_blocks, self.solver_block_size))
    new_vmapped_cmat = jnp.ones((self.num_blocks, self.solver_block_size, self.num_configs)) * 10
    for i, old_job_idx in enumerate(keep_idxs):
      # copy primal, slack and duals for non-early bound jobs
      jobname = self.job_ordering[i]
      block_id, job_idx_in_block = (i // self.solver_block_size), i % self.solver_block_size
      old_block_id, old_job_idx_in_block = (old_job_idx // self.solver_block_size), old_job_idx % self.solver_block_size
      new_job_primals_k = new_job_primals_k.at[block_id, job_idx_in_block, :].set(job_primals_k[old_block_id, old_job_idx_in_block, :])
      new_job_slacks_k = new_job_slacks_k.at[block_id, job_idx_in_block].set(job_slacks_k[old_block_id, old_job_idx])
      new_job_duals_k = new_job_duals_k.at[block_id, job_idx_in_block].set(job_duals_k[old_block_id, old_job_idx])
      new_vmapped_cmat = new_vmapped_cmat.at[block_id, job_idx_in_block, :].set(vmapped_cmat[old_block_id, old_job_idx_in_block, :])
    
    # 4. Update constraints to reflect early bound jobs
    new_Amat = Amat
    # need to device_put because job_primals_partial_sum is sharded, so new_bvec also becomes sharded [don't want]
    new_bvec = bvec - (Amat @ job_primals_partial_sum)
    # new_bvec = jax.device_put(bvec - (Amat @ job_primals_partial_sum), Amat.sharding)

    new_optstate = (new_job_primals_k, new_job_slacks_k, new_job_duals_k, gpu_duals_k, gpu_slacks_k)
    new_params = (new_vmapped_cmat, new_Amat, new_bvec, primal_bounds)
    return new_optstate, new_params

  def solve(self):
    setup_start_t = time.time()
    # populate block info
    self.__populate_block_info()

    ##### Extract problem parameters
    # job utilities --> set cost = -utility
    cmat = np.array([(-np.array(self.job_utilities[jobname]) + self.lambda_no_alloc) for jobname in self.job_ordering])
    # rprint(f"Cost matrix: {cmat}")
    # block-wise job utilities --> set dummy job utility to +1
    vmapped_cmat = jnp.ones((self.num_blocks, self.solver_block_size, self.num_configs))
    for i in range(self.num_blocks):
      sub_cmat = cmat[self.block_to_jobs_map[i], :]
      vmapped_cmat = vmapped_cmat.at[i, :self.num_valid_jobs[i], :].set(sub_cmat)
    # last block may have dummy jobs
    # config constraints
    Amat = jnp.array(self.config_cnstr_matrix)
    # config constraints
    bvec = jnp.array(self.config_cnstr_vec)
    if self.solver_normalize_cnstrs:
      cnstr_scale_factor = bvec
      rprint(f"Scaling down A, b by {cnstr_scale_factor}")
      Amat = Amat / cnstr_scale_factor.reshape(-1, 1)
      bvec = bvec / cnstr_scale_factor
    else:
      cnstr_scale_factor = 1.0
    if self.solver_normalize_obj:
      obj_scale_factor = jnp.max(jnp.abs(vmapped_cmat))
      rprint(f"Scaling down c by {obj_scale_factor}")
      vmapped_cmat = vmapped_cmat / obj_scale_factor
    else:
      obj_scale_factor = 1.0
    rprint(f"Norm of Amat: {jnp.linalg.norm(Amat)}, Norm of bvec: {jnp.linalg.norm(bvec)}")

    #### Initialize solver state
    def init_subproblem_solver_helper(opt_state, params, constants):
      # unpack args
      job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k = opt_state
      vmapped_cmat, Amat, bvec, primal_bounds = params
      solver_viol_beta, solver_prox_mu = constants
      # additional variables
      sum_vec_k = job_slacks_k - 1 - job_duals_k
      vmapped_tk = (Amat @ job_primals_k.sum(axis=[0, 1]) - bvec - gpu_duals_k)
      gpu_vec_k = vmapped_tk - jax.vmap(lambda x: Amat @ x)(job_primals_k.sum(axis=1)) + gpu_slacks_k
      subproblem_solver_init_params = {
        "job_primals_k": job_primals_k, "job_primal_bounds": primal_bounds,
        "job_slacks_k": job_slacks_k, "job_duals_k": job_duals_k,
        "gpu_duals_k": gpu_duals_k, "gpu_slacks_k": gpu_slacks_k,
        "gpu_vec_k": gpu_vec_k, "sum_vec_k": sum_vec_k, "vmapped_cmat": vmapped_cmat,
      }
      auglag_other_params = {
        "block_size": self.solver_block_size, "num_configs": self.num_configs, 
        "lambda_no_alloc": self.lambda_no_alloc, "Amat": Amat, 
        "require_binary_solutions": self.require_binary_solutions
      }
      lbfgs_solver_args = {
        "max_iters" : self.lbfgs_max_iters,
        "history_size" : self.lbfgs_history_size,
        "viol_beta" : solver_viol_beta,
        "prox_mu" : solver_prox_mu,
        "solver_backend": self.solver_backend
      }

      return initialize_subproblem_solver(lbfgs_solver_args, subproblem_solver_init_params, auglag_other_params)
    
    # _k denotes variables at iter k, _kp1 denotes variables at iter (k+1)
    job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k = self.__get_warm_start_guess()
    primal_bounds = (jnp.zeros_like(job_primals_k[0, :].reshape(-1)), jnp.ones_like(job_primals_k[0, :].reshape(-1)))
    opt_state = (job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k)
    params = (vmapped_cmat, Amat, bvec, primal_bounds)
    constants = (self.solver_viol_beta, self.solver_prox_mu)
    subproblem_solver, vmapped_subproblem_state = init_subproblem_solver_helper(opt_state, params, constants)
    #### Start solver loop
    stats_k = {
      "gpu_cnstr_viol_norms": jnp.zeros(self.solver_iters_per_sync),
      "sumto1_cnstr_viol_norms": jnp.zeros(self.solver_iters_per_sync),
      "binarization": jnp.zeros(self.solver_iters_per_sync),
      "obj_vals": jnp.zeros(self.solver_iters_per_sync),
    }
    # other_state = (eta, d_k, alpha_k)
    other_state = jnp.array([0.999, 0.0, 1.0])

    init_loop_state = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, 
                       job_duals_k, vmapped_subproblem_state, other_state, stats_k)
    
    problem_args = {
      "Amat": Amat, "bvec": bvec, "vmapped_cmat": vmapped_cmat, "job_primal_bounds": primal_bounds,
      "cnstr_scale_factor": cnstr_scale_factor, "obj_scale_factor": obj_scale_factor
    }
    prox_mu = self.solver_viol_beta * (job_primals_k.shape[0]*job_primals_k.shape[1] - 1)
    iter_args = {
      "solver_viol_beta": self.solver_viol_beta, 
      "solver_prox_mu": prox_mu, "solver_dual_tau": self.solver_dual_tau
    }

    ### Run solver loop
    def get_solver_loop_fn():
      cache_key = (self.num_blocks, self.solver_block_size, self.num_configs)
      if cache_key in self.cached_solver_loop_fns:
        rprint(f"CACHE HIT:: Using cached prox-jacobi-admm loop. Problem size: ({self.num_blocks}, {self.solver_block_size}, {self.num_configs})")
        jitted_iter_func = self.cached_solver_loop_fns[cache_key]
      else:
        rprint(f"CACHE MISS:: Compiling prox-jacobi-admm loop. Problem size: ({self.num_blocks}, {self.solver_block_size}, {self.num_configs})")
        iter_fun = jax.tree_util.Partial(pjadmm_iter_fun, block_size=self.solver_block_size, 
                                        num_configs=self.num_configs, subproblem_solver=subproblem_solver, 
                                        solver_backend=self.solver_backend)
        jitted_iter_func = jax.jit(iter_fun, backend=self.solver_backend)
        # jitted_iter_func = jitted_iter_func.lower(k=0, state=init_loop_state, problem_args=problem_args, iter_args=iter_args).compile()
        self.cached_solver_loop_fns[cache_key] = jitted_iter_func
      # jitted_iter_func = iter_fun
      if self.print_compilation_stats:
        compiled_func = jitted_iter_func.lower(0, init_loop_state, problem_args, iter_args).compile()
        cost_analysis = compiled_func.cost_analysis()[0]
        bytes_accessed = 0
        for k, v in cost_analysis.items():
          if 'bytes' in k:
            bytes_accessed += v
        flop_to_bytes_ratio = cost_analysis['flops'] / bytes_accessed
        rprint(f"Cost analysis: flops={cost_analysis['flops'] / 1e6} MFLOPs/iter, flop:bytes_accessed ratio={flop_to_bytes_ratio}")
        memory_analysis = compiled_func.memory_analysis()
        rprint(f"Memory analysis: {memory_analysis}")
      return jitted_iter_func
    
    jitted_iter_func = get_solver_loop_fn()
    rprint(f"Running prox-jacobi-admm loop...")
    final_loop_state = init_loop_state
    iter_times_ms, previous_iter_vals = [], None
    previous_iter_vals, track_stats = None, None
    solve_start_t = time.time()
    has_solver_converged = False
    for i in range(self.solver_max_iters):
      # check for early convergence
      if (i+1) % self.solver_iters_per_sync == 0:
        jax.block_until_ready(final_loop_state)
        job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k, vmapped_subproblem_state, other_state, stats_k = final_loop_state
        # jax.debug.breakpoint()
        if previous_iter_vals is None:
          previous_iter_vals = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k)
          track_stats = jax.tree_map(lambda x: x.copy(), stats_k)
          track_stats = jax.device_put(track_stats, Amat.sharding)
        else:
          target_sharding = job_primals_k.sharding
          previous_iter_vals = jax.device_put(previous_iter_vals, target_sharding)
          # break if progress < tol
          prev_job_primals_k, prev_gpu_duals_k, prev_gpu_slacks_k, prev_job_slacks_k, prev_job_duals_k = previous_iter_vals
          primal_diff = jnp.linalg.norm(job_primals_k - prev_job_primals_k)
          gpu_dual_diff = jnp.linalg.norm(gpu_duals_k - prev_gpu_duals_k)
          gpu_slack_diff = jnp.linalg.norm(gpu_slacks_k - prev_gpu_slacks_k)
          job_slack_diff = jnp.linalg.norm(job_slacks_k - prev_job_slacks_k)
          job_dual_diff = jnp.linalg.norm(job_duals_k - prev_job_duals_k)
          rprint(f"Iteration {i} [t = {(time.time() - solve_start_t)*1000:.2f} ms] :: primal_change={primal_diff:.3f}, gpu_dual_change={gpu_dual_diff:.3f}, gpu_slack_change={gpu_slack_diff:.3f}, job_slack_change={job_slack_diff:.3f}, job_dual_change={job_dual_diff:.3f}")
          rprint(f"\t obj={stats_k['obj_vals'][i-1]:.3f}, gpu_cnstr_viol_norm:{stats_k['gpu_cnstr_viol_norms'][i-1]:.3f}, job_cnstr_viol_norm:{stats_k['sumto1_cnstr_viol_norms'][i-1]:.3f}, d_k:{other_state[1].round(3)}, alpha_k:{other_state[2].round(3)}")
          gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms = stats_k['gpu_cnstr_viol_norms'], stats_k['sumto1_cnstr_viol_norms']
          iter_args["solver_prox_mu"] = iter_args["solver_prox_mu"] * 1.03
          # score convergence based on binarization and job-level constraints
          binarization = jnp.abs(jnp.multiply(job_primals_k, 1-job_primals_k).sum(axis=2)).flatten()
          sum_cnstr_viols = jnp.abs(jnp.sum(job_primals_k, axis=2) + job_slacks_k - 1).flatten()
          convergence_score = (binarization + sum_cnstr_viols)[:self.num_jobs]
          # rprint(f"\tBinarization: {binarization}, prox_mu: {iter_args['solver_prox_mu']}")
          # rprint(f"\tConvergence scores: {convergence_score.round(2)}")
          converged_idxs = jnp.where(convergence_score < 1e-3)[0]
          rprint(f"\tConverged job IDs: {converged_idxs} --> {len(converged_idxs)}/{self.num_jobs}")

          stats_k = jax.device_put(stats_k, Amat.sharding)
          track_stats = jax.tree_map(lambda x, y: jnp.append(x, y), track_stats, stats_k)
          if (gpu_cnstr_viol_norms[-1] < self.solver_tol and sumto1_cnstr_viol_norms[-1] < self.solver_tol):
            rprint(f"Breaking after {i+1} iterations : gpu_cntr_viol_norm = {gpu_cnstr_viol_norms[-1]}, sumto1_cnstr_viol_norms = {sumto1_cnstr_viol_norms[-1]} < tol={self.solver_tol}")
            has_solver_converged = True
            break
          else:
            rprint(f"Job primals shape: {job_primals_k.shape}, gpu_duals shape: {gpu_duals_k.shape}, gpu_slacks shape: {gpu_slacks_k.shape}, job_slacks shape: {job_slacks_k.shape}")
            previous_iter_vals = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k)

          # early bind values for converged jobs
          if len(converged_idxs) >= (self.solver_block_size // 2) :
            opt_state = (job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k)
            params = (vmapped_cmat, Amat, bvec, primal_bounds)
            opt_state, params, converged_idxs = jax.device_put((opt_state, params, converged_idxs), Amat.sharding)
            constants = (iter_args["solver_viol_beta"], iter_args["solver_prox_mu"])
            opt_state, params = self.__early_bind_jobs(opt_state, params, converged_idxs)
            job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k = opt_state
            vmapped_cmat, Amat, bvec, primal_bounds = params
            # re-initialize subproblem solver
            _, vmapped_subproblem_state = init_subproblem_solver_helper(opt_state, params, constants)
            jitted_iter_func = get_solver_loop_fn()
            # update other loop variables
            final_loop_state = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k,
                                vmapped_subproblem_state, other_state, stats_k)
            problem_args = {
              "Amat": Amat, "bvec": bvec, "vmapped_cmat": vmapped_cmat, "job_primal_bounds": primal_bounds,
              "cnstr_scale_factor": cnstr_scale_factor, "obj_scale_factor": obj_scale_factor
            }
            previous_iter_vals = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k)
      # append stats to previous iters
      start_t = time.time()
      final_loop_state = jitted_iter_func(k=(i % self.solver_iters_per_sync), state=final_loop_state,
                                          problem_args=problem_args, iter_args=iter_args)
      end_t = time.time()
      iter_time = end_t - start_t
      iter_times_ms.append(iter_time*1000)
    jax.block_until_ready(final_loop_state)
    solve_end_t = time.time()

    ### Post-processing solver output
    job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k, vmapped_subproblem_state, other_state, stats_k = final_loop_state
    num_iters = len(iter_times_ms)
    rprint(f"Prox-Jacobi-ADMM: finished {num_iters} iterations in {(solve_end_t - solve_start_t)*1000:.2f} ms")
    job_primals = np.squeeze(np.array(job_primals_k.reshape(-1, self.num_configs))).round(2)
    gpu_duals = np.squeeze(np.array(gpu_duals_k))
    gpu_slacks = np.squeeze(np.array(gpu_slacks_k))
    job_slacks = np.squeeze(np.array(job_slacks_k.reshape(-1,)))
    job_duals = np.squeeze(np.array(job_duals_k.reshape(-1,)))
    rprint(f"Shapes: primals={job_primals.shape}, gpu_duals={gpu_duals.shape}, gpu_slacks={gpu_slacks.shape}, job_slacks={job_slacks.shape}, job_duals={job_duals.shape}")
    rprint(f"\t sum(alloc) for dummy jobs = {np.sum(job_primals[self.num_jobs:, :].reshape(-1))}")

    if self.solver_normalize_cnstrs:
      gpu_slacks = gpu_slacks * cnstr_scale_factor
    setup_time_ms = (solve_start_t - setup_start_t)*1000
    solve_time_ms = (solve_end_t - solve_start_t)*1000
    total_runtime = (solve_end_t - setup_start_t)*1000
    rprint(f"Solver timings: setup={setup_time_ms:.2f}ms, solve={solve_time_ms:.2f}ms, total={total_runtime:.2f}ms")

    # scatter allocs back to jobs, persist state for warm-start
    self.__save_warm_start_state(job_primals, job_slacks, job_duals, gpu_duals, gpu_slacks)

    # persist allocs
    stat = {"time": self.current_time, "num_jobs": self.num_jobs, "num_vars": self.num_jobs*self.num_configs, 
            "setup_time_ms": setup_time_ms, "solve_time_ms": solve_time_ms, 
            "solver_status": str(has_solver_converged), "objective_val": track_stats["obj_vals"][-1]}
    return stat, has_solver_converged

  def get_save_state(self):
    state = super().get_save_state()
    return state
  
  def load_saved_state(self, state, jobs):
    super().load_saved_state(state, jobs)

  # override optimize_allocations to use LP relaxation of ILP + rounding
  def optimize_allocations(self):
    # solve problem
    solver_stat, has_solver_converged = self.solve()
    if not has_solver_converged:
      self.solver_stats.append(solver_stat)
      return
    self.job_ordering = sorted(list(self.active_jobs.keys()))
    self.num_jobs = len(self.job_ordering)

    # extract allocations only if solver has converged
    cluster_free_gpus = {cluster: cluster_max_gpus for cluster, cluster_max_gpus in zip(self.cluster_ordering, self.max_ngpus)}
    partial_allocs = {}
    partial_allocs_obj_val = 0
    for i, jobname in enumerate(self.job_utilities.keys()):
      job_alloc = self.job_primals.get(jobname, None)
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
          partial_allocs_obj_val += np.dot(job_alloc, self.job_utilities[jobname])
          valid_idxs = np.where(job_alloc > 0)[0]
          config_weights = job_alloc[job_alloc > 0]
          config_choices = [self.configs[idx] for idx in valid_idxs]
          partial_allocs[jobname] = sorted([(x,y) for x,y in zip(config_choices, config_weights)], key=lambda x: x[1], reverse=True)
      else:
        rprint(f"[red]ERROR :: Job {jobname} has invalid allocation: {np.sum(job_alloc)}[/red]")
        self.allocations[jobname] = None
    if len(partial_allocs) > 0:
      rprint(f"[yellow]#Partial allocations: {len(partial_allocs)}/{self.num_jobs}[/yellow]")
      rprint(f"\tPartial allocations objective value: {partial_allocs_obj_val}")
      rounded_allocs = self.round_allocations(partial_allocs, cluster_free_gpus)
      for k in rounded_allocs.keys():
        # rprint(f"\tJob: {k}, partial allocation: {partial_allocs[k]} -> rounded allocation: {rounded_allocs[k]}")
        pass
      self.allocations.update(rounded_allocs)
    
    stat = solver_stat
    stat.update({"num_partial_allocs": len(partial_allocs), "partial_allocs_obj_val": partial_allocs_obj_val})
    self.solver_stats.append(stat)
    
    '''
    rprint(f"Cluster GPU usage:")
    for cluster, ngpus in cluster_alloced_gpus.items():
      assert ngpus <= self.cluster_gpus[cluster], f"GPU type: {cluster} overallocated: allocated={ngpus} > available={self.cluster_gpus[cluster]}"
      rprint(f"\t{cluster} = {ngpus} / {self.cluster_gpus[cluster]} GPUs ({round(ngpus / self.cluster_gpus[cluster] * 100, 2)}%)")
    '''