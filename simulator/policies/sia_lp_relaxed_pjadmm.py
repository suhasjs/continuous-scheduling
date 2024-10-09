import jaxopt.tree_util
from .sia_ilp import SiaILP
from .policy_utils import round_allocations_largest
import cvxpy as cp
import numpy as np
import jax.numpy as jnp
import jax
import jaxopt
from rich import print as rprint
import time
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=16'

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
    self.solver_iters_per_sync = solver_options.pop('iters_per_sync', 10)
    self.solver_max_iters = solver_options.pop('max_iters', 5000)
    self.solver_prox_mu = solver_options.pop('prox_mu', 50)
    self.solver_viol_beta = solver_options.pop('viol_beta', 0.2)
    self.solver_dual_tau = solver_options.pop('dual_tau', 0.25)
    self.solver_tol = solver_options.pop('tol', 1e-5)
    self.solver_backend = solver_options.pop('backend', 'cpu')
    self.solver_normalize_cnstrs = solver_options.pop('normalize_cnstrs', True)
    self.solver_normalize_obj = solver_options.pop('normalize_obj', True)
    self.require_binary_solutions = solver_options.pop('require_binary_solutions', True)
    self.lbfgs_max_iters = 8
    self.lbfgs_history_size = 6


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
    self.jobid_to_block_map = dict() # {jobid: (block_id, job_idx_in_block)}
    self.job_ordering = None # sorted list of jobnames
    
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
    self.jobid_to_block_map.clear()
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
      self.jobid_to_block_map[i] = (block_idx, job_idx_in_block)
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
      block_id, job_idx_in_block = self.jobid_to_block_map[i]
      if jobname not in self.job_primals:
        # zero allocation for this job
        # non-zero duals/slacks for this job
        job_slacks_k = job_slacks_k.at[block_id, job_idx_in_block].set(0.01)
        job_duals_k = job_duals_k.at[block_id, job_idx_in_block].set(0.01)
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
    # _k denotes variables at iter k, _kp1 denotes variables at iter (k+1)
    job_primals_k, job_slacks_k, job_duals_k, gpu_duals_k, gpu_slacks_k = self.__get_warm_start_guess()
    # additional variables to simplify computation
    gpu_vec_k =  jnp.zeros((self.num_blocks, self.num_gputypes))
    sum_vec_k = jnp.zeros((self.num_blocks, self.solver_block_size))
    primal_bounds = (jnp.zeros_like(job_primals_k[0, :].reshape(-1)), jnp.ones_like(job_primals_k[0, :].reshape(-1)))

    '''
    Augmented Lagrangian:
    L(x, s, f, u, v) = sum_i { c_i^T x_i } +                                  <--- objective
                      beta/2 * || A * sum_i (x_i) + s - b - u ||_2^2 +       <--- GPU constraint violation
                      beta/2 * sum_i (|| x_i^T * 1 + f_i - 1 - v_i ||_2^2) + <--- sum-to-1 constraint violation
                      mu/2 * || x - x^k ||_2^2 +                             <--- proximal term for primal
    [IGNORE]           mu/2 * || s - s^k ||_2^2 +                             <--- proximal term for GPU slack
    [IGNORE]           mu/2 * || f - f^k ||_2^2                               <--- proximal term for alloc slack
    where x = [x_1, x_2, ..., x_n],         <--- primal variables, one for each (job, config) pair
          s = [s_1, s_2, ..., s_m],         <--- slack variables, one for each GPU type
          f = [f_1, f_2, ..., f_n]          <--- slack variables, one for each job
          u = [u_1, u_2, ..., u_m]          <--- dual variables, one for each GPU type
          v = [v_1, v_2, ..., v_n]          <--- dual variables, one for each job
    subject to 0 <= x_i <= 1 for all i

    ### Prox Jacobi ADMM Iteration (k -> k+1):
    (1) x_i^{k+1} = argmin_{x_i} { c_i^T x_i +                                 <--- objective
                                  (beta/2) * || A * x_i + r_i^k ||_2^2  +      <--- GPU constraint violation
                                  (beta/2) * || x_i^T * 1 + y_i^k ||_2^2 +     <--- sum-to-1 constraint violation
                                  (mu/2) * || x_i - x_i^k ||_2^2               <--- proximal term
                                }
      where r_i^k = (A * (sum_{j != i} x_j^{k}) + s^k - b - u^k)        --> jacobi decomposition of GPU constraints
            y_i^k = (f_i^k - 1 - v_i^k)                                 --> jacobi decomposition of sum-to-1 constraints
      subject to 0 <= x_i <= 1 for all i                                --> box constraints on x_i
    (2) s^{k+1} = argmin_{s} { (beta/2) * || s + t^k ||_2^2 +           <--- GPU constraint violation
                              (mu/2) * || s - s^k ||_2^2               <--- proximal term
                            }
      where t^k = (sum_i (A * x_i^{k}) - b - u^k)
      We can compute s^{k+1} in closed form as:
      s^{k+1} = min(max(0, (mu * s^k - beta * t^k) / (mu + beta)), b)
    (3) f_i^{k+1} = argmin_{f_i} { (beta/2) * || f_i + z_i^k ||_2^2 +   <--- sum-to-1 constraint violation
                                  (mu/2) * || f_i - f_i^k ||_2^2       <--- proximal term
                                }
      where z_i^k = (x_i^{k}^T * 1 - 1 - v_i^k)
      We can compute f_i^{k+1} in closed form as:
      f_i^{k+1} = min(max(0, (mu * f_i^k - beta * z_i^k) / (mu + beta)), 1)
    (4) u^{k+1} = u^k - tau * (A * x^{k+1} + s^{k+1} - b)         <--- dual update for GPU constraints
    (5) v^{k+1} = v^k - tau * (x^{k+1}^T*1 + f^{k+1} - 1)             <--- dual update for sum-to-1 constraints
    '''
    def auglag_fun(xikp1, c_is, rki, xki, yki, aug_viol_beta, aug_prox_mu):
      # reshape xikp1 to (-1, nconfigs)
      reshaped_xikp1 = jax.tree_map(lambda x: x.reshape(-1, self.num_configs), xikp1)
      # add <c_i, x_i> term --> cost (negative of utility)
      ret = jaxopt.tree_util.tree_vdot(c_is, reshaped_xikp1)
      # add -lambda_no_alloc * sum_i (x_i) term --> incentivize allocation (penalize no allocation)
      scalar_add_val = jax.tree_map(lambda x: jnp.sum(x), reshaped_xikp1)
      ret = jaxopt.tree_util.tree_add_scalar_mul(ret, -1 * self.lambda_no_alloc, scalar_add_val)
      # add beta/2 * || A * sum_i (x_i) + s - b - u ||_2^2 term --> GPU constraint violation
      gpu_cnstr_viol = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, rki: Amat @ jnp.sum(x, axis=0) + rki, reshaped_xikp1, rki), squared=True)
      ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2), gpu_cnstr_viol)
      # add beta/2 * || sum (x_i) + f - 1 - v ||_2^2 term --> sum-to-1 constraint violation
      sumto_1_cnstr_viol = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, yki: jnp.sum(x, axis=1) + yki, reshaped_xikp1, yki), squared=True)
      ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2), sumto_1_cnstr_viol)
      # add mu/2 * || x - x^k ||_2^2 term --> proximal term for primals
      prox_term = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, y: x - y, reshaped_xikp1, xki), squared=True)
      ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_prox_mu / 2), prox_term)
      # binarization = x * (1 - x) --> penalize non-binary solutions if required
      if self.require_binary_solutions:
        rhs = jaxopt.tree_util.tree_add_scalar_mul(1, -1, reshaped_xikp1)
        binarization = jaxopt.tree_util.tree_vdot(reshaped_xikp1, rhs)
        ret = jaxopt.tree_util.tree_add_scalar_mul(ret, 1e-1, binarization)
      return ret

    #### Initialize state for LBFGS-B solver
    subproblem_solver = jaxopt.LBFGSB(fun=auglag_fun, jit=True, unroll=True, implicit_diff=False,
                                      maxiter=self.lbfgs_max_iters, tol=1e-4, stepsize=0.9, 
                                      history_size=self.lbfgs_history_size, use_gamma=True)
    # TODO :: re-use lbfgs state from previous problem if size hasn't changed
    subproblem_solver_state = subproblem_solver.init_state(init_params=job_primals_k[0, :].reshape(-1), 
                                                           bounds=primal_bounds, c_is=vmapped_cmat[0, :, :], 
                                                           rki=gpu_vec_k[0, :], xki=job_primals_k[0, :], 
                                                           yki=sum_vec_k[0, :], aug_viol_beta=self.solver_viol_beta, 
                                                           aug_prox_mu=self.solver_prox_mu)
    subproblem_solver_optstep = jaxopt.OptStep(job_primals_k[0].reshape(-1), subproblem_solver_state)

    def vmap_run(init_params, c_is, rki, xki, yki, aug_beta, aug_mu):
      return subproblem_solver.run(init_params=init_params, bounds=primal_bounds,
                                   c_is=c_is, rki=rki, xki=xki, yki=yki, 
                                   aug_viol_beta=aug_beta, aug_prox_mu=aug_mu)
    def hardware_mapper_fun(fun, in_axes, backend):
      if backend=="cpu":
        return jax.pmap(fun, in_axes=in_axes, backend='cpu')
      else:
        return jax.vmap(fun, in_axes=in_axes)
    jax_vmap_fun = hardware_mapper_fun(vmap_run, in_axes=(jaxopt.OptStep(0, 0), 0, 0, 0, 0, None, None), backend=self.solver_backend)
    vmapped_subproblem_state = None
    def init_subproblem_solver():
      rprint(f"Initializing LBFGS state by running one iteration")
      block_sum_xk_0 = jnp.sum(job_primals_k, axis=1)
      y_ks = sum_vec_k - 1 - job_duals_k
      t_k = (Amat @ jnp.sum(block_sum_xk_0, axis=0)) - bvec - gpu_duals_k
      r_ks = t_k - jax.vmap(lambda x: Amat @ x)(block_sum_xk_0) + gpu_slacks_k
      
      # create inputs for vmapped solve
      init_xks = jax.vmap(lambda x: x.reshape(-1), in_axes=(0), out_axes=(0))(job_primals_k)
      rprint(f"Initializing LBFGS state with zero history")
      init_vmapped_optstep = jaxopt.OptStep(init_xks, subproblem_solver_optstep.state)
      vmap_run_broadcast_state = jax.vmap(vmap_run, in_axes=(jaxopt.OptStep(0, None), 0, 0, 0, 0, None, None))
      return vmap_run_broadcast_state(init_vmapped_optstep, vmapped_cmat, r_ks, job_primals_k, 
                                      y_ks, self.solver_viol_beta, self.solver_prox_mu)
    vmapped_subproblem_state = init_subproblem_solver()
    # vmapped_subproblem_state.block_until_ready()

    #### Start solver loop
    reshaped_primals_k = jax.vmap(lambda x: x.reshape(-1, self.num_configs))(job_primals_k)
    stats_k = {
      "gpu_cnstr_viol_norms": jnp.zeros(self.solver_iters_per_sync),
      "sumto1_cnstr_viol_norms": jnp.zeros(self.solver_iters_per_sync),
      "binarization": jnp.zeros(self.solver_iters_per_sync),
      "obj_vals": jnp.zeros(self.solver_iters_per_sync),
    }
    init_loop_state = (reshaped_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, 
                       job_duals_k, vmapped_subproblem_state, stats_k)

    # One iteration of Proximal Jacobi ADMM solver applied to Sia policy (LP or ILP)
    # Args: (k, state) -> state
    # Captures: [vmapped_cmat, Amat, bvec, cnstr_scale_factor, obj_scale_factor, self.num_configs, 
    #            self.solver_viol_beta, self.solver_prox_mu, self.solver_dual_tau]
    # state: (x_ks, u_k, s_k, f_ks, v_ks, vmapped_subproblem_state, stats)
    # x_ks: (num_blocks, block_size, nconfigs) --> job_primals_k
    # u_k: (num_blocks, num_gputypes) --> gpu_duals_k
    # s_k: (num_blocks, block_size) --> job_slacks_k
    # f_ks: (num_blocks, block_size) --> job_slacks_k
    # v_ks: (num_blocks, block_size) --> job_duals_k
    # vmapped_subproblem_state: OptStep --> subproblem_solver_optstep
    # stats: {gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals}
    def solver_loop_body_fun(k, state, perturb=0):
      x_ks, u_k, s_k, f_ks, v_ks, vmapped_subproblem_state, stats = state
      x_ks = x_ks + perturb
      f_ks = f_ks + perturb
      s_k = s_k + perturb
      # compute rki, tk, yki, zki and set parameter vals for x,s problems
      # sum over jobs for each config in a block
      # shape: (num_blocks, nconfigs)
      block_sum_xk_0 = jnp.sum(x_ks, axis=1)
      # sum over configs for each job in a block
      # shape: (num_blocks, block_size)
      block_sum_xk_1 = jnp.sum(x_ks, axis=2)
      # compute vmapped
      z_ks = block_sum_xk_1 - 1 - v_ks
      y_ks = f_ks - 1 - v_ks
      vmapped_tk = (Amat @ jnp.sum(block_sum_xk_0, axis=0)) - bvec - u_k
      r_ks = vmapped_tk - jax.vmap(lambda x: Amat @ x)(block_sum_xk_0) + s_k
      
      # compute x^{k+1}, f^{k+1}
      # create inputs for vmapped solve
      init_xks = jax.vmap(lambda x: x.reshape(-1), in_axes=(0), out_axes=(0))(x_ks)
      vmapped_subproblem_state = jaxopt.OptStep(init_xks, vmapped_subproblem_state.state)
      vmapped_subproblem_state = jax_vmap_fun(vmapped_subproblem_state, vmapped_cmat, r_ks, x_ks, y_ks, 
                                              self.solver_viol_beta, self.solver_prox_mu)
      vmapped_xkp1_res = vmapped_subproblem_state.params
      x_kp1s = jax.vmap(lambda x: x.reshape(-1, self.num_configs), in_axes=(0), out_axes=(0))(vmapped_xkp1_res)
      # replace non-vmapped variants with vmapped
      x_kp1 = x_kp1s.reshape(-1, self.num_configs)
      # compute s^{k+1}
      # s_kp1 = (aug_prox_mu * s_k - aug_viol_beta * t_k) / (aug_prox_mu + aug_viol_beta)
      s_kp1 = -vmapped_tk
      s_kp1 = jnp.clip(s_kp1, 0, bvec)

      # compute f^{k+1}
      f_kp1s = -z_ks
      f_kp1s = jnp.clip(f_kp1s, 0, 1)

      # compute u^{k+1}
      vmapped_sum_xkp1 = jnp.sum(jnp.sum(x_kp1s, axis=1), axis=0)
      vmapped_ukp1 = u_k - self.solver_dual_tau * ((Amat @ vmapped_sum_xkp1) + s_kp1 - bvec)
      u_kp1 = vmapped_ukp1

      # compute v^{k+1}
      # sum_xkp1_tilde = jnp.sum(x_kp1, axis=1)
      # v_kp1 = v_k - dual_tau * (sum_xkp1_tilde + f_kp1 - 1)
      sum_xkp1s_tilde = jnp.sum(x_kp1s, axis=2)
      v_kp1s = v_ks - self.solver_dual_tau * (sum_xkp1s_tilde + f_kp1s - 1)

      # compute stats for iteration
      obj_val = jnp.sum(jnp.multiply(vmapped_cmat, x_kp1s)) * obj_scale_factor
      gpu_cnstr_viol = jnp.sum(jnp.clip((Amat @ vmapped_sum_xkp1 + s_kp1 - bvec) * cnstr_scale_factor, 0, None))
      sumto1_cnstr_viol = (sum_xkp1s_tilde + f_kp1s - 1)
      binarized = jnp.sum(x_kp1 * (1 - x_kp1))
      gpu_cnstr_viol_norm = gpu_cnstr_viol
      sumto1_cnstr_viol_norm = jnp.linalg.norm(sumto1_cnstr_viol).round(3)
      iter_stats = {
        "gpu_cnstr_viol_norms": gpu_cnstr_viol_norm,
        "sumto1_cnstr_viol_norms": sumto1_cnstr_viol_norm,
        "obj_vals": obj_val,
        "binarization": binarized
      }
      stats = jax.tree_map(lambda x, y: x.at[k].set(y), stats, iter_stats)

      # swap state_k=(x_k, u_k, s_k, f_k, v_k) with (x_kp1, u_kp1, s_kp1, f_kp1, v_kp1)
      # under-relaxation (works better than nestrov)
      alpha_k = 1.4
      gamma_k = 1
      u_kp1 = u_k - gamma_k*alpha_k*(u_k - u_kp1)
      s_kp1 = s_k - gamma_k*alpha_k*(s_k - s_kp1)
      x_kp1s = (x_ks - gamma_k*alpha_k*(x_ks - x_kp1s)).round(2)
      f_kp1s = f_ks - gamma_k*alpha_k*(f_ks - f_kp1s)
      v_kp1s = v_ks - gamma_k*alpha_k*(v_ks - v_kp1s)
      x_diff = jnp.linalg.norm(x_kp1s - x_ks)
      # jax.debug.print("Iteration {}: obj_val = {}, gpu_cnstr_viol_norm = {}, sumto1_cnstr_viol_norm={}, x_progress={}", k, obj_val.round(3), gpu_cnstr_viol_norm.round(3), sumto1_cnstr_viol_norm.round(3), x_diff.round(3))

      new_state = (x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s, vmapped_subproblem_state, stats)
      return new_state

    ### Run solver loop
    rprint(f"Compiling prox-jacobi-admm loop")
    # TODO :: cache the compiled function for fixed problem sizes
    jitted_loop_body_fun = jax.jit(solver_loop_body_fun, backend=self.solver_backend).lower(0, init_loop_state, 0.0).compile()
    cost_analysis = jitted_loop_body_fun.cost_analysis()[0]
    bytes_accessed = 0
    for k, v in cost_analysis.items():
      if 'bytes' in k:
        bytes_accessed += v
    flop_to_bytes_ratio = cost_analysis['flops'] / bytes_accessed
    rprint(f"Cost analysis: flops={cost_analysis['flops'] / 1e6} MFLOPs/iter, flop:bytes_accessed ratio={flop_to_bytes_ratio}")
    memory_analysis = jitted_loop_body_fun.memory_analysis()
    rprint(f"Memory analysis: {memory_analysis}")
    rprint(f"Running prox-jacobi-admm loop...")
    final_loop_state = init_loop_state
    iter_times_ms, previous_iter_vals = [], None
    previous_iter_vals, track_stats = None, None
    solve_start_t = time.time()
    perturb_freq = 100000
    max_perturb = 0.001
    has_solver_converged = False
    for i in range(self.solver_max_iters):
      if (i+1) % perturb_freq == 0:
        # perturb = np.random.uniform(0, max_perturb)
        perturb = np.random.uniform(0, max_perturb)
        rprint(f"Slow convergence: Perturbing by {perturb}")
      else:
        perturb = 0.0
      # check for early convergence
      if (i+1) % self.solver_iters_per_sync == 0:
        jax.block_until_ready(final_loop_state)
        job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k, subproblem_solver_state, stats_k = final_loop_state
        if previous_iter_vals is None:
          previous_iter_vals = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k)
          track_stats = jax.tree_map(lambda x: x.copy(), stats_k)
        else:
          # break if progress < tol
          prev_job_primals_k, prev_gpu_duals_k, prev_gpu_slacks_k, prev_job_slacks_k, prev_job_duals_k = previous_iter_vals
          primal_diff = jnp.linalg.norm(job_primals_k - prev_job_primals_k)
          gpu_dual_diff = jnp.linalg.norm(gpu_duals_k - prev_gpu_duals_k)
          gpu_slack_diff = jnp.linalg.norm(gpu_slacks_k - prev_gpu_slacks_k)
          job_slack_diff = jnp.linalg.norm(job_slacks_k - prev_job_slacks_k)
          job_dual_diff = jnp.linalg.norm(job_duals_k - prev_job_duals_k)
          rprint(f"Iteration {i} [t = {(time.time() - solve_start_t)*1000:.2f} ms] :: primal_change={primal_diff:.3f}, gpu_dual_change={gpu_dual_diff:.3f}, gpu_slack_change={gpu_slack_diff:.3f}, job_slack_change={job_slack_diff:.3f}, job_dual_change={job_dual_diff:.3f}")
          rprint(f"\t obj={stats_k['obj_vals'][i-1]:.3f}, gpu_cnstr_viol_norm:{stats_k['gpu_cnstr_viol_norms'][i-1]:.3f}, \
                 job_cnstr_viol_norm:{stats_k['sumto1_cnstr_viol_norms'][i-1]:.3f}, binarization={stats_k['binarization'][i-1]:.3f}")
          gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms = stats_k['gpu_cnstr_viol_norms'], stats_k['sumto1_cnstr_viol_norms']

          track_stats = jax.tree_map(lambda x, y: jnp.append(x, y), track_stats, stats_k)
          if (gpu_cnstr_viol_norms[-1] < self.solver_tol and sumto1_cnstr_viol_norms[-1] < self.solver_tol):
            rprint(f"Breaking after {i+1} iterations : gpu_cntr_viol_norm = {gpu_cnstr_viol_norms[-1]}, sumto1_cnstr_viol_norms = {sumto1_cnstr_viol_norms[-1]} < tol={self.solver_tol}")
            has_solver_converged = True
            break
          else:
            previous_iter_vals = (job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k)
        # append stats to previous iters
      start_t = time.time()
      final_loop_state = jitted_loop_body_fun(i % self.solver_iters_per_sync, final_loop_state, perturb)
      end_t = time.time()
      iter_time = end_t - start_t
      iter_times_ms.append(iter_time*1000)
    jax.block_until_ready(final_loop_state)
    solve_end_t = time.time()

    ### Post-processing solver output
    job_primals_k, gpu_duals_k, gpu_slacks_k, job_slacks_k, job_duals_k, subproblem_solver_state, stats_k = final_loop_state
    num_iters = len(iter_times_ms)
    rprint(f"Prox-Jacobi-ADMM: finished {num_iters} iterations in {(solve_end_t - solve_start_t)*1000:.2f} ms")
    job_primals = np.squeeze(np.array(job_primals_k.reshape(-1, self.num_configs)))
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

    # extract allocations only if solver has converged
    cluster_free_gpus = {cluster: cluster_max_gpus for cluster, cluster_max_gpus in zip(self.cluster_ordering, self.max_ngpus)}
    partial_allocs = {}
    partial_allocs_obj_val = 0
    for i, jobname in enumerate(self.job_ordering):
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