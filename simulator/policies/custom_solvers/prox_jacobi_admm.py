import jaxopt.tree_util
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
from jaxopt import loop
from functools import partial
from jaxopt import LBFGSB
from rich import print as rprint
import time
from jaxopt._src.lbfgsb import LbfgsbState

jax_backend='cpu'
jax.devices(jax_backend)
# jax.config.update('jax_log_compiles', True)
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

def jax_solve_sia_separable_lp_using_prox_jacobi_admm_box(prob, solver_params, 
                                                          init_x = None, init_dual = None,
                                                          init_state = None):
  # jax.config.update('jax_log_compiles', True)
  # rprint(f"Init dual: {init_dual}")
  st_time = time.time()
  # extract problem def
  cs, A, b = prob['cs'], prob['G'].copy(), prob['g'].copy()
  c_is = [c.copy() for c in cs]
  c_ismat = jnp.array(c_is)
  Amat = jnp.array(A)
  bvec = jnp.array(b)
  ngpu_types, nconfigs = A.shape
  njobs = len(c_is)
  valid_njobs = njobs

  # extract solver params
  num_blocks = solver_params.get('num_blocks', njobs)
  block_size = solver_params.get('block_size', -1)
  if block_size == -1:
    block_size = np.ceil(njobs * 1.0 / num_blocks)
  # add dummy jobs with +1 cost across all configs
  if njobs % block_size != 0:
    rprint(f"Adding {block_size - (njobs % block_size)} dummy jobs")
    while njobs % block_size != 0 :
      njobs += 1
      c_is.append(np.ones(nconfigs))
    c_ismat = jnp.array(c_is)
  num_blocks = int(njobs // block_size)
  rprint(f"Using block size = {block_size} jobs/block, num blocks = {num_blocks}")

  max_itr = solver_params['max_itr']
  iters_per_sync = solver_params.get('iters_per_sync', None)
  if iters_per_sync == -1 or iters_per_sync is None:
    iters_per_sync = max_itr
  tol = solver_params.get('tol', 1e-2)
  aug_viol_beta = solver_params['aug_viol_beta']
  aug_prox_mu = solver_params['aug_prox_mu']
  dual_tau = solver_params.get('dual_tau', 1)
  normalize_A = solver_params.get('normalize_A', False)
  normalize_c = solver_params.get('normalize_c', False)
  use_only_box_constraints = solver_params.get('use_only_box_constraints', False)
  float_dtype = np.float32 if solver_params.get('use_float32', False) else np.float64
  require_binary_solutions = solver_params.get('require_binary', False)
  assert not use_only_box_constraints, "This solver does not support using only box constraints.."
  if normalize_A:
    cnstr_scale_factor = b
    rprint(f"Scaling down A, b by {cnstr_scale_factor}")
    A = A / cnstr_scale_factor.reshape(-1, 1)
    Amat = Amat / cnstr_scale_factor.reshape(-1, 1)
    b = b / cnstr_scale_factor
    bvec = bvec / cnstr_scale_factor
  obj_scale_factor = 1
  if normalize_c:
    max_c = jnp.max(jnp.abs(c_is[0]))
    for i in range(valid_njobs):
      max_c = max(max_c, jnp.max(jnp.abs(c_is[i])))
    obj_scale_factor = max_c
    rprint(f"Scaling down c by {max_c}")
    for i in range(valid_njobs):
      c_is[i] = (c_is[i] / obj_scale_factor)
    c_ismat = c_ismat / obj_scale_factor

  # initialize variables for k iteration
  # x_k: primal variables at iteration k
  x_k = jnp.zeros((njobs, nconfigs), dtype=float_dtype)
  if init_x is not None:
    x_k = x_k.at[:valid_njobs, :].set(init_x[:, :])
  # u_k: scaled dual variables at iteration k
  u_k = jnp.zeros((ngpu_types, ), dtype=float_dtype)
  # s_k: slack variables at iteration k
  s_k = jnp.array(bvec - Amat @ jnp.sum(x_k, axis=0))
  # f_k: slack variables for sum-to-1 constraints
  f_k = jnp.zeros(njobs, dtype=float_dtype)
  # f_k = np.zeros(njobs, dtype=float_dtype)
  # v_k: dual variables for sum-to-1 constraints
  v_k = jnp.zeros(njobs, dtype=float_dtype)
  if init_dual is not None:
    init_u, init_v = init_dual
    u_k = u_k.at[:valid_njobs].set(init_u[:])
    v_k = v_k.at[:valid_njobs].set(init_v[:])
  rprint(f"Initial slack: {s_k * cnstr_scale_factor}")

  # partition jobs into blocks
  partitions = jnp.array_split(jnp.arange(njobs), num_blocks)
  partitions_complement = [jnp.setdiff1d(jnp.arange(njobs), p, assume_unique=True) for p in partitions]
  partition_valid_jobs = [block_size] * (num_blocks - 1)
  partition_valid_jobs.append(valid_njobs - np.sum(partition_valid_jobs))
  rprint(f"Partitioned {njobs} jobs into {num_blocks} blocks: {partitions}")
  rprint(f"Valid # jobs in each block: {partition_valid_jobs}")

  # kth iteration for vmap use
  vmapped_cmats = jnp.array([c_ismat[part_idxs, :] for part_idxs in partitions])
  x_ks = jnp.array([x_k[part_idxs, :] for part_idxs in partitions])
  f_ks = jnp.array([f_k[part_idxs] for part_idxs in partitions])
  v_ks = jnp.array([v_k[part_idxs] for part_idxs in partitions])

  # initialize variables for (k+1) iteration
  x_kp1 = jnp.zeros_like(x_k)
  u_kp1 = jnp.zeros_like(u_k)
  s_kp1 = jnp.zeros_like(s_k)
  f_kp1 = jnp.zeros_like(f_k)
  v_kp1 = jnp.zeros_like(v_k)
  # for vmap use
  x_kp1s = jnp.array([jnp.zeros_like(x_k[part_idxs, :]) for part_idxs in partitions])
  f_kp1s = jnp.array([jnp.zeros_like(f_k[part_idxs]) for part_idxs in partitions])
  v_kp1s = jnp.array([jnp.zeros_like(v_k[part_idxs]) for part_idxs in partitions])

  # initialize parameters for nestrov acceleration
  alpha_k = 1
  alpha_kp1 = 1

  # additional vars for kth iteration
  r_k = jnp.zeros((num_blocks, ngpu_types), dtype=float_dtype)
  t_k = jnp.zeros((ngpu_types, ), dtype=float_dtype)
  y_k = jnp.zeros((njobs, ), dtype=float_dtype)
  z_k = jnp.zeros((njobs, ), dtype=float_dtype)
  # for vmap use
  r_ks = jnp.array([jnp.zeros((ngpu_types,), dtype=float_dtype) for _ in partitions])
  y_ks = jnp.array([jnp.zeros_like(y_k[part_idxs]) for part_idxs in partitions])
  z_ks = jnp.array([jnp.zeros_like(z_k[part_idxs]) for part_idxs in partitions])

  # trackers
  obj_vals = []
  iter_times_ms = []

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
  def xi_fun_pytree(xikp1, c_is, rki, xki, yki, aug_viol_beta, aug_prox_mu):
    # reshape xikp1 to (-1, nconfigs)
    reshaped_xikp1 = jax.tree_map(lambda x: x.reshape(-1, nconfigs), xikp1)
    ret = jaxopt.tree_util.tree_vdot(c_is, reshaped_xikp1)
    gpu_cnstr_viol = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, rki: Amat @ jnp.sum(x, axis=0) + rki, reshaped_xikp1, rki), squared=True)
    ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2), gpu_cnstr_viol)
    sumto_1_cnstr_viol = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, yki: jnp.sum(x, axis=1) + yki, reshaped_xikp1, yki), squared=True)
    ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2), sumto_1_cnstr_viol)
    prox_term = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, y: x - y, reshaped_xikp1, xki), squared=True)
    ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_prox_mu / 2), prox_term)
    # binarization = x * (1 - x)
    if require_binary_solutions:
      rhs = jaxopt.tree_util.tree_add_scalar_mul(1, -1, reshaped_xikp1)
      binarization = jaxopt.tree_util.tree_vdot(reshaped_xikp1, rhs)
      ret = jaxopt.tree_util.tree_add_scalar_mul(ret, 1e-3, binarization)
    return ret
  
  lbfgs_max_iters = 15
  jax_xproblem = LBFGSB(fun=xi_fun_pytree, jit=True, unroll=True, implicit_diff=False,
                        maxiter=lbfgs_max_iters, tol=1e-4, stepsize=0.8, history_size=10,
                        use_gamma=True)
  '''
  Initialize solver state, JIT compile JAX functions
  '''
  init_lbfgs_state = None
  init_state = None
  if init_state is not None:
    rprint(f"Reusing LBFGS state from previous run")
    # carry over lbfgs state from previous run
    # set value, grad and error to None (not used inside init_state)
    init_state = init_state.state
    reinit_lbfgs_state = LbfgsbState(iter_num=init_state.iter_num,
                                    s_history=init_state.s_history,
                                    y_history=init_state.y_history,
                                    stepsize=init_state.stepsize,
                                    num_updates=init_state.num_updates,
                                    theta=init_state.theta,
                                    value=None, grad=None, error=None)
    init_lbfgs_state = jaxopt.OptStep(x_ks[0, :].reshape(-1), reinit_lbfgs_state)
  else:
    rprint(f"Initializing LBFGS state with zero history")
    init_lbfgs_state = x_ks[0, :].reshape(-1)
  bounds = (jnp.zeros_like(x_ks[0, :].reshape(-1)), jnp.ones_like(x_ks[0, :].reshape(-1)))
  rprint(f"Initializing LBFGS-B state")
  lbfgs_state = jax_xproblem.init_state(init_params=init_lbfgs_state, bounds=bounds, 
                                        c_is=vmapped_cmats[0, :, :], rki=r_ks[0, :], xki=x_ks[0, :], 
                                        yki=y_ks[0, :], aug_viol_beta=aug_viol_beta, aug_prox_mu=aug_prox_mu)
  lbfgs_optstep = jaxopt.OptStep(x_ks[0].reshape(-1), lbfgs_state)
  rprint(f"Compiling LBFGS-B run function")
  # compiled_xikp1_fun = jax.jit(jax_xproblem.run, backend=jax_backend)
  compiled_xikp1_fun = jax_xproblem.run
  # cost_analysis = compiled_xikp1_fun.cost_analysis()
  # rprint(f"Cost analysis: flops={cost_analysis[0]['flops']}")
  # memory_analysis = compiled_xikp1_fun.memory_analysis()
  # rprint(f"Memory analysis: {memory_analysis}")
  def vmap_func(init_params, c_is, rki, xki, yki, aug_beta, aug_mu):
    return compiled_xikp1_fun(init_params=init_params, bounds=bounds, 
                              c_is=c_is, rki=rki, xki=xki, yki=yki, 
                              aug_viol_beta=aug_beta, aug_prox_mu=aug_mu)
  jax_vmap_fun_broadcast_state = jax.vmap(vmap_func, in_axes=(jaxopt.OptStep(0, None), 0, 0, 0, 0, None, None))
  jax_vmap_fun = jax.pmap(vmap_func, in_axes=(jaxopt.OptStep(0, 0), 0, 0, 0, 0, None, None))
  init_time = time.time() - st_time
  it_solve_time, it_setup_time = 0, 0
  vmapped_lbfgs_optstep = None
  def init_lbfgs_optstep():
    rprint(f"Initializing LBFGS state by running one iteration")
    block_sum_xk_0 = jnp.sum(x_ks, axis=1)
    y_ks = f_ks - 1 - v_ks
    t_k = (Amat @ jnp.sum(block_sum_xk_0, axis=0)) - bvec - u_k
    r_ks = t_k - jax.vmap(lambda x: Amat @ x)(block_sum_xk_0) + s_k
    
    # create inputs for vmapped solve
    init_xks = jax.vmap(lambda x: x.reshape(-1), in_axes=(0), out_axes=(0))(x_ks)
    rprint(f"Initializing LBFGS state with zero history")
    vmapped_lbfgs_optstep = jaxopt.OptStep(init_xks, lbfgs_optstep.state)
    vmapped_lbfgs_optstep = jax_vmap_fun_broadcast_state(vmapped_lbfgs_optstep, vmapped_cmats, 
                                                             r_ks, x_ks, y_ks, aug_viol_beta, aug_prox_mu)
    return vmapped_lbfgs_optstep
  vmapped_lbfgs_optstep = init_lbfgs_optstep()
  x_ks = jax.vmap(lambda x: x.reshape(-1, nconfigs))(vmapped_lbfgs_optstep.params)
  gpu_cnstr_viol_norms = jnp.zeros(max_itr)
  sumto1_cnstr_viol_norms = jnp.zeros(max_itr)
  binarization = jnp.zeros(max_itr)
  obj_vals = jnp.zeros(max_itr)
  iter_times_ms = jnp.zeros(max_itr)
  stats = (gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals)
  init_loop_state = (x_ks, u_k, s_k, f_ks, v_ks, vmapped_lbfgs_optstep, stats)
  # start prox-jacobi-admm iteration
  def loop_body_fun(k, state):
    x_ks, u_k, s_k, f_ks, v_ks, vmapped_lbfgs_optstep, stats = state
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
    vmapped_lbfgs_optstep = jaxopt.OptStep(init_xks, vmapped_lbfgs_optstep.state)
    vmapped_lbfgs_optstep = jax_vmap_fun(vmapped_lbfgs_optstep, vmapped_cmats, r_ks, x_ks, y_ks, 
                                            aug_viol_beta, aug_prox_mu)
    vmapped_xkp1_res = vmapped_lbfgs_optstep.params
    x_kp1s = jax.vmap(lambda x: x.reshape(-1, nconfigs), in_axes=(0), out_axes=(0))(vmapped_xkp1_res)
    # replace non-vmapped variants with vmapped
    x_kp1 = x_kp1s.reshape(-1, nconfigs)
    # compute s^{k+1}
    # s_kp1 = (aug_prox_mu * s_k - aug_viol_beta * t_k) / (aug_prox_mu + aug_viol_beta)
    s_kp1 = -vmapped_tk
    s_kp1 = jnp.clip(s_kp1, 0, b)

    # compute f^{k+1}
    f_kp1s = -z_ks
    f_kp1s = jnp.clip(f_kp1s, 0, 1)

    # compute u^{k+1}
    vmapped_sum_xkp1 = jnp.sum(jnp.sum(x_kp1s, axis=1), axis=0)
    vmapped_ukp1 = u_k - dual_tau * ((Amat @ vmapped_sum_xkp1) + s_kp1 - bvec)
    u_kp1 = vmapped_ukp1

    # compute v^{k+1}
    # sum_xkp1_tilde = jnp.sum(x_kp1, axis=1)
    # v_kp1 = v_k - dual_tau * (sum_xkp1_tilde + f_kp1 - 1)
    sum_xkp1s_tilde = jnp.sum(x_kp1s, axis=2)
    v_kp1s = v_ks - dual_tau * (sum_xkp1s_tilde + f_kp1s - 1)

    # add to trackers
    obj_val = jnp.sum(jnp.multiply(c_ismat, x_kp1)) * obj_scale_factor
    gpu_cnstr_viol = jnp.sum(jnp.clip((Amat @ vmapped_sum_xkp1 + s_kp1 - bvec) * cnstr_scale_factor, 0, None))
    sumto1_cnstr_viol = (sum_xkp1s_tilde + f_kp1s - 1)
    binarized = jnp.sum(x_kp1 * (1 - x_kp1))
    gpu_cnstr_viol_norm = gpu_cnstr_viol
    sumto1_cnstr_viol_norm = jnp.linalg.norm(sumto1_cnstr_viol).round(3)
    gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals = stats
    gpu_cnstr_viol_norms = gpu_cnstr_viol_norms.at[k].set(gpu_cnstr_viol_norm)
    sumto1_cnstr_viol_norms = sumto1_cnstr_viol_norms.at[k].set(sumto1_cnstr_viol_norm)
    obj_vals = obj_vals.at[k].set(obj_val)
    binarization = binarization.at[k].set(binarized)
    stats = (gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals)

    # swap (x_k, u_k, s_k, f_k, v_k) with (x_kp1, u_kp1, s_kp1, f_kp1, v_kp1)

    # under-relaxation (works better than nestrov)
    alpha_k = 1
    gamma_k = 1.2
    u_kp1 = u_k - gamma_k*alpha_k*(u_k - u_kp1)
    s_kp1 = s_k - gamma_k*alpha_k*(s_k - s_kp1)
    x_kp1s = (x_ks - gamma_k*alpha_k*(x_ks - x_kp1s)).round(2)
    f_kp1s = f_ks - gamma_k*alpha_k*(f_ks - f_kp1s)
    v_kp1s = v_ks - gamma_k*alpha_k*(v_ks - v_kp1s)
    x_diff = jnp.linalg.norm(x_kp1s - x_ks)
    # jax.debug.print("Iteration {}: obj_val = {}, gpu_cnstr_viol_norm = {}, sumto1_cnstr_viol_norm={}, x_progress={}", k, obj_val.round(3), gpu_cnstr_viol_norm.round(3), sumto1_cnstr_viol_norm.round(3), x_diff.round(3))

    new_state = (x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s, vmapped_lbfgs_optstep, stats)
    return new_state
  
  # run the loop
  rprint(f"Compiling prox-jacobi-admm loop")
  jitted_loop_body_fun = jax.jit(loop_body_fun, backend=jax_backend).lower(0, init_loop_state).compile()
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
  iter_times_ms = []
  last_values = None
  t_start_solve = time.time()
  for i in range(max_itr):
    # check for early convergence
    if (i+1) % iters_per_sync == 0:
      jax.block_until_ready(final_loop_state)
      x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s, lbfgs_state, stats = final_loop_state
      gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals = stats
      if last_values is None:
        last_values = (x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s)
      else:
        # break if progress < tol
        prev_x_kp1s, prev_u_kp1, prev_s_kp1, prev_f_kp1s, prev_v_kp1s = last_values
        x_diff = jnp.linalg.norm(x_kp1s - prev_x_kp1s)
        u_diff = jnp.linalg.norm(u_kp1 - prev_u_kp1)
        s_diff = jnp.linalg.norm(s_kp1 - prev_s_kp1)
        f_diff = jnp.linalg.norm(f_kp1s - prev_f_kp1s)
        v_diff = jnp.linalg.norm(v_kp1s - prev_v_kp1s)
        rprint(f"Iteration {i} [t = {(time.time() - t_start_solve)*1000:.2f} ms] :: x_progress={x_diff:.3f}, u_progress={u_diff:.3f}, s_progress={s_diff:.3f}, f_progress={f_diff:.3f}, v_progress={v_diff:.3f}")
        rprint(f"\t obj_val = {obj_vals[i-1]:.3f}, gpu_cnstr_viol_norm:{gpu_cnstr_viol_norms[i-1]:.3f}, \
               sumto1_cnstr_viol_norm:{sumto1_cnstr_viol_norms[i-1]:.3f}, binarization={binarization[i-1]:.3f}")
        if (gpu_cnstr_viol_norms[i-1] < tol and sumto1_cnstr_viol_norms[i-1] < tol):
          rprint(f"Breaking after {i+1} iterations : gpu_cntr_viol_norm = {gpu_cnstr_viol_norms[i-1]}, sumto1_cnstr_viol_norms = {sumto1_cnstr_viol_norms[i-1]}")
          break
        else:
          last_values = (x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s)
    start_t = time.time()
    final_loop_state = jitted_loop_body_fun(i, final_loop_state)
    end_t = time.time()
    iter_time = end_t - start_t
    iter_times_ms.append(iter_time*1000)

  # final_loop_state = jax.lax.fori_loop(0, max_itr, loop_body_fun, init_loop_state, unroll=5)
  jax.block_until_ready(final_loop_state)
  t_end_solve = time.time()
  x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s, lbfgs_state, stats = final_loop_state
  gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals = stats
  # rprint("Iteration {}: obj_val = {}, gpu_cnstr_viol_norm = {}, sumto1_cnstr_viol_norm={}, x_progress={}".format(i, obj_vals[i].round(3), gpu_cnstr_viol_norms[i].round(3), sumto1_cnstr_viol_norms[i].round(3), iter_times_ms[i].round(3)))

  iter_times_ms = np.array(iter_times_ms)
  num_iters = len(iter_times_ms)
  rprint(f"Resizing stats to {num_iters} iterations")
  rprint(f"Prox-Jacobi-ADMM: finished {num_iters} iterations in {(t_end_solve - t_start_solve)*1000:.2f} ms")
  obj_vals = np.array(obj_vals[:num_iters])
  gpu_cnstr_viol_norms = np.array(gpu_cnstr_viol_norms[:num_iters])
  sumto1_cnstr_viol_norms = np.array(sumto1_cnstr_viol_norms[:num_iters])
  iter_times_ms = np.array(iter_times_ms[:num_iters])
  
  '''
  for i in range(num_iters):
    rprint(f"Iteration {i}: gpu_cnstr_viol_norm = {gpu_cnstr_viol_norms[i]:.2f}, sumto1_cnstr_viol_norm = {sumto1_cnstr_viol_norms[i]:.3f}, obj_val = {obj_vals[i]:.3f}, binarization={binarization[i]:.3f}, time = {iter_times_ms[i]:.2f}ms")
  '''

  # extract final state
  x_kp1 = np.array(x_kp1s.reshape(-1, nconfigs))
  f_kp1 = np.array(f_kp1s.reshape(-1,))
  v_kp1 = np.array(v_kp1s.reshape(-1,))
  u_kp1 = np.array(u_kp1)
  s_kp1 = np.array(s_kp1)
  rprint(f"Final slack: {(s_k * cnstr_scale_factor).round(3)}")
  final_state = {
    'x' : np.array(x_kp1[:valid_njobs, :]),
    'u' : np.array(u_kp1),
    's' : np.array(s_kp1),
    'f' : np.array(f_kp1[:valid_njobs]),
    'v' : np.array(v_kp1[:valid_njobs]),
    'solver_state' : lbfgs_state,
  }
  rprint(f"sum(allocation) for dummy jobs = {jnp.sum(x_kp1[valid_njobs:].reshape(-1))}")

  track_state = {
    'obj_vals' : obj_vals,
    'iter_times_ms': iter_times_ms,
    'gpu_cnstr_viol_norms': gpu_cnstr_viol_norms,
    'sumto1_cnstr_viol_norms': sumto1_cnstr_viol_norms,
  }

  if normalize_A:
    # scale-back slack variables
    final_state['s'] = final_state['s'] * cnstr_scale_factor
    # for i in range(len(track_state['slack_vals'])):
    #   s, f = track_state['slack_vals'][i]
    #   track_state['slack_vals'][i] = (s * cnstr_scale_factor, f)

  rprint(f"Solver timings:")
  rprint(f"Init time: {init_time:.2f}s")
  rprint(f"Total time: {np.sum(iter_times_ms):.2f}ms")
  # rprint(f"Avg. skipped subproblems: {np.mean([len(s) for s in skipped_subproblems]):.2f} / {num_blocks}")
  # return
  return final_state, track_state