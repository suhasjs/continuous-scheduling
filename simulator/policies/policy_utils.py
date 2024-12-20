# import jaxopt
# import jax
# import jax.numpy as jnp

def round_allocations_largest(partial_allocations, cluster_free_gpus):
  # allocate the largest possible config to each job
  rounded_allocs = {}
  for jobname, partial_alloc in partial_allocations.items():
    alloced_gpus = 0
    for config, _ in partial_alloc:
      _, ngpus, cluster = config
      if cluster_free_gpus[cluster] >= ngpus:
        rounded_allocs[jobname] = config
        cluster_free_gpus[cluster] -= ngpus
        alloced_gpus = ngpus
        break
    if alloced_gpus == 0:
      rounded_allocs[jobname] = None
  return rounded_allocs

'''
    Augmented Lagrangian:
    L(x, s, f, u, v) = sum_i { c_i^T x_i } +                                  <--- objective
                      beta/2 * || A * sum_i (x_i) + s - b + u ||_2^2 +       <--- GPU constraint violation
                      beta/2 * sum_i (|| x_i^T * 1 + f_i - 1 + v_i ||_2^2) + <--- sum-to-1 constraint violation
    where x = [x_1, x_2, ..., x_n],         <--- primal variables, one for each (job, config) pair
          s = [s_1, s_2, ..., s_m],         <--- slack variables, one for each GPU type
          f = [f_1, f_2, ..., f_n]          <--- slack variables, one for each job
          u = [u_1, u_2, ..., u_m]          <--- dual variables, one for each GPU type
          v = [v_1, v_2, ..., v_n]          <--- dual variables, one for each job
    subject to 0 <= x_i <= 1 for all i

    ### Prox Jacobi ADMM Iteration (k -> k+1):
    (1) x_i^{k+1} = argmin_{x_i} { L(x_j^{k}, s^{k}, f^{k}, u^{k}, v^{k}) +  <--- AL objective
                                  (mu/2) * || x_i - x_i^k ||_2^2               <--- proximal term
                                }
      where r_i^k = (A * (sum_{j != i} x_j^{k}) + s^k - b + u^k)        --> jacobi decomposition of GPU constraints
            y_i^k = (f_i^k - 1 + v_i^k)                                 --> jacobi decomposition of sum-to-1 constraints
      subject to 0 <= x_i <= 1 for all i                                --> box constraints on x_i
    (2) s^{k+1} = argmin_{s} { L(x_i^{k}, s, f^{k}, u^{k}, v_i^{k}) + <--- AL objective
                              (mu/2) * || s - s^k ||_2^2                <--- proximal term
                            }
      where t^k = (sum_i (A * x_i^{k}) - b + u^k)
      We can compute s^{k+1} in closed form as:
      s^{k+1} = max(0, (mu * s^k - beta * t^k) / (mu + beta))
    (3) f_i^{k+1} = argmin_{f_i} { L(x_i^{k}, s^{k}, f_i, u^{k}, v_i^{k}) + <--- AL objective
                                  (mu/2) * || f_i - f_i^k ||_2^2              <--- proximal term
                                }
      where z_i^k = (x_i^{k}^T * 1 - 1 + v_i^k)
      We can compute f_i^{k+1} in closed form as:
      f_i^{k+1} = max(0, (mu * f_i^k - beta * z_i^k) / (mu + beta))
    (4) u^{k+1} = u^k + tau * (A * sum_i x_i^{k+1} + s^{k+1} - b)         <--- dual ascent update for GPU constraints
    (5) v_i^{k+1} = v_i^k + tau * (x_i^{k+1}^T*1 + f_i^{k+1} - 1) <--- dual ascent update for sum-to-1 constraints
'''
# Captures: num_configs, lambda_no_alloc, Amat, bvec
# Augmented Lagrangian for the Sia ILP/LP-relaxation policy
def sia_auglag_fun(xikp1, c_is, rki, xki, yki, aug_viol_beta, aug_prox_mu, aug_bin_lambda, 
                   lambda_no_alloc, Amat, block_size, num_configs):
  # reshape xikp1 to (-1, nconfigs)
  reshaped_xikp1 = jax.tree_map(lambda x: x.reshape(block_size, num_configs), xikp1)
  # add <c_i, x_i> term --> cost (negative of utility)
  ret = jaxopt.tree_util.tree_vdot(c_is, reshaped_xikp1)
  # add -lambda_no_alloc * sum_i (x_i) term --> incentivize allocation (penalize no allocation)
  scalar_add_val = jax.tree_map(lambda x: jnp.sum(x), reshaped_xikp1)
  ret = jaxopt.tree_util.tree_add_scalar_mul(ret, -1 * lambda_no_alloc, scalar_add_val)
  # add beta/2 * || A * sum_i (x_i) + s - b - u ||_2^2 term --> GPU constraint violation
  A_norm = jnp.linalg.norm(Amat)
  gpu_cnstr_viol = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, rki: Amat @ jnp.sum(x, axis=0) + rki, reshaped_xikp1, rki), squared=True)
  ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2) * (1 / A_norm), gpu_cnstr_viol)
  # ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2), gpu_cnstr_viol)
  # add beta/2 * || sum (x_i) + f - 1 - v ||_2^2 term --> sum-to-1 constraint violation
  sumto_1_cnstr_viol = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, yki: jnp.sum(x, axis=1) + yki, reshaped_xikp1, yki), squared=True)
  ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_viol_beta / 2), sumto_1_cnstr_viol)
  # add mu/2 * || x - x^k ||_2^2 term --> proximal term for primals
  prox_term = jaxopt.tree_util.tree_l2_norm(jax.tree_map(lambda x, y: x - y, reshaped_xikp1, xki), squared=True)
  ret = jaxopt.tree_util.tree_add_scalar_mul(ret, (aug_prox_mu / 2), prox_term)
  # binarization = x * (1 - x) --> penalize non-binary solutions if required
  rhs = jaxopt.tree_util.tree_add_scalar_mul(1, -1, reshaped_xikp1)
  binarization = jaxopt.tree_util.tree_vdot(reshaped_xikp1, rhs)
  ret = jaxopt.tree_util.tree_add_scalar_mul(ret, aug_bin_lambda, binarization)
  return ret

# Initialize an LBFGS-B solver for the primal subproblem in Prox Jacobi ADMM
def initialize_subproblem_solver(lbfgs_solver_params, init_params, auglag_other_params):
  # extract params
  job_primals_k, job_primal_bounds = init_params["job_primals_k"], init_params["job_primal_bounds"]
  job_primals_k = jax.device_put(job_primals_k, jax.devices(backend='cpu')[0])
  job_duals_k, gpu_duals_k = init_params["job_duals_k"], init_params["gpu_duals_k"]
  job_slacks_k, gpu_slacks_k = init_params["job_slacks_k"], init_params["gpu_slacks_k"]
  gpu_vec_k, sum_vec_k = init_params["gpu_vec_k"], init_params["sum_vec_k"]
  vmapped_cmat = init_params["vmapped_cmat"]
  lbfgs_max_iters, lbfgs_history_size = lbfgs_solver_params["max_iters"], lbfgs_solver_params["history_size"]
  solver_viol_beta, solver_prox_mu = lbfgs_solver_params["viol_beta"], lbfgs_solver_params["prox_mu"]
  solver_bin_lambda = lbfgs_solver_params["bin_lambda"]
  solver_backend = lbfgs_solver_params["solver_backend"]

  specialized_auglag_fun = jax.jit(jax.tree_util.Partial(sia_auglag_fun, **auglag_other_params), backend=solver_backend)

  #### Initialize LBFGS-B solver
  subproblem_solver = jaxopt.LBFGSB(fun=specialized_auglag_fun, jit=True, unroll=False, implicit_diff=False,
                                    maxiter=lbfgs_max_iters, tol=5e-3, stepsize=0.9, 
                                    history_size=lbfgs_history_size, use_gamma=False)
  # Initialize subproblem state for one block
  subproblem_solver_state = subproblem_solver.init_state(init_params=job_primals_k[0, :].reshape(-1), 
                                                          bounds=job_primal_bounds, c_is=vmapped_cmat[0, :, :], 
                                                          rki=gpu_vec_k[0, :], xki=job_primals_k[0, :], 
                                                          yki=sum_vec_k[0, :], aug_viol_beta=solver_viol_beta, 
                                                          aug_prox_mu=solver_prox_mu, aug_bin_lambda=solver_bin_lambda)
  subproblem_solver_optstep = jaxopt.OptStep(job_primals_k[0].reshape(-1), subproblem_solver_state)
  # create inputs for vmapped solve
  init_xks = jax.vmap(lambda x: x.reshape(-1), in_axes=(0), out_axes=(0))(job_primals_k)
  init_vmapped_optstep = jaxopt.OptStep(init_xks, subproblem_solver_optstep.state)
  vmap_broadcast_in_axes = (jaxopt.OptStep(0, None), (None, None), 0, 0, 0, 0, None, None, None)
  broadcast_args = (init_vmapped_optstep, job_primal_bounds, vmapped_cmat, gpu_vec_k, job_primals_k, 
                    sum_vec_k, solver_viol_beta, solver_prox_mu, solver_bin_lambda)
  if solver_backend == "cpu":
    vmapped_lbfgsb_state= jax.pmap(subproblem_solver.run, in_axes=vmap_broadcast_in_axes, backend="cpu")(*broadcast_args)
    # vmapped_lbfgsb_state= jax.vmap(subproblem_solver.run, in_axes=vmap_broadcast_in_axes)(*broadcast_args)
  else:
    vmapped_lbfgsb_state= jax.vmap(subproblem_solver.run, in_axes=vmap_broadcast_in_axes)(*broadcast_args)
  return subproblem_solver, vmapped_lbfgsb_state

# One iteration of Proximal Jacobi ADMM solver applied to Sia policy (LP or ILP)
# Args: (k, state) -> state
# Captures: [vmapped_cmat, Amat, bvec, cnstr_scale_factor, obj_scale_factor, self.num_configs, 
#            self.solver_viol_beta, self.solver_prox_mu, self.solver_dual_tau]
# state: (x_ks, u_k, s_k, f_ks, v_ks, vmapped_lbfgsb_state, stats)
# x_ks: (num_blocks, block_size, nconfigs) --> job_primals_k
# u_k: (num_blocks, num_gputypes) --> gpu_duals_k
# s_k: (num_blocks, block_size) --> job_slacks_k
# f_ks: (num_blocks, block_size) --> job_slacks_k
# v_ks: (num_blocks, block_size) --> job_duals_k
# vmapped_lbfgsb_state: OptStep --> subproblem_solver_optstep
# stats: {gpu_cnstr_viol_norms, sumto1_cnstr_viol_norms, binarization, obj_vals}
def pjadmm_iter_fun(k, state, problem_args, iter_args, 
                    block_size, num_configs, subproblem_solver, solver_backend="cpu"):
  # problem args
  Amat, bvec = problem_args["Amat"], problem_args["bvec"]
  vmapped_cmat = problem_args["vmapped_cmat"]
  lambda_no_alloc = problem_args["lambda_no_alloc"]
  job_primal_bounds = problem_args["job_primal_bounds"]
  cnstr_scale_factor, obj_scale_factor = problem_args["cnstr_scale_factor"], problem_args["obj_scale_factor"]

  # iter args
  solver_viol_beta, solver_prox_mu = iter_args["solver_viol_beta"], iter_args["solver_prox_mu"]
  solver_bin_lambda = iter_args["solver_bin_lambda"]
  solver_dual_tau = iter_args["solver_dual_tau"]

  x_ks, u_k, s_k, f_ks, v_ks, vmapped_lbfgsb_state, other_state, stats = state
  # compute rki, tk, yki, zki and set parameter vals for x,s problems
  # sum over jobs for each config in a block
  # shape: (num_blocks, nconfigs)
  block_sum_xk_1 = jnp.sum(x_ks, axis=1)
  # sum over configs for each job in a block
  # shape: (num_blocks, block_size)
  block_sum_xk_2 = jnp.sum(x_ks, axis=2)
  # compute vmapped
  z_ks = block_sum_xk_2 - 1 + v_ks
  # jax.debug.print("Shape(z_ks): {zksshape}", zksshape=z_ks.shape)
  y_ks = f_ks - 1 + v_ks
  # jax.debug.print("Shape(f_ks): {fksshape}", fksshape=f_ks.shape)
  # jax.debug.print("Shape(y_ks): {yksshape}", yksshape=y_ks.shape)
  # t^k = (sum_i (A * x_i^{k}) - b + u^k)
  t_k = (Amat @ jnp.sum(block_sum_xk_1, axis=0)) - bvec + u_k
  # jax.debug.print("Shape(t_ks): {tksshape}", tksshape=t_k.shape)
  # r_i^k = (A * (sum_{j != i} x_j^{k}) + s^k - b + u^k) --> i = block ID
  r_ks = t_k + s_k - jax.vmap(lambda x: Amat @ x)(block_sum_xk_1)
  # jax.debug.print("Shape(r_ks): {rksshape}", rksshape=r_ks.shape)
  
  #### Compute x^{k+1} with LBFGS-B solver #####
  # create inputs for vmapped solve
  init_xks = jax.vmap(lambda x: x.reshape(-1), in_axes=(0), out_axes=(0))(x_ks)
  # jax.debug.print("Shape(init_xks): {ixksshape}", ixksshape=init_xks.shape)
  vmapped_lbfgsb_state = jaxopt.OptStep(init_xks, vmapped_lbfgsb_state.state)
  vmapped_args = (vmapped_lbfgsb_state, job_primal_bounds, vmapped_cmat, r_ks, x_ks, y_ks, 
                  solver_viol_beta, solver_prox_mu, solver_bin_lambda)
  vmap_in_axes = (jaxopt.OptStep(0, 0), (None, None), 0, 0, 0, 0, None, None, None)
  if solver_backend == "cpu":
    vmapped_lbfgsb_state = jax.pmap(subproblem_solver.run, in_axes=vmap_in_axes, backend='cpu')(*vmapped_args)
    # vmapped_lbfgsb_state = jax.vmap(subproblem_solver.run, in_axes=vmap_in_axes)(*vmapped_args)
  else:
    vmapped_lbfgsb_state = jax.vmap(subproblem_solver.run, in_axes=vmap_in_axes)(*vmapped_args)

  vmapped_xkp1_res = vmapped_lbfgsb_state.params
  x_kp1s = jax.vmap(lambda x: x.reshape(block_size, num_configs), in_axes=(0), out_axes=(0))(vmapped_xkp1_res)
  # jax.debug.print("Shape(x_kp1s): {xkps1shape}", xkps1shape=x_kp1s.shape)
  
  ####  Compute s^{k+1}, f^{k+1} using  closed-form ####
  # s^{k+1} = max(0, (mu * s^k - beta * t^k) / (mu + beta))
  t_k = (Amat @ jnp.sum(x_kp1s, axis=[0, 1])) - bvec + u_k
  # s_kp1 = (solver_prox_mu * s_k - solver_viol_beta * t_k) / (solver_prox_mu + solver_viol_beta)
  s_kp1 = -t_k
  # jax.debug.print("s_kp1: {s_kp1}", s_kp1=s_kp1)
  s_kp1 = jnp.clip(s_kp1, 0, None)

  # f_i^{k+1} = max(0, (mu * f_i^k - beta * z_i^k) / (mu + beta))
  z_ks = jnp.sum(x_kp1s, axis=2) - 1 + v_ks
  # f_kp1s = (solver_prox_mu * f_ks - solver_viol_beta * z_ks) / (solver_prox_mu + solver_viol_beta)
  f_kp1s = -z_ks
  f_kp1s = jnp.clip(f_kp1s, 0, None)

  #### Compute Dual updates ####
  # u^{k+1} = u^k + tau * (A * sum_i x_i^{k+1} + s^{k+1} - b)
  u_kp1 = u_k + solver_dual_tau * ((Amat @ jnp.sum(x_kp1s, axis=[0,1])) + s_kp1 - bvec)

  # compute v^{k+1}
  # v_i^{k+1} = v_i^k + tau * (x_i^{k+1}^T*1 + f_i^{k+1} - 1)
  v_kp1s = v_ks + solver_dual_tau * (jnp.sum(x_kp1s, axis=2) + f_kp1s - 1)

  # compute residual for fast ADMM update
  eta, d_k, alpha_k = other_state[0], other_state[1], other_state[2]
  sum_xk = jnp.sum(x_ks, axis=[0, 1])
  sum_xkp1 = jnp.sum(x_kp1s, axis=[0, 1])
  d_kp1 = solver_viol_beta * jnp.linalg.norm(Amat @ (sum_xkp1 - sum_xk)) + (1 / solver_viol_beta) * (jnp.linalg.norm(x_kp1s - x_ks))
  # initialize d_k > (d_kp1 / eta) so that the first update does not cause restart
  d_k = jnp.where(d_k < 1e-3, d_kp1 / eta + 1, d_k)

  # updates with restart
  # Could probably use momentum/ADAM or one of the other optimizers here
  # alpha_kp1 = jnp.where(d_kp1 < eta * d_k, (1 + jnp.sqrt(1 + 4 * alpha_k**2)) / 2, alpha_k / 2)
  alpha_kp1 = jnp.where(d_kp1 < eta * d_k, (1 + jnp.sqrt(1 + 4 * alpha_k**2)) / 2, 1)
  alpha_kp1 = jnp.clip(alpha_kp1, 1, 200.0)
  scale_factor = (alpha_k - 1) / alpha_kp1
  # scale_factor = 0
  # x_kp1s = jnp.where(d_kp1 < eta * d_k, x_kp1s + scale_factor * (x_kp1s - x_ks), x_kp1s)
  # u_kp1 = jnp.where(d_kp1 < eta * d_k, u_kp1 + scale_factor * (u_kp1 - u_k), u_k)
  # v_kp1s = jnp.where(d_kp1 < eta * d_k, v_kp1s + scale_factor * (v_kp1s - v_ks), v_ks)
  # x_kp1s = x_kp1s + scale_factor * (x_kp1s - x_ks)
  # s_kp1 = s_kp1 + scale_factor * (s_kp1 - s_k)
  # u_kp1 = u_kp1 + scale_factor * (u_kp1 - u_k)
  v_kp1s = v_kp1s + scale_factor * (v_kp1s - v_ks)
  # f_kp1s = f_kp1s + scale_factor * (f_kp1s - f_ks)
  # s_kp1 = jnp.clip(s_kp1, 0, None)
  # f_kp1s = jnp.clip(f_kp1s, 0, None)
  d_k = jnp.where(d_kp1 < eta * d_k, d_kp1, d_k / eta)
  # x_diff = jnp.linalg.norm(x_kp1s - x_ks)
  # jax.debug.print("x_diff: {x_diff}", x_diff=x_diff.round(3))
  # jax.debug.print("Iteration {}: obj_val = {}, gpu_cnstr_viol_norm = {}, sumto1_cnstr_viol_norm={}, x_progress={}", k, obj_val.round(3), gpu_cnstr_viol_norm.round(3), sumto1_cnstr_viol_norm.round(3), x_diff.round(3))
  other_state = other_state.at[1].set(d_kp1)
  other_state = other_state.at[2].set(alpha_kp1)

  # compute stats for iteration
  obj_val = jnp.sum(jnp.multiply(vmapped_cmat, x_kp1s)) - lambda_no_alloc * jnp.sum(x_kp1s)
  binarized = jnp.sum(x_kp1s * (1 - x_kp1s))
  gpu_cnstr_viol = bvec - s_kp1 - Amat @ jnp.sum(x_kp1s, axis=[0, 1])
  # jax.debug.print("GPU constraint violation: {gpu_cnstr_viol}", gpu_cnstr_viol=gpu_cnstr_viol)
  sumto1_cnstr_viol = (jnp.sum(x_kp1s, axis=2) + f_kp1s - 1).flatten()
  # jax.debug.print("Sumto1 constraint violation: {sumto1_cnstr_viol}", sumto1_cnstr_viol=sumto1_cnstr_viol)
  gpu_cnstr_viol_norm = jnp.linalg.norm(gpu_cnstr_viol).round(3)
  sumto1_cnstr_viol_norm = jnp.linalg.norm(sumto1_cnstr_viol).round(3)
  # jax.debug.print("Sum(x)={x_sum}", x_sum=jnp.sum(x_kp1s, axis=[0, 1, 2]).round(3))
  iter_stats = {
    "gpu_cnstr_viol_norms": gpu_cnstr_viol_norm,
    "sumto1_cnstr_viol_norms": sumto1_cnstr_viol_norm,
    "obj_vals": obj_val,
    "binarization": binarized
  }
  stats = jax.tree_map(lambda x, y: x.at[k].set(y), stats, iter_stats)

  new_state = (x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s, vmapped_lbfgsb_state, other_state, stats)
  return new_state
