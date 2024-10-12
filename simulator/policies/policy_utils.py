import jaxopt
import jax
import jax.numpy as jnp

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
# Captures: num_configs, lambda_no_alloc, Amat, bvec, require_binary_solutions
# Augmented Lagrangian for the Sia ILP/LP-relaxation policy
def sia_auglag_fun(xikp1, c_is, rki, xki, yki, aug_viol_beta, aug_prox_mu, other_params):
  Amat = other_params["Amat"]
  num_configs = other_params["num_configs"]
  lambda_no_alloc = other_params["lambda_no_alloc"]
  require_binary_solutions = other_params["require_binary_solutions"]
  
  # reshape xikp1 to (-1, nconfigs)
  reshaped_xikp1 = jax.tree_map(lambda x: x.reshape(-1, num_configs), xikp1)
  # add <c_i, x_i> term --> cost (negative of utility)
  ret = jaxopt.tree_util.tree_vdot(c_is, reshaped_xikp1)
  # add -lambda_no_alloc * sum_i (x_i) term --> incentivize allocation (penalize no allocation)
  scalar_add_val = jax.tree_map(lambda x: jnp.sum(x), reshaped_xikp1)
  ret = jaxopt.tree_util.tree_add_scalar_mul(ret, -1 * lambda_no_alloc, scalar_add_val)
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
  if require_binary_solutions:
    rhs = jaxopt.tree_util.tree_add_scalar_mul(1, -1, reshaped_xikp1)
    binarization = jaxopt.tree_util.tree_vdot(reshaped_xikp1, rhs)
    ret = jaxopt.tree_util.tree_add_scalar_mul(ret, 1e-1, binarization)
  return ret

# Initialize an LBFGS-B solver for the primal subproblem in Prox Jacobi ADMM
def initialize_subproblem_solver(lbfgs_solver_params, init_params, auglag_other_params):
  # extract params
  job_primals_k, job_primal_bounds = init_params["job_primals_k"], init_params["job_primal_bounds"]
  job_duals_k, gpu_duals_k = init_params["job_duals_k"], init_params["gpu_duals_k"]
  job_slacks_k, gpu_slacks_k = init_params["job_slacks_k"], init_params["gpu_slacks_k"]
  gpu_vec_k, sum_vec_k = init_params["gpu_vec_k"], init_params["sum_vec_k"]
  vmapped_cmat = init_params["vmapped_cmat"]
  lbfgs_max_iters, lbfgs_history_size = lbfgs_solver_params["max_iters"], lbfgs_solver_params["history_size"]
  solver_viol_beta, solver_prox_mu = lbfgs_solver_params["viol_beta"], lbfgs_solver_params["prox_mu"]

  specialized_auglag_fun = jax.tree_util.Partial(sia_auglag_fun, other_params=auglag_other_params)

  #### Initialize LBFGS-B solver
  subproblem_solver = jaxopt.LBFGSB(fun=specialized_auglag_fun, jit=True, unroll=True, implicit_diff=False,
                                    maxiter=lbfgs_max_iters, tol=1e-4, stepsize=0.9, 
                                    history_size=lbfgs_history_size, use_gamma=True)
  # Initialize subproblem state for one block
  subproblem_solver_state = subproblem_solver.init_state(init_params=job_primals_k[0, :].reshape(-1), 
                                                          bounds=job_primal_bounds, c_is=vmapped_cmat[0, :, :], 
                                                          rki=gpu_vec_k[0, :], xki=job_primals_k[0, :], 
                                                          yki=sum_vec_k[0, :], aug_viol_beta=solver_viol_beta, 
                                                          aug_prox_mu=solver_prox_mu, other_params=auglag_other_params)
  subproblem_solver_optstep = jaxopt.OptStep(job_primals_k[0].reshape(-1), subproblem_solver_state)

  def vmap_run(init_params, c_is, rki, xki, yki, aug_beta, aug_mu):
    return subproblem_solver.run(init_params=init_params, bounds=job_primal_bounds,
                                  c_is=c_is, rki=rki, xki=xki, yki=yki, 
                                  aug_viol_beta=aug_beta, aug_prox_mu=aug_mu)
  def hardware_mapper_fun(fun, in_axes, backend):
    if backend=="cpu":
      return jax.pmap(fun, in_axes=in_axes, backend='cpu')
    else:
      return jax.vmap(fun, in_axes=in_axes)
  
  jax_vmap_fun = hardware_mapper_fun(vmap_run, in_axes=(jaxopt.OptStep(0, 0), 0, 0, 0, 0, None, None),
                                      backend=solver_backend)
  # create inputs for vmapped solve
  init_xks = jax.vmap(lambda x: x.reshape(-1), in_axes=(0), out_axes=(0))(job_primals_k)
  init_vmapped_optstep = jaxopt.OptStep(init_xks, subproblem_solver_optstep.state)
  vmap_run_broadcast_state = jax.vmap(vmap_run, in_axes=(jaxopt.OptStep(0, None), 0, 0, 0, 0, None, None))
  lbfgs_state =  vmap_run_broadcast_state(init_vmapped_optstep, vmapped_cmat, gpu_vec_k, job_primals_k, 
                                          sum_vec_k, solver_viol_beta, solver_prox_mu)
  return jax_vmap_fun, lbfgs_state