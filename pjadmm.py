import jax
import jax.numpy as jnp
import jaxopt
from rich import print as rprint

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
def eval_augmented_lagrangian(xikp1, c_is, rki, xki, yki, aug_viol_beta, aug_prox_mu, # dynamic args
                              nconfigs, Amat, binary_constant = 1e-3, require_binary_solutions=False): # static args
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

'''
One iteration of Proximal Jacobi ADMM
state: (x_ks, u_k, s_k, f_ks, v_ks, LBFGS-state)
      x_ks: primals per partition block
      u_k: duals for GPU constraints
      s_k: slack variables for GPU constraints
      f_ks: slack variables for sum-to-1 constraints
      v_ks: duals for sum-to-1 constraints
      LBFGS-state: state for LBFGS optimizer
params: (Amat, bvec, cmats, nconfigs, jax_vmap_fun, aug_viol_beta, aug_prox_mu)
      Amat: matrix for GPU constraints
      bvec: vector for GPU constraints
      cmats: cost matrices for each partition block
      nconfigs: number of configurations
      jax_vmap_fun: L-BFGS optimizer vmap function
      aug_viol_beta: cnstr violation parameter for augmented lagrangian
      aug_prox_mu: proximal parameter for augmented lagrangian
'''
def iter_prox_jacobi_admm(k, state, params):
  # unpack args
  x_ks, u_k, s_k, f_ks, v_ks, vmapped_lbfgs_optstep = state
  Amat, bvec, cmats, nconfigs, jax_vmap_fun, aug_viol_beta, aug_prox_mu = params
  
  # hard-coded params
  dual_tau = 1  # dual update step size
  
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
  vmapped_lbfgs_optstep = jax_vmap_fun(vmapped_lbfgs_optstep, cmats, r_ks, x_ks, y_ks, 
                                       aug_viol_beta, aug_prox_mu)
  vmapped_xkp1_res = vmapped_lbfgs_optstep.params
  x_kp1s = jax.vmap(lambda x: x.reshape(-1, nconfigs), in_axes=(0), out_axes=(0))(vmapped_xkp1_res)

  # compute s^{k+1}
  # s_kp1 = (aug_prox_mu * s_k - aug_viol_beta * t_k) / (aug_prox_mu + aug_viol_beta)
  s_kp1 = -vmapped_tk
  s_kp1 = jnp.clip(s_kp1, 0, bvec)

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

  # swap (x_k, u_k, s_k, f_k, v_k) with (x_kp1, u_kp1, s_kp1, f_kp1, v_kp1)
  # under-relaxation (works better than nestrov)
  alpha_k = 1
  gamma_k = 1.2
  u_kp1 = u_k - gamma_k*alpha_k*(u_k - u_kp1)
  s_kp1 = s_k - gamma_k*alpha_k*(s_k - s_kp1)
  # x_kp1s = (x_ks - gamma_k*alpha_k*(x_ks - x_kp1s)).round(2)
  x_kp1s = x_ks - gamma_k*alpha_k*(x_ks - x_kp1s)
  f_kp1s = f_ks - gamma_k*alpha_k*(f_ks - f_kp1s)
  v_kp1s = v_ks - gamma_k*alpha_k*(v_ks - v_kp1s)

  new_state = (x_kp1s, u_kp1, s_kp1, f_kp1s, v_kp1s, vmapped_lbfgs_optstep)
  return new_state