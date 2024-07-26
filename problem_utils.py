import pickle
import numpy as np
from rich import print as rprint

# Read sia problems from a dump file; convert into separable form
def read_sia_problems(dump_file, start_idx=0, num_problems=None):
  with open(dump_file, 'rb') as f:
    sia_problems = pickle.load(f)
  if start_idx is not None:
    if start_idx > len(sia_problems):
      raise ValueError(f"Start index {start_idx} out of bounds :: {len(sia_problems)} problems available")
    sia_problems = sia_problems[start_idx:]
  if num_problems is not None:
    if num_problems > len(sia_problems):
      raise ValueError(f"Number of problems {num_problems} out of bounds:: {len(sia_problems)} problems available")
    sia_problems = sia_problems[:num_problems]
  # pick a program
  prog = sia_problems[0]
  # get the separable form matrix, vector
  A, b, c = prog['A'], prog['b'], prog['c']
  m, _ = A.shape
  njobs, nconfigs = prog['njobs'], prog['nconfigs']
  # first n-rows of A are for L1-norm constraints for each job
  ngpu_types = m - njobs
  rprint(f"njobs: {njobs}, nconfigs: {nconfigs}, ngpu_types: {ngpu_types}")
  G_mat = A[njobs:njobs + ngpu_types, :nconfigs]
  g_vec = b[njobs:njobs + ngpu_types]

  # convert all problems into separable form with changes
  costs = []
  for prog in sia_problems:
    c = prog['c']
    njobs = prog['njobs']
    jobnames = prog['jobnames']
    c_dict = {jobnames[i]: np.array(c[i*nconfigs:(i+1)*nconfigs]) for i in range(njobs)}
    costs.append(c_dict)
  
  problems = dict()
  problems['G_mat'] = G_mat
  problems['g_vec'] = g_vec
  problems['nconfigs'] = nconfigs
  problems['costs'] = costs

  return problems