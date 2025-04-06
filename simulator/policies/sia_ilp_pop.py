from .policy import AbstractPolicy
import cvxpy as cp
import numpy as np
from rich import print as rprint
import time
from copy import deepcopy
from .sia_ilp import SiaILP
import ray

MAX_NUM_GPUS_IN_CONFIG = 2048
NUM_CORES = 8
 
@ray.remote(num_cpus=1)
class SiaILPRayActor(SiaILP):
  def get_solver_stat(self):
    return self.solver_stats[-1] if len(self.solver_stats) > 0 else None

class SiaILPPOP(AbstractPolicy):
  def __init__(self, num_nodes, ngpus_per_node, policy_options, solver_options, num_subproblems=4):
    self.subproblem_ngpus_per_node = [deepcopy(ngpus_per_node) for _ in range(num_subproblems)]
    self.subproblem_policy_options = [deepcopy(policy_options) for _ in range(num_subproblems)]
    self.subproblem_solver_options = [deepcopy(solver_options) for _ in range(num_subproblems)]
    self.subproblem_num_nodes = []
    num_remaining_nodes = deepcopy(num_nodes)
    for i in range(num_subproblems):
      subproblem_nnodes = deepcopy(num_nodes)
      for cluster in subproblem_nnodes.keys():
        subproblem_nnodes[cluster] = min(num_remaining_nodes[cluster], num_nodes[cluster] // num_subproblems)
        num_remaining_nodes[cluster] -= subproblem_nnodes[cluster]
      self.subproblem_num_nodes.append(subproblem_nnodes)

    # create subproblems
    self.subproblems = []
    for i in range(num_subproblems):
      subproblem = SiaILPRayActor.remote(self.subproblem_num_nodes[i], self.subproblem_ngpus_per_node[i], 
                          self.subproblem_policy_options[i], self.subproblem_solver_options[i])
      self.subproblems.append(subproblem)

    # cluster configuration
    self.num_nodes = num_nodes
    self.ngpus_per_node = ngpus_per_node
    self.cluster_ordering = sorted(list(num_nodes.keys()))
    self.num_gputypes = len(self.cluster_ordering)
    self.cluster_gpus = {cluster: num_nodes[cluster] * ngpus_per_node[cluster] for cluster in self.cluster_ordering}
    self.max_ngpus = np.asarray([self.cluster_gpus[cluster] for cluster in self.cluster_ordering])
    self.total_num_gpus = sum(self.cluster_gpus.values())

    # cluster state
    self.active_jobs = {}
    self.jobs_to_subproblems_map = {}
    self.subproblem_to_jobs_map = {i : [] for i in range(num_subproblems)}
    self.allocations = {}
    self.job_utilities = {}
    self.current_time = 0

    # stats
    self.solver_stats = []
  
  def get_save_state(self):
    state = {
      "current_time": self.current_time,
      "active_jobs": [job.name for job in self.active_jobs.values()],
      "allocations": self.allocations,
      "job_utilities": self.job_utilities,
      "solver_stats": self.solver_stats,
      "cluster_ordering": self.cluster_ordering,
      "max_ngpus": self.max_ngpus,
      "subproblems" : ray.get([subproblem.get_save_state.remote() for subproblem in self.subproblems])
    }
    return state
  
  def load_saved_state(self, state, jobs):
    self.current_time = state["current_time"]
    self.active_jobs = {job.name:job for job in jobs if job.name in state["active_jobs"]}
    self.allocations = state["allocations"]
    self.job_utilities = state["job_utilities"]
    self.solver_stats = state["solver_stats"]
    for old_cluster_name, new_cluster_name in zip(state["cluster_ordering"], self.cluster_ordering):
      assert old_cluster_name == new_cluster_name, f"Cluster ordering mismatch: {old_cluster_name} != {new_cluster_name}"
    for old_cluster_ngpus, new_cluster_ngpus in zip(state["max_ngpus"], self.max_ngpus):
      assert old_cluster_ngpus == new_cluster_ngpus, f"Cluster GPU count mismatch: {old_cluster_ngpus} != {new_cluster_ngpus}"
    for i, subproblem in enumerate(self.subproblems):
      subproblem_state = state["subproblems"][i]
      ray.get(subproblem.load_saved_state.remote(subproblem_state, jobs))
  
  def get_configurations(self):
    configs_per_job = {}
    subproblem_configs = ray.get([subproblem.get_configurations.remote() for subproblem in self.subproblems])
    for jobname, _ in self.active_jobs.items():
      subproblem_id = self.jobs_to_subproblems_map[jobname]
      configs_per_job[jobname] = subproblem_configs[subproblem_id]
    return configs_per_job
  
  def update_failed_nodes(self, failed_nodes):
    pass

  def update_job_utilities(self, new_job_utilities):
    for jobname, utilities in new_job_utilities.items():
      subproblem_id = self.jobs_to_subproblems_map[jobname]
      self.subproblems[subproblem_id].update_job_utilities.remote({jobname: utilities})

  def add_new_jobs(self, new_jobs):
    rprint(f"[green]Adding {len(new_jobs)} new jobs...[/green]")
    added_jobs_list = [list() for _ in range(len(self.subproblems))]
    # add job to subproblems with least number of jobs
    for jobname, job in new_jobs.items():
      # find subproblem with least number of jobs
      id_counts = [(i, len(self.subproblem_to_jobs_map[i])) for i in range(len(self.subproblems))]
      id_counts = sorted(id_counts, key=lambda x: x[1])
      subproblem_id = id_counts[0][0]
      assert jobname not in self.subproblem_to_jobs_map[subproblem_id], f"Job {jobname} already exists in subproblem {subproblem_id}"
      self.subproblem_to_jobs_map[subproblem_id].append(jobname)
      self.jobs_to_subproblems_map[jobname] = subproblem_id
      # add job to subproblem
      self.active_jobs[jobname] = job
      added_jobs_list[subproblem_id].append(jobname)
      self.subproblems[subproblem_id].add_new_jobs.remote({jobname: job})
    for i, added_jobs in enumerate(added_jobs_list):
      rprint(f"\tSubproblem #{i} -> {added_jobs}")

  def remove_completed_jobs(self, completed_jobs):
    for jobname in completed_jobs:
      if jobname not in self.active_jobs:
        continue
      else:
        self.active_jobs.pop(jobname)
        subproblem_id = self.jobs_to_subproblems_map[jobname]
        self.subproblem_to_jobs_map[subproblem_id].remove(jobname)
        self.subproblems[subproblem_id].remove_completed_jobs.remote([jobname])
        self.jobs_to_subproblems_map.pop(jobname)

  def step(self, seconds):
    self.current_time += seconds
    for i, subproblem in enumerate(self.subproblems):
      subproblem.step.remote(seconds)

  def optimize_allocations(self):
    # start setup time 
    solve_start = time.time()
    self.allocations = {}
    _ = ray.get([subproblem.optimize_allocations.remote() for subproblem in self.subproblems])
    for subproblem in self.subproblems:
      self.allocations.update(ray.get(subproblem.get_allocations.remote()))
    
    # # solve each subproblem
    # # Single-threaded version
    # for i in range(len(self.subproblems)):
    #   subproblem = self.subproblems[i]
    #   # solve subproblem
    #   subproblem.optimize_allocations()
    #   # update allocations and job utilities
    #   self.allocations.update(subproblem.allocations)
    solve_end = time.time()
    total_solve_time = solve_end - solve_start
    subproblem_solver_stats = ray.get([subproblem.get_solver_stat.remote() for subproblem in self.subproblems])
    stat = {"time": self.current_time, 
            "num_jobs": sum([subproblem_solver_stat['num_jobs'] for subproblem_solver_stat in subproblem_solver_stats]),
            "num_vars": sum([subproblem_solver_stat['num_vars'] for subproblem_solver_stat in subproblem_solver_stats]),
            "setup_time_ms": 0, 
            "solve_time_ms": total_solve_time * 1000, 
            "solver_status": "Optimal", 
            "objective_val": sum([subproblem_solver_stat['objective_val'] for subproblem_solver_stat in subproblem_solver_stats]),
            "num_subproblems": len(self.subproblems),
            "subproblem_stats": subproblem_solver_stats,
            }
    # update solver stats
    self.solver_stats.append(stat)
    rprint(f"[yellow]POP stats: {stat}[/yellow]")

  def get_allocations(self):
    return self.allocations