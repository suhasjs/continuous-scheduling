from .policy import AbstractPolicy
import cvxpy as cp
import numpy as np
from rich import print as rprint
import time
from copy import deepcopy
from .sia_ilp import SiaILP
from ray.util.multiprocessing import Pool

MAX_NUM_GPUS_IN_CONFIG = 2048
NUM_CORES = 8

# code copied from pop-ncflow-lptop/lib/runtime_utils.py
from heapq import heappush, heappop

VERBOSE = False


def heapsched_rt(lrts, k):
    h = []
    for rt in lrts[:k]:
        heappush(h, rt)

    curr_rt = 0
    for rt in lrts[k:]:
        curr_rt = heappop(h)
        heappush(h, rt + curr_rt)

    while len(h) > 0:
        curr_rt = heappop(h)

    return curr_rt


def parallelized_rt(lrts, k):
    if len(lrts) == 0:
        return 0.0
    inorder_rt = heapsched_rt(lrts, k)
    cp_bound = max(lrts)
    area_bound = sum(lrts) / k
    lrts.sort(reverse=True)
    two_approx = heapsched_rt(lrts, k)

    if VERBOSE:
        print("-- in incoming order, schedule= ", inorder_rt)
        print("-- bounds cp= ", cp_bound, "; area= ", area_bound)
        print("-- sorted rts: ", lrts)
        print("-- in sorted order, schedule ", two_approx)

    return two_approx


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
      subproblem = SiaILP(self.subproblem_num_nodes[i], self.subproblem_ngpus_per_node[i], 
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

    # startup a multiprocessing pool
    num_processes = min(NUM_CORES, num_subproblems)
    self.pool = Pool(num_processes)
  
  def get_save_state(self):
    state = {
      "current_time": self.current_time,
      "active_jobs": [job.name for job in self.active_jobs.values()],
      "allocations": self.allocations,
      "job_utilities": self.job_utilities,
      "solver_stats": self.solver_stats,
      "cluster_ordering": self.cluster_ordering,
      "max_ngpus": self.max_ngpus,
      "subproblems" : [subproblem.get_save_state() for subproblem in self.subproblems]
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
      subproblem.load_saved_state(subproblem_state, jobs)
  
  def get_configurations(self):
    configs_per_job = {}
    for jobname, _ in self.active_jobs.items():
      subproblem_id = self.jobs_to_subproblems_map[jobname]
      subproblem = self.subproblems[subproblem_id]
      configs = subproblem.get_configurations()
      configs_per_job[jobname] = configs
    return configs_per_job
  
  def update_failed_nodes(self, failed_nodes):
    pass

  def update_job_utilities(self, new_job_utilities):
    for jobname, utilities in new_job_utilities.items():
      subproblem_id = self.jobs_to_subproblems_map[jobname]
      self.subproblems[subproblem_id].update_job_utilities({jobname: utilities})

  def add_new_jobs(self, new_jobs):
    print(f"[green]Adding {len(new_jobs)} new jobs...[/green]")
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
      print(f"\t{jobname} -> subproblem {subproblem_id}")
      self.subproblems[subproblem_id].add_new_jobs({jobname: job})

  def remove_completed_jobs(self, completed_jobs):
    for jobname in completed_jobs:
      if jobname not in self.active_jobs:
        continue
      else:
        self.active_jobs.pop(jobname)
        subproblem_id = self.jobs_to_subproblems_map[jobname]
        self.subproblem_to_jobs_map[subproblem_id].remove(jobname)
        self.subproblems[subproblem_id].remove_completed_jobs([jobname])
        self.jobs_to_subproblems_map.pop(jobname)

  def step(self, seconds):
    self.current_time += seconds
    for i, subproblem in enumerate(self.subproblems):
      subproblem.step(seconds)

  def optimize_allocations(self):
    # start setup time 
    solve_start = time.time()
    self.allocations = {}
    def optimize_subproblem(subproblem):
      subproblem.optimize_allocations()
      return subproblem
    # parallelize subproblem optimization
    self.subproblems = self.pool.map(optimize_subproblem, self.subproblems)
    for subproblem in self.subproblems:
      self.allocations.update(subproblem.allocations)
    
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
    stat = {"time": self.current_time, 
            "num_jobs": sum([subproblem.solver_stats[-1]['num_jobs'] for subproblem in self.subproblems]),
            "num_vars": sum([subproblem.solver_stats[-1]['num_vars'] for subproblem in self.subproblems]),
            "setup_time_ms": 0, 
            "solve_time_ms": total_solve_time * 1000, 
            "solver_status": "Optimal", 
            "objective_val": sum([subproblem.solver_stats[-1]['objective_val'] for subproblem in self.subproblems]),
            "num_subproblems": len(self.subproblems),
            "subproblem_stats": [subproblem.solver_stats[-1] for subproblem in self.subproblems],
            }
    # update solver stats
    self.solver_stats.append(stat)
    if self.subproblem_solver_options[0]['verbose']:
      rprint(f"[yellow]POP stats: {stat}[/yellow]")
  def get_allocations(self):
    return self.allocations