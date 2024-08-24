from .job import AbstractJob, JobStatus
from rich import print as rprint
import numpy as np
import pickle

# Represents a class of adaptive jobs (one per model) run with Sia scheduler
# Progress functions *do* change over time
class SiaJobClass:
  CALIBRATION_FACTORS = {"cifar10": 1.8, "imagenet": 1.2, "yolov3": 1.2, 
                         "bert": 1.2, "deepspeech2": 1.2, "ncf": 1.5, "gpt_pmp": 1.0}
  # progress_profile = {configs: list of configs per GPU type,
  #                     logs: List[ progress: progress at log time,
  #                                 goodputs: {GPU type: goodputs for all configs of GPU type}]}
  def __init__(self, model_name, progress_profiles_path, total_cluster_gpus):
    with open(progress_profiles_path, 'rb') as f:
      progress_profiles = pickle.load(f)
    self.configs = progress_profiles['configs']
    self.progress_profiles = progress_profiles['logs']
    self.model_name = model_name
    self.calibration_factor = 1 / SiaJobClass.CALIBRATION_FACTORS[model_name]
    self.max_progress = self.progress_profiles[-1]['progress']

    # pre-process progress profiles
    # config -> idx map
    self.configs_to_idx = {}
    self.idx_to_configs = {}
    self.cluster_ordering = sorted(list(self.configs.keys()))
    idx = 0
    for cluster_name in self.cluster_ordering:
      num_nodes, num_gpus = self.configs[cluster_name]['num_nodes'], self.configs[cluster_name]['num_gpus']
      for nnodes, ngpus in zip(num_nodes, num_gpus):
        config = (nnodes, ngpus, cluster_name)
        self.configs_to_idx[config] = idx
        self.idx_to_configs[idx] = config
        idx += 1
    self.progresses = np.asarray([log['progress'] for log in self.progress_profiles])
    self.num_configs = len(self.configs_to_idx)
    self.progress_vals = np.zeros(shape=(len(self.progress_profiles), self.num_configs), dtype=np.float32)
    for i, log in enumerate(self.progress_profiles):
      cur_goodputs = log['goodputs']
      goodputs = []
      for cluster_name in self.cluster_ordering:
        goodputs.extend(cur_goodputs[cluster_name])
      self.progress_vals[i, :] = np.asarray(goodputs)
    # FILTER OUT CONFIGS NOT NEEDED
    chosen_config_idxs = []
    for i in range(self.num_configs):
      nnodes, ngpus, gpu_type = self.idx_to_configs[i]
      if gpu_type not in total_cluster_gpus:
        continue
      if ngpus > total_cluster_gpus[gpu_type]:
        continue
      chosen_config_idxs.append(i)
    self.progress_vals = self.progress_vals[:, chosen_config_idxs]
    new_configs_to_idx, new_idx_to_configs = {}, {}
    for i, idx in enumerate(chosen_config_idxs):
      config = self.idx_to_configs[idx]
      new_configs_to_idx[config] = i
      new_idx_to_configs[i] = config
    self.configs_to_idx = new_configs_to_idx
    self.idx_to_configs = new_idx_to_configs
    rprint(f"Class: {self.model_name}, filtered configs: {self.configs_to_idx.keys()}")

  # candidate_allocations: List of candidate allocations to evaluate
  # returns: List of normalized utilities for each candidate allocation
  def evaluate_allocations(self, progress, candidate_allocations):
    row_idx = np.searchsorted(self.progresses, progress, side='left')
    if row_idx == len(self.progresses):
      raise ValueError(f"Progress {progress} is greater than max progress {self.max_progress}")
    progress_vals = self.progress_vals[row_idx, :]
    candidate_utilities = [progress_vals[self.configs_to_idx[config]] if config in self.configs_to_idx else 0 for config in candidate_allocations]
    
    # normalize utilities so min non-zero utility is num_gpus for that config
    nonzeroutil_ngpus_tuples = [(x, y[1]) for x, y in zip(candidate_utilities, candidate_allocations) if x > 0]
    if len(nonzeroutil_ngpus_tuples) > 0:
      min_util, min_ngpus = min(nonzeroutil_ngpus_tuples, key=lambda x: x[0])
      norm_factor = (min_ngpus / min_util)
      candidate_utilities = [x * norm_factor for x in candidate_utilities]
    return candidate_utilities
  
  # returns: Tuple of (rate, valid_for) where rate is the progress rate for the given config
  #          and valid_for is the time until the next progress log
  def get_progress_rate(self, cur_progress, config):
    assert config is not None, "Config cannot be None"
    row_idx = np.searchsorted(self.progresses, cur_progress, side='left')
    if row_idx == len(self.progresses):
      return 0
    config_idx = self.configs_to_idx[config]
    rate = self.calibration_factor * self.progress_vals[row_idx, config_idx]
    valid_for = (self.progresses[row_idx + 1] - cur_progress) / rate
    return rate, valid_for

# Represents a class of jobs with only one phase
# Progress functions do not change over time
class SiaJob(AbstractJob):
  def __init__(self, name, submission_time, job_class_obj):
    super().__init__(name, submission_time)
    self.progress = 0
    self.job_class = job_class_obj
    self.max_progress = job_class_obj.max_progress
    self.events.append((self.time, self.progress, self.status, None))
  
  def evaluate_allocations(self, candidate_allocations):
    return self.job_class.evaluate_allocations(self.progress, candidate_allocations)
  
  def reallocate(self, new_allocation):
    if self.allocation == new_allocation:
      return
    # new_allocation = (1, 8, "dgx-ext")
    rprint(f"Reallocating job {self.name}:{self.allocation} --> {new_allocation}")
    if self.allocation is not None and new_allocation is not None:
      self.events.append((self.time, self.progress, JobStatus.REALLOCATING, (self.allocation, new_allocation)))
      self.allocation = new_allocation
      self.status = JobStatus.RUNNING
    elif new_allocation is not None:
      self.allocation = new_allocation
      self.status = JobStatus.RUNNING
      self.events.append((self.time, self.progress, self.status, self.allocation))
    else:
      self.allocation = None
      self.status = JobStatus.QUEUED
      self.events.append((self.time, self.progress, self.status, None))
  
  def __repr__(self):
    return f"SiaJob(name={self.name}, status={self.status}, progress={self.progress}/{self.max_progress}, alloc={self.allocation}, runtime={self.time}"

  def step(self, seconds):
    if self.allocation is None:
      self.time += seconds
      self.progress += 0
      return
    
    ### self.allocation is not None ###
    seconds_left = seconds
    while seconds_left > 0:
      # get rate of progress
      progress_rate, valid_for = self.job_class.get_progress_rate(self.progress, self.allocation)
      rprint(f"\t{self.name}, rate: {progress_rate:.2f}, valid_for: {valid_for:.2f}, progress={self.progress:.2f}/{self.max_progress:.2f}")
      run_for = min(seconds_left, valid_for)

      # update progress
      added_progress = progress_rate * run_for
      self.progress += added_progress
      self.time += run_for
      seconds_left -= run_for

      # check if job is completed
      if np.abs(self.progress - self.max_progress) < 1e-2:
        rprint(f"Job {self.name} completed: runtime={self.time}")
        self.status = JobStatus.COMPLETED
        self.progress = self.max_progress
        self.completion_time = self.time + self.submission_time
        # update allocation
        self.allocation = None
        self.events.append((self.time, self.progress, self.status, None))
        break