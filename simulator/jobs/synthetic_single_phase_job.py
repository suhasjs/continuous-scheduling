from .job import AbstractJob, JobStatus
from rich import print as rprint
import pickle

class SyntheticSinglePhaseJobClass:
  def __init__(self, model_name):
    self.model_name = model_name
    profile_filename = "./jobs/profiles/synthetic_{}.pkl"
    if model_name == "synthetic_linear_short":
      profile_filename = profile_filename.format("linear_short")
    elif model_name == "synthetic_linear_long":
      profile_filename = profile_filename.format("linear_long")
    elif model_name == "synthetic_mixed_short":
      profile_filename = profile_filename.format("mixed_short")
    elif model_name == "synthetic_mixed_long":
      profile_filename = profile_filename.format("mixed_long")
    elif model_name == "synthetic_sublinear_short":
      profile_filename = profile_filename.format("sublinear_short")
    elif model_name == "synthetic_sublinear_long":
      profile_filename = profile_filename.format("sublinear_long")
    else:
      raise ValueError(f"Model {model_name} not supported")
    with open(profile_filename, 'rb') as f:
      self.profiles = pickle.load(f)

    self.max_num_gpus = 64
    
    self.max_progress = self.profiles["max_progress"]
    self.goodputs = self.profiles["goodputs"]
    self.num_gpus_to_idx = {v : k for k, v in enumerate(self.goodputs["num_gpus"])}
    self.gpu_types_rename = {
      "dgx-ext": "DGX-A100-40GB",
      "azure": "DGX-V100-32GB",
      "a100-pcie": "A100-40GB-PCIe",
      "a10-pcie": "A10-24GB-PCIe",
      "rtx": "RTX-2080Ti-11GB",
      "quad": "RTX-2080Ti-11GB",
      "aws": "T4-16GB"
    }

  def evaluate_allocations(self, candidate_allocations):
    candidate_utilities = []
    for _, ngpus, gpu_type in candidate_allocations:
      if gpu_type not in self.gpu_types_rename.keys() or ngpus not in self.num_gpus_to_idx.keys():
        candidate_utilities.append(0)
      elif ngpus > self.max_num_gpus:
          candidate_utilities.append(0)
      else:
        renamed_gpu_type = self.gpu_types_rename[gpu_type]
        ngpus_idx = self.num_gpus_to_idx[ngpus]
        goodput = self.goodputs[renamed_gpu_type][ngpus_idx]
        candidate_utilities.append(goodput)
    # normalize utilities so min non-zero utility is num_gpus
    nonzeroutil_ngpus_tuples = [(x, y[1]) for x, y in zip(candidate_utilities, candidate_allocations) if x > 0]
    if len(nonzeroutil_ngpus_tuples) > 0:
      min_util, min_ngpus = min(nonzeroutil_ngpus_tuples, key=lambda x: x[0])
      norm_factor = (min_ngpus / min_util)
      candidate_utilities = [x * norm_factor for x in candidate_utilities]
    return candidate_utilities
  
  def get_throughput(self, allocation):
    if allocation is None:
      return 0
    _, ngpus, gpu_type = allocation
    if gpu_type not in self.gpu_types_rename.keys() or ngpus not in self.num_gpus_to_idx.keys():
      return 0
    renamed_gpu_type = self.gpu_types_rename[gpu_type]
    ngpus_idx = self.num_gpus_to_idx[ngpus]
    goodput = self.goodputs[renamed_gpu_type][ngpus_idx]
    return goodput

def get_synthetic_job_classes(total_cluster_gpus):
  synthetic_job_classes = {
    "synthetic_linear_short": SyntheticSinglePhaseJobClass("synthetic_linear_short"),     # ~10k GPU-secs
    "synthetic_linear_long": SyntheticSinglePhaseJobClass("synthetic_linear_long"),       # ~500k GPU-secs
    "synthetic_mixed_short": SyntheticSinglePhaseJobClass("synthetic_mixed_short"),       # ~5k GPU-secs
    "synthetic_mixed_long": SyntheticSinglePhaseJobClass("synthetic_mixed_long"),         # ~200k GPU-secs
    "synthetic_sublinear_short": SyntheticSinglePhaseJobClass("synthetic_sublinear_short"),# ~10k GPU-secs
    "synthetic_sublinear_long": SyntheticSinglePhaseJobClass("synthetic_sublinear_long")  # ~100k GPU-secs
  }
  return synthetic_job_classes

# Represents a class of synthetic jobs with a fixed scaling curve
# Progress functions do not change over time (similar to SinglePhaseJob)
class SyntheticSinglePhaseJob(AbstractJob):
  def __init__(self, name, submission_time, jobclass):
    super().__init__(name, submission_time)
    self.progress = 0
    self.jobclass = jobclass
    self.max_progress = jobclass.max_progress
    self.events.append((self.time, self.progress, self.status, None))
  
  def get_save_state(self):
    state = super().get_save_state()
    state["jobclass"] = self.jobclass.model_name
    return state
  
  def load_saved_state(self, state):
    assert self.jobclass.model_name == state["jobclass"], f"Job class mismatch: {self.jobclass.model_name} != {state['jobclass']}"
    super().load_saved_state(state)
  
  def evaluate_allocations(self, candidate_allocations):
    utilities = self.jobclass.evaluate_allocations(candidate_allocations)
    # rprint(f"Job: {self.name}, utilities: {list(zip(candidate_allocations, utilities))}")
    return utilities
  
  def reallocate(self, new_allocation):
    if self.allocation == new_allocation:
      return
    # rprint(f"Job: {self.name}, change of allocation: {self.allocation} -> {new_allocation}")
    if self.allocation is not None and new_allocation is not None:
      self.events.append((self.time, self.progress, JobStatus.REALLOCATING, (self.allocation, new_allocation)))
      self.allocation = new_allocation
      self.status = JobStatus.RUNNING
    elif new_allocation is not None:
      self.allocation = new_allocation
      self.status = JobStatus.RUNNING
      self.events.append((self.time, self.progress, self.status, self.allocation))
    else:
      self.allocation = new_allocation
      self.status = JobStatus.QUEUED
      self.events.append((self.time, self.progress, self.status, None))

  def step(self, seconds):
    if self.allocation is None:
      self.time += seconds
      self.progress += 0
      return
    
    ### self.allocation is not None ###
    # get rate of progress
    throughput = self.jobclass.get_throughput(self.allocation)
    if throughput == 0:
      rprint(f"[yellow] Job {self.name} has 0 throughput with allocation {self.allocation}; simulating 0 progress")
      self.time += seconds
      self.queue_time += seconds
      return
    # update progress
    max_added_progress = self.max_progress - self.progress
    added_progress = throughput * seconds
    added_progress = min(added_progress, max_added_progress)
    # rprint(f"Throughput: {throughput}, added progress: {added_progress}")
    self.progress += added_progress
    self.time += round(added_progress / throughput)

    # check if job is completed
    if self.progress >= self.max_progress:
      # mark job as completed
      self.status = JobStatus.COMPLETED
      self.progress = self.max_progress
      self.allocation = None
      self.completion_time = self.time + self.submission_time
      self.events.append((self.time, self.progress, self.status, None))