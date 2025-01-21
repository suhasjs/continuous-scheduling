from .job import AbstractJob, JobStatus
from .profiles.batch_inference import imagenet_resnet50, llama_8b_wikipedia, llama_8b_commoncrawl
from rich import print as rprint

class BatchInferenceJobClass:
  def __init__(self, model_name):
    self.model_name = model_name
    if model_name == "imagenet_resnet50":
      self.profiles = imagenet_resnet50
    elif model_name == "llama_8b_wikipedia":
      self.profiles = llama_8b_wikipedia
    elif model_name == "llama_8b_commoncrawl":
      self.profiles = llama_8b_commoncrawl
    else:
      raise ValueError(f"Model {model_name} not supported")
    self.scale_unit = {cluster: profile["min_gpus"] for cluster, profile in self.profiles["gpu_profiles"].items()}
    self.max_progress = self.profiles["num_iters"]
    # TODO :: make this a configurable parameter
    # setting to 16 to not let it take up all GPUs
    self.max_scale_units = 16
    self.speedup = self.profiles["sim_speedup"]
    self.restart_penalty = 90

  def evaluate_allocations(self, candidate_allocations):
    candidate_utilities = []
    for _, ngpus, gpu_type in candidate_allocations:
      if gpu_type not in self.profiles["gpu_profiles"]:
        candidate_utilities.append(0)
      else:
        profile = self.profiles["gpu_profiles"][gpu_type]
        per_unit_throughput = profile["throughput"]
        num_units = ngpus // self.scale_unit[gpu_type]
        if num_units > self.max_scale_units:
          throughput = 0
        else:
          throughput = per_unit_throughput * num_units
        candidate_utilities.append(throughput)
    # normalize utilities so min non-zero utility is num_gpus
    nonzeroutil_ngpus_tuples = [(x, y[1]) for x, y in zip(candidate_utilities, candidate_allocations) if x > 0]
    if len(nonzeroutil_ngpus_tuples) > 0:
      min_util, min_ngpus = min(nonzeroutil_ngpus_tuples, key=lambda x: x[0])
      norm_factor = (min_ngpus / min_util)
      candidate_utilities = [x * norm_factor for x in candidate_utilities]
    return candidate_utilities
  
  def get_throughput(self, allocation):
    _, ngpus, gpu_type = allocation
    if gpu_type not in self.profiles["gpu_profiles"]:
      return 0
    profile = self.profiles["gpu_profiles"][gpu_type]
    per_unit_throughput = profile["throughput"]
    num_units = ngpus // self.scale_unit[gpu_type]
    throughput = per_unit_throughput * num_units * self.speedup
    return throughput

def get_batch_inference_job_classes(total_cluster_gpus):
  batch_inference_job_classes = {"imagenet_resnet50": BatchInferenceJobClass("imagenet_resnet50"), 
                                 "llama_8b_wikipedia": BatchInferenceJobClass("llama_8b_wikipedia"),
                                 "llama_8b_commoncrawl": BatchInferenceJobClass("llama_8b_commoncrawl")
                                }
  return batch_inference_job_classes

# Represents a class of **batch** inference jobs
# Progress functions do not change over time (similar to SinglePhaseJob)
class BatchInferenceJob(AbstractJob):
  def __init__(self, name, submission_time, jobclass):
    super().__init__(name, submission_time)
    self.progress = 0
    self.jobclass = jobclass
    self.max_progress = jobclass.max_progress
    self.events.append((self.time, self.progress, self.status, None))
    self.realloc_countdown = 0
    # total time spent reallocating across all reallocs
    self.realloc_time = 0
    self.num_restarts = 0
    self.run_time = 0
  
  def get_save_state(self):
    state = super().get_save_state()
    state["jobclass"] = self.jobclass.model_name
    state["realloc_countdown"] = self.realloc_countdown
    state["realloc_time"] = self.realloc_time
    state["num_restarts"] = self.num_restarts
    state["run_time"] = self.run_time
    return state
  
  def load_saved_state(self, state):
    assert self.jobclass.model_name == state["jobclass"], f"JobClass mismatch: {self.jobclass.model_name} != {state['jobclass']}"
    super().load_saved_state(state)
    self.realloc_countdown = state["realloc_countdown"]
    self.realloc_time = state["realloc_time"]
    self.num_restarts = state["num_restarts"]
    self.run_time = state["run_time"]

  def evaluate_allocations(self, candidate_allocations):
    utilities = self.jobclass.evaluate_allocations(candidate_allocations)
    realloc_factor = self.run_time / (self.run_time + self.jobclass.restart_penalty)
    realloc_factor = 1
    if self.status == JobStatus.REALLOCATING:
      realloc_factor = 0
    # print(f"Job: {self.name} --> realloc_factor: {realloc_factor}")
    for i in range(len(utilities)):
      if candidate_allocations[i] != self.allocation:
        utilities[i] *= realloc_factor
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
      self.realloc_countdown = self.jobclass.restart_penalty
      self.status = JobStatus.REALLOCATING
      self.events.append((self.time, self.progress, self.status, self.allocation))
    else:
      self.allocation = None
      self.status = JobStatus.QUEUED
      self.events.append((self.time, self.progress, self.status, None))

  def step(self, seconds):
    if self.allocation is None:
      self.time += seconds
      self.queue_time += seconds
      self.progress += 0
      return
    
    seconds_left = seconds
    if self.status == JobStatus.REALLOCATING:
      # job is reallocating
      subtract_time = min(self.realloc_countdown, seconds)
      self.realloc_countdown -= subtract_time
      self.time += subtract_time
      seconds_left -= subtract_time
      if self.realloc_countdown == 0:
        # reallocation complete ==> transition to running
        self.status = JobStatus.RUNNING
        self.realloc_time += self.jobclass.restart_penalty
        self.num_restarts += 1
        self.events.append((self.time, self.progress, self.status, self.allocation))
    
    # no time left to make progress
    if seconds_left == 0:
      return
    ### self.allocation is not None ###
    # get rate of progress
    throughput = self.jobclass.get_throughput(self.allocation)
    if throughput == 0:
      rprint(f"[yellow] Job {self.name} has 0 throughput on {self.allocation}; simulating 0 progress[/yellow]")
      self.time += seconds_left
      self.queue_time += seconds_left
      return
    
    # update progress
    max_added_progress = self.max_progress - self.progress
    added_progress = throughput * seconds_left
    added_progress = min(added_progress, max_added_progress)
    # rprint(f"Throughput: {throughput}, added progress: {added_progress}")
    self.progress += added_progress
    self.run_time += round(added_progress / throughput)
    self.time += round(added_progress / throughput)

    # check if job is completed
    if self.progress >= self.max_progress:
      # mark job as completed
      self.status = JobStatus.COMPLETED
      self.progress = self.max_progress
      self.allocation = None
      self.completion_time = self.time + self.submission_time
      self.events.append((self.time, self.progress, self.status, None))
