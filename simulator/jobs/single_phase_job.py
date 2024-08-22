from job import AbstractJob, JobStatus

# Represents a class of jobs with only one phase
# Progress functions do not change over time
class SinglePhaseJob(AbstractJob):
  def __init__(self, name, max_progress, progress_fns):
    super().__init__(name)
    self.progress = 0
    self.progress_fns = progress_fns
    self.max_progress = max_progress
    self.status = JobStatus.QUEUED
    self.events.append((self.time, self.progress, self.status, None))
  
  def evaluate_allocations(self, candidate_allocations):
    candidate_utilities = []
    for alloc in candidate_allocations:
      num_nodes, num_gpus, gpu_type = alloc
      if gpu_type not in self.progress_fns:
        candidate_utilities.append(0)
      else:
        util_fn = self.progess_fns[gpu_type]
        util = util_fn(num_nodes, num_gpus)
        candidate_utilities.append(util)
    # normalize utilities so min non-zero utility is num_gpus
    nonzeroutil_ngpus_tuples = [(x, y) for x, y in zip(candidate_utilities, candidate_allocations) if x > 0]
    if len(nonzeroutil_ngpus_tuples) > 0:
      min_util, min_ngpus = min(nonzeroutil_ngpus_tuples, key=lambda x: x[0])
      norm_factor = (min_ngpus / min_util)
      candidate_utilities = [x * norm_factor for x in candidate_utilities]
    return candidate_utilities
  
  def reallocate(self, new_allocation):
    if self.allocation is not None and new_allocation is not None:
      self.events.append((self.time, self.progress, JobStatus.REALLOCATING, (self.allocation, new_allocation)))
      self.allocation = new_allocation
      self.status = JobStatus.RUNNING
    elif new_allocation is not None:
      self.allocation = new_allocation
      self.status = JobStatus.RUNNING
      self.events.append((self.time, self.progress, self.status, self.allocation))
    else:
      self.status = JobStatus.QUEUED
      self.events.append((self.time, self.progress, self.status, None))

  def step(self, seconds):
    if self.allocation is None:
      self.time += seconds
      self.progress += 0
      return
    ### self.allocation is not None ###
    # get rate of progress
    num_nodes, num_gpus, gpu_type = self.allocation
    progress_fn = self.progress_fns[gpu_type]
    progress_rate = progress_fn(num_nodes, num_gpus)
    # update progress
    max_added_progress = self.max_progress - self.progress
    added_progress = progress_rate * seconds
    added_progress = min(added_progress, max_added_progress)
    self.progress += added_progress
    self.time += round(added_progress / progress_rate)

    # check if job is completed
    if self.progress >= self.max_progress:
      self.status = JobStatus.COMPLETED
      self.progress = self.max_progress
    
    # update allocation
    self.allocation = None
    self.events.append((self.time, self.progress, self.status, None))