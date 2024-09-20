from enum import Enum

# Define the JobStatus Enum
class JobStatus(Enum):
  INVALID = -1
  QUEUED = 0
  RUNNING = 1
  REALLOCATING = 2
  COMPLETED = 3

class AbstractJob:
  def __init__(self, name, submission_time):
    self.name = name
    self.status = JobStatus.INVALID
    self.allocation = None
    self.time = 0
    self.queue_time = 0
    self.submission_time = submission_time
    self.completion_time = None
    self.reallocation_penalty = 0
    self.progress = None
    self.max_progress = None
    self.progress_fns = None
    self.events = []
  
  def get_save_state(self):
    state = {
      "name": self.name,
      "status": self.status,
      "allocation": self.allocation,
      "time": self.time,
      "submission_time": self.submission_time,
      "completion_time": self.completion_time,
      "queue_time": self.queue_time,
      "reallocation_penalty": self.reallocation_penalty,
      "progress": self.progress,
      "max_progress": self.max_progress,
      "events": self.events
    }
    return state
  
  def load_saved_state(self, state):
    self.name = state["name"]
    self.status = state["status"]
    self.allocation = state["allocation"]
    self.time = state["time"]
    self.submission_time = state["submission_time"]
    self.completion_time = state["completion_time"]
    self.queue_time = state["queue_time"]
    self.reallocation_penalty = state["reallocation_penalty"]
    self.progress = state["progress"]
    self.max_progress = state["max_progress"]
    self.events = state["events"]

  def __repr__(self):
    return f"Job(name={self.name}, status={self.status}, progress={self.progress}/{self.max_progress}, alloc={self.allocation}, runtime={self.time}"

  # Evaluate the utility of the job on proposed candidate_allocations
  # candidate_allocations: List[(num_nodes, num_gpus, gpu_type)]
  # returns List[utility of candidate_allocation #i]
  def evaluate_allocations(self, candidate_allocations):
    pass

  # Update allocation to new_allocation
  def reallocate(self, new_allocation):
    pass

  # Migrate job to new GPU type
  def migrate(self, new_allocation):
    return self.reallocate(new_allocation)

  # Scale up job to new_allocation on same GPU type
  def scale_up(self, new_allocation):
    return self.reallocate(new_allocation)

  # Scale down job to new_allocation on same GPU type
  def scale_down(self, new_allocation):
    return self.reallocate(new_allocation)

  # Update progress of job for given time
  def step(self, seconds):
    pass