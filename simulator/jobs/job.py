from enum import Enum

# Define the JobStatus Enum
class JobStatus(Enum):
  INVALID = -1
  QUEUED = 0
  RUNNING = 1
  COMPLETED = 2

class AbstractJob:
  def __init__(self, name):
    self.name = name
    self.status = JobStatus.INVALID
    self.allocation = None
    self.time = 0
    self.progress = None
    self.max_progress = None
    self.progress_fns = None
    self.events = []

  # Evaluate the utility of the job on proposed candidate_allocations
  # returns a dict of {candidate_allocation -> utility}
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