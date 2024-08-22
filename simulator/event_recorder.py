class EventRecorder:
  def __init__(self, jobs, num_nodes, ngpus_per_node):
    self.jobs = jobs
    self.active_jobs = []
    self.num_nodes = None
    self.ngpus_per_node = None
    self.total_num_gpus = None
    self.failed_nodes = None
    self.current_time = 0

  # accumulate events for `seconds` time and return the events
  def step(self, seconds):
    pass

  def simulate_failure(self, node_id):
    pass