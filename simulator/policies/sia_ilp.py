from policy import AbstractPolicy

class SiaILP(AbstractPolicy):
  def __init__(self, num_nodes, ngpus_per_node):
    self.num_nodes = num_nodes
    self.ngpus_per_node = ngpus_per_node
    self.cluster_ordering = sorted(list(num_nodes.keys()))
    self.cluster_gpus = {cluster: num_nodes[cluster] * ngpus_per_node for cluster in self.cluster_ordering}
    self.total_num_gpus = sum(self.cluster_gpus.values())
    self.allocations = {}
    self.active_jobs = []

    # policy parameters
    self.lambda_no_alloc = 1.1
    self.p_value = 0.5
    pass

  def update_job_utilities(self, new_job_utilities):
    pass

  def add_new_jobs(self, new_jobs):
    pass

  def remove_completed_jobs(self, completed_jobs):
    pass

  def step(self, seconds):
    pass
