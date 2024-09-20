from jobs.job import JobStatus
from rich import print as rprint
import time
class EventRecorder:
  def __init__(self, jobs, num_nodes, ngpus_per_node):
    self.jobs = jobs
    # rprint(f"Jobs: {jobs}")
    self.active_jobs = {}
    self.num_nodes = num_nodes
    self.ngpus_per_node = ngpus_per_node
    self.total_num_gpus = {cluster: num_nodes[cluster] * ngpus_per_node[cluster] for cluster in num_nodes.keys()}
    self.failed_nodes = None
    self.current_time = 0
    self.job_completions = []

  def get_save_state(self):
    state = {}
    state["active_jobs"] = [job.name for job in self.active_jobs.values()]
    state["current_time"] = self.current_time
    state["job_completions"] = self.job_completions
    state["jobs"] = [job.get_save_state() for job in self.jobs]
    return state

  def load_saved_state(self, state):
    for job, job_state in zip(self.jobs, state["jobs"]):
      job.load_saved_state(job_state)
    self.current_time = state["current_time"]
    self.active_jobs = {job.name:job for job in self.jobs if job.name in state["active_jobs"]}
    self.job_completions = state["job_completions"]

  # accumulate events for `seconds` time and return the events
  def step(self, seconds):
    t_start = time.time()
    # step all active jobs for `seconds` time
    for job in self.active_jobs.values():
      job.step(seconds)
    # update current time
    self.current_time += seconds

    # check for completions
    completed_jobs = []
    for jobname, job in self.active_jobs.items():
      if job.status == JobStatus.COMPLETED:
        completed_jobs.append(jobname)
      # remove completed jobs
    for jobname in completed_jobs:
      job_obj = self.active_jobs[jobname]
      entry = {"name": jobname, "submission_time": job_obj.submission_time, 
               "queue_time": job_obj.queue_time, "jct": job_obj.time}
      self.job_completions.append(entry)
      self.active_jobs.pop(jobname)
    
    # new jobs to add
    new_jobs = {job.name:job for job in self.jobs if job.submission_time < self.current_time and job.status == JobStatus.INVALID}
    # rprint(f"New jobs: {list(new_jobs.keys())}")
    for job in new_jobs.values():
      # rprint(f"Adding job {job.name} to active jobs")
      self.active_jobs[job.name] = job
      job.status = JobStatus.QUEUED
    t_end = time.time()
    rprint(f"[cyan]EventRecorder::step took {(t_end - t_start)*1000:.2f} ms for {len(self.active_jobs)} active jobs[/cyan]")
    return {"arrivals": new_jobs, "completions": completed_jobs}
  
  def get_active_jobs(self):
    return self.active_jobs
  
  def get_completed_jobs(self):
    return [job for job in self.jobs if job.status == JobStatus.COMPLETED]
  
  def simulate_failure(self, node_id):
    pass