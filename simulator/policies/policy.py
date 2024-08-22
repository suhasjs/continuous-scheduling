class AbstractPolicy:
  def __init__(self):
    pass

  def update_job_utilities(self, new_job_utilities):
    pass

  def add_new_jobs(self, new_jobs):
    pass

  def remove_completed_jobs(self, completed_jobs):
    pass

  def step(self, seconds):
    pass
