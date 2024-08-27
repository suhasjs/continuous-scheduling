from policies.sia_ilp import SiaILP
from jobs.job import JobStatus
from jobs.sia_job import SiaJobClass
from jobs.sia_job import SiaJob
from event_recorder import EventRecorder

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

argparser = ArgumentParser()
argparser.add_argument('--job-trace', type=str, default=None, help='Path to job trace file')
argparser.add_argument('--round-duration', type=int, default=60, help='Duration of each round in seconds')
argparser.add_argument('--simulate-scheduler-delay', action='store_true', help='Whether to include scheduler latency in simulation: if True, the simulator will incorporate scheduler latency to next round duration [default: False]')

# parse args
args = argparser.parse_args()
round_duration = args.round_duration
simulate_scheduler_delay = args.simulate_scheduler_delay
job_trace_file = args.job_trace

# cluster configuration
cluster_nnodes = {"azure": 5, "aws": 16, "dgx-ext": 2, "quad": 1, "rtx": 3}
cluster_ngpus_per_node = {"aws": 4, "azure" : 8, "dgx-ext": 8, "quad" : 4, "rtx": 8}
total_cluster_gpus = {cluster: cluster_nnodes[cluster] * cluster_ngpus_per_node[cluster] for cluster in cluster_nnodes.keys()}

# base sia job classes
sia_job_classes = {}
sia_job_classes['cifar10'] = SiaJobClass("cifar10", "./jobs/profiles/cifar10.pkl", total_cluster_gpus)
sia_job_classes['bert'] = SiaJobClass("bert", "./jobs/profiles/bert.pkl", total_cluster_gpus)
sia_job_classes['imagenet'] = SiaJobClass("imagenet", "./jobs/profiles/imagenet.pkl", total_cluster_gpus)
sia_job_classes['deepspeech2'] = SiaJobClass("deepspeech2", "./jobs/profiles/deepspeech2.pkl", total_cluster_gpus)
sia_job_classes['yolov3'] = SiaJobClass("yolov3", "./jobs/profiles/yolov3.pkl", total_cluster_gpus)
sia_job_classes['ncf'] = SiaJobClass("ncf", "./jobs/profiles/ncf.pkl", total_cluster_gpus)

# load job trace
jobs = []
with open(job_trace_file, 'r') as f:
  jobs_pd = pd.read_csv(f)
for i, row in jobs_pd.iterrows():
  is_sia_job = True
  if is_sia_job:
    model_class = row['application']
    model_obj = sia_job_classes[model_class]
    job = SiaJob(row['name'], row['time'], model_obj)
  jobs.append(job)

# initialize event recorder
event_recorder = EventRecorder(jobs, cluster_nnodes, cluster_ngpus_per_node)

# initialize policy
sia_policy_options = {'lambda_no_alloc': 1.1, 'p_value': 0.5}
sia_solver_options = {'solver': 'GLPK_MI'}
policy = SiaILP(cluster_nnodes, cluster_ngpus_per_node, sia_policy_options, sia_solver_options)

# simulate till all jobs complete
all_jobs_complete = False
while event_recorder.current_time < 10000 or not all_jobs_complete:
  all_jobs_complete = all([job.status == JobStatus.COMPLETED for job in jobs])
  if all_jobs_complete:
    break
  
  # get changes to job set
  job_changes = event_recorder.step(round_duration)
  arrivals = job_changes['arrivals']
  completions = job_changes['completions']
  # make changes to policy
  policy.remove_completed_jobs(completions)
  policy.add_new_jobs(arrivals)

  # get updated utilities
  active_jobs = event_recorder.get_active_jobs()
  configs = policy.get_configurations()
  utilities = {}
  for jobname, job in active_jobs.items():
    utilities[jobname] = job.evaluate_allocations(configs)
  policy.update_job_utilities(utilities)

  # optimize allocations
  policy.step(round_duration)
  policy.optimize_allocations()
  new_allocs = policy.get_allocations()

  # update allocs for job
  for jobname, new_alloc in new_allocs.items():
    job = active_jobs[jobname]
    job.reallocate(new_alloc)

  table = Table(title=f"SIMULATOR TIME:{event_recorder.current_time}")
  table.add_column("Jobname", justify="left", style="cyan", no_wrap=True)
  table.add_column("Category", justify="left", style="cyan", no_wrap=True)
  table.add_column("Status", justify="left", style="white", no_wrap=True)
  table.add_column("Runtime", justify="left", style="white", no_wrap=True)
  table.add_column("Progress", justify="left", style="white", no_wrap=True)
  table.add_column("Restarts", justify="left", style="white", no_wrap=True)
  table.add_column("Allocation", justify="left", style="white", no_wrap=True)

  for jobname, job in active_jobs.items():
    progress_perc = round(job.progress / job.max_progress * 100, 2)
    table.add_row(jobname, "SiaJob", job.status.name, str(round(job.time, 1)), str(progress_perc), \
                  str(job.num_restarts), str(job.allocation))

  console = Console()
  console.print(table)
  # print resource consumption
  gpu_counts = {cluster: 0 for cluster in cluster_nnodes.keys()}
  for jobname, job in active_jobs.items():
    if job.allocation is not None:
      _, ngpus, cluster = job.allocation
      gpu_counts[cluster] += ngpus
  rprint(f"GPU usage:")
  for cluster, ngpus in gpu_counts.items():
    assert ngpus <= total_cluster_gpus[cluster], "GPU type: {cluster} overallocated: allocated={ngpus} > available={total_cluster_gpus[cluster]}"
    rprint(f"\t{cluster} = {ngpus} / {total_cluster_gpus[cluster]} GPUs ({round(ngpus / total_cluster_gpus[cluster] * 100, 2)}%)")
  # print JCTs for all completed jobs
  completed_jobs = event_recorder.get_completed_jobs()
  jcts_dict = {job.name: job.time for job in completed_jobs}
  avg_jct = np.mean(list(jcts_dict.values())) if len(jcts_dict) > 0 else 0
  rprint(f"Avg JCT: {avg_jct:.2f}")
  # rprint(f"JCTs: {jcts_dict}")
  rprint(f"--------------------------------------------")