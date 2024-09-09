from policies.sia_ilp import SiaILP
from jobs.job import JobStatus
from jobs.sia_job import get_sia_job_classes, SiaJob
from jobs.batch_inference_job import get_batch_inference_job_classes, BatchInferenceJob
from jobs.synthetic_single_phase_job import get_synthetic_job_classes, SyntheticSinglePhaseJob
from event_recorder import EventRecorder
from utils.solver_params import get_solver_params

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import pickle
from rich.console import Console
from rich.table import Table
from rich import print as rprint


argparser = ArgumentParser()
argparser.add_argument('--job-trace', type=str, default=None, help='Path to job trace file')
argparser.add_argument('--round-duration', type=int, default=60, help='Duration of each round in seconds')
argparser.add_argument('--cluster-scale', type=int, default=1, help='Scale factor for cluster size')
argparser.add_argument('--solver-timeout', type=int, default=1200, help='Timeout for solver (in seconds)')
argparser.add_argument('--solver-rtol', type=float, default=1e-4, help='Relative solution tolerance for solver')
argparser.add_argument('--solver-name', type=str, default="GLPK_MI", help='Solver to use for policy optimization')
argparser.add_argument('--simulator-timeout', type=float, default=-1, help='How many seconds of simulation to run (-1 for infinite)')
argparser.add_argument('--debug', action='store_true', help='Whether to pause after every simulator step')
argparser.add_argument('--output-log', type=str, help='Filename to log solver stats to', default=None)
argparser.add_argument('--disable-status', action='store_true', help='Whether to stop displaying status updates per step')
argparser.add_argument('--simulate-scheduler-delay', action='store_true', help='Whether to include scheduler latency in simulation: if True, the simulator will incorporate scheduler latency to next round duration [default: False]')

# parse args
args = argparser.parse_args()
round_duration = args.round_duration
simulate_scheduler_delay = args.simulate_scheduler_delay
job_trace_file = args.job_trace
cluster_scale = args.cluster_scale
solver_name = args.solver_name
solver_timeout = args.solver_timeout
simulator_timeout = args.simulator_timeout
if simulator_timeout < 0:
  simulator_timeout = 1e7
solver_rtol = args.solver_rtol
debug = args.debug
disable_status = args.disable_status

# cluster configuration
cluster_nnodes = {"azure": 5, "aws": 8, "dgx-ext": 4, "quad": 6, "rtx": 6, "a10-pcie": 8, "a100-pcie": 4}
cluster_ngpus_per_node = {"aws": 4, "azure" : 8, "dgx-ext": 8, "quad" : 4, "rtx": 8, "a10-pcie": 4, "a100-pcie": 4}
# cluster_nnodes = {"aws": 6, "dgx-ext": 2, "rtx": 3}
for cluster in cluster_nnodes.keys():
  cluster_nnodes[cluster] *= cluster_scale
# cluster_ngpus_per_node = {"aws": 4, "dgx-ext": 8, "rtx": 8}
total_cluster_gpus = {cluster: cluster_nnodes[cluster] * cluster_ngpus_per_node[cluster] for cluster in cluster_nnodes.keys()}
rprint(f"Cluster size: {sum(total_cluster_gpus.values())} GPUs")

# Objects for job classes
sia_job_classes = get_sia_job_classes(total_cluster_gpus)
batch_inference_job_classes = get_batch_inference_job_classes(total_cluster_gpus)
synthetic_job_classes = get_synthetic_job_classes(total_cluster_gpus)

# load job trace
jobs = []
with open(job_trace_file, 'r') as f:
  jobs_pd = pd.read_csv(f)
for i, row in jobs_pd.iterrows():
  is_sia_job = (row['category'] == 'SiaJob')
  is_batch_inference_job = (row['category'] == 'BatchInferenceJob')
  is_synthetic_job = (row['category'] == 'SyntheticSinglePhaseJob')
  if is_sia_job:
    model_class = row['application']
    model_obj = sia_job_classes[model_class]
    job = SiaJob(row['name'], row['time'], model_obj)
  elif is_batch_inference_job:
    model_class = row['application']
    model_obj = batch_inference_job_classes[model_class]
    job = BatchInferenceJob(row['name'], row['time'], model_obj)
  elif is_synthetic_job:
    model_class = row['application']
    model_obj = synthetic_job_classes[model_class]
    job = SyntheticSinglePhaseJob(row['name'], row['time'], model_obj)
  else:
    raise ValueError(f"Job category {row['category']} not supported")
  jobs.append(job)

# initialize event recorder
event_recorder = EventRecorder(jobs, cluster_nnodes, cluster_ngpus_per_node)

# initialize policy
sia_policy_options = {'lambda_no_alloc': 1.1, 'p_value': 0.5}
sia_solver_options = {'solver': solver_name}
sia_solver_options.update(get_solver_params(solver_name, solver_timeout, solver_rtol))
rprint(f"Policy solver options: {sia_solver_options}")
policy = SiaILP(cluster_nnodes, cluster_ngpus_per_node, sia_policy_options, sia_solver_options)

# simulate till all jobs complete
all_jobs_complete = False
time_stats = []
while event_recorder.current_time < simulator_timeout and not all_jobs_complete:
  print_str = ['-']*80
  print_str = "".join(print_str)
  rprint(print_str)
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
  # green text for arrivals
  rprint(f"\t[green]Arrivals: {list(arrivals.keys())}")
  rprint(f"\t[red]Completions: {completions}")

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

  # print status
  if not disable_status:
    table = Table(title=f"SIMULATOR TIME:{event_recorder.current_time}")
    table.add_column("Jobname", justify="left", style="cyan", no_wrap=True)
    table.add_column("Category", justify="left", style="cyan", no_wrap=True)
    table.add_column("Status", justify="left", style="white", no_wrap=True)
    table.add_column("Runtime", justify="left", style="white", no_wrap=True)
    table.add_column("Progress", justify="left", style="white", no_wrap=True)
    table.add_column("Restarts", justify="left", style="white", no_wrap=True)
    table.add_column("Allocation", justify="left", style="white", no_wrap=True)

    for jobname, job in active_jobs.items():
      progress_perc = str(round(job.progress / job.max_progress * 100, 2)) + "%"
      is_sia_job = isinstance(job, SiaJob)
      is_batch_inference_job = isinstance(job, BatchInferenceJob)
      is_synthetic_job = isinstance(job, SyntheticSinglePhaseJob)
      if is_sia_job:
        table.add_row(jobname, "SiaJob", job.status.name, str(round(job.time, 1)), progress_perc, \
                      str(job.num_restarts), str(job.allocation))
      elif is_batch_inference_job:
        table.add_row(jobname, "BatchInferenceJob", job.status.name, str(round(job.time, 1)), progress_perc, \
                      "-", str(job.allocation))
      elif is_synthetic_job:
        table.add_row(jobname, "SyntheticSinglePhaseJob", job.status.name, str(round(job.time, 1)), progress_perc, \
                      "-", str(job.allocation))
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
    assert ngpus <= total_cluster_gpus[cluster], f"GPU type: {cluster} overallocated: allocated={ngpus} > available={total_cluster_gpus[cluster]}"
    rprint(f"\t{cluster} = {ngpus} / {total_cluster_gpus[cluster]} GPUs ({round(ngpus / total_cluster_gpus[cluster] * 100, 2)}%)")
  # print JCTs for all completed jobs
  completed_jobs = event_recorder.get_completed_jobs()
  jcts_dict = {job.name: job.time for job in completed_jobs}
  avg_jct = np.mean(list(jcts_dict.values())) if len(jcts_dict) > 0 else 0
  rprint(f"Avg JCT: {avg_jct:.2f}, Active jobs: {len(active_jobs)}, Completed jobs: {len(completed_jobs)}")
  # rprint(f"JCTs: {jcts_dict}")
  
  if debug:
    key = input("Press any key to continue... [c to disable debug, x to exit]")
    if key == 'c':
      debug = False
    elif key == 'x':
      break

if args.output_log is not None:
  logfile_name = args.output_log
  with open(logfile_name, 'wb') as f:
    dump_dict = {"solver_stats": policy.solver_stats, "jcts": event_recorder.job_completions}
    pickle.dump(dump_dict, f)