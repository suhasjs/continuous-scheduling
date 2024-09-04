# modeled after trace sampler written by Aurick Qiao for Pollux experiments
import argparse
import numpy as np
import os
import pandas
import random
from rich import print as rprint

from datetime import datetime, timedelta

trace_file_names = {
  "philly": "philly_normalized.csv",
  "helios-saturn": "helios_saturn_normalized.csv",
  "alibaba-pai": "alibaba_pai_normalized.csv"
}


def generate(num_jobs, start=0, duration=24, seed=0, trace="philly", oversample=False):
  trace_file = trace_file_names[trace]
  trace_csv = os.path.join(os.path.dirname(__file__), "../traces/", trace_file)
  rprint(f"Reading trace from {trace_csv}")
  # read trace from file
  trace = pandas.read_csv(trace_csv)
  rprint(f"Read {len(trace)} jobs")

  # filter out jobs that are too short or too long
  trace = trace[trace.duration >= 15]
  trace = trace[trace.gpu_time < 1000 * 3600]
  rprint(f"Filtered to {len(trace)} jobs")

  start_tstamp = start*3600
  end_tstamp = start_tstamp + duration*3600
  trace = trace[trace.time >= start_tstamp]
  trace = trace[trace.time <= end_tstamp]
  print(f"Trace limits: {start_tstamp} -> {end_tstamp} seconds, # jobs: {len(trace)}, oversample={oversample}")
  if len(trace) < num_jobs and not oversample:
    raise ValueError(f"Trace has {len(trace)} jobs, which is less than the requested {num_jobs} jobs. Use --oversample to oversample the trace.")
  workload_rngs = [np.random.RandomState(seed + i) for i in range(1, 4)]
  category_rngs = [np.random.RandomState(seed + i) for i in range(5, 10)]
  row_rng = np.random.RandomState(seed)
  jobs = []
  # sample with replacement if len(trace) < num_jobs
  row_idxs = row_rng.choice(len(trace), num_jobs, replace=(len(trace) < num_jobs))

  for row_idx in row_idxs:
      row = trace.iloc[row_idx]
      new_job = {"time": row.time}
      # <0.1 GPU hrs: SiaJob[cifar10*0.8, ncf*0.2], BatchInferenceJob[imagenet_resnet50]
      if row.gpu_time < 0.1 * 3600:
        category_choice = category_rngs[0].choice(["SiaJob", "BatchInferenceJob"], p=[0.85, 0.15])
        if category_choice == "SiaJob":
          new_job["category"] = "SiaJob"
          new_job["application"] = workload_rngs[0].choice(["cifar10", "ncf"], p=[0.8, 0.2])
        else:
          new_job["category"] = "BatchInferenceJob"
          new_job["application"] = "imagenet_resnet50"
      # <1 GPU hr: SiaJob[bert], BatchInferenceJob[llama_8b_wikipedia], SyntheticSinglePhaseJob[synthetic_mixed_short]
      elif row.gpu_time < 1*3600:
        category_choice = category_rngs[1].choice(["SiaJob", "BatchInferenceJob", "SyntheticSinglePhaseJob"], 
                                                  p=[0.7, 0.2, 0.1])
        if category_choice == "SiaJob":
          new_job["category"] = "SiaJob"
          new_job["application"] = "bert"
        elif category_choice == "BatchInferenceJob":
          new_job["category"] = "BatchInferenceJob"
          new_job["application"] = "llama_8b_wikipedia"
        else:
          new_job["category"] = "SyntheticSinglePhaseJob"
          new_job["application"] = "synthetic_mixed_short"
      # <5 GPU hrs: SiaJob[deepspeech2], SyntheticSinglePhaseJob[synthetic_linear_short, synthetic_sublinear_short]
      elif row.gpu_time < 5*3600:
        category_choice = category_rngs[2].choice(["SiaJob", "SyntheticSinglePhaseJob"], p=[0.6, 0.4])
        if category_choice == "SiaJob":
          new_job["category"] = "SiaJob"
          new_job["application"] = "deepspeech2"
        else:
          new_job["category"] = "SyntheticSinglePhaseJob"
          new_job["application"] = workload_rngs[1].choice(["synthetic_linear_short", "synthetic_sublinear_short"])
      # <10 GPU hrs: SiaJob[yolov3], BatchInferenceJob[llama_8b_commoncrawl]
      elif row.gpu_time < 10 * 3600:
        category_choice = category_rngs[3].choice(["SiaJob", "BatchInferenceJob"], p=[0.9, 0.1])
        if category_choice == "SiaJob":
          new_job["category"] = "SiaJob"
          new_job["application"] = "yolov3"
        else:
          new_job["category"] = "BatchInferenceJob"
          new_job["application"] = "llama_8b_commoncrawl"
      # <100 GPU hrs: SiaJob[imagenet], SyntheticSinglePhaseJob[synthetic_mixed_long]
      elif row.gpu_time < 100 * 3600:
        category_choice = category_rngs[4].choice(["SiaJob", "SyntheticSinglePhaseJob"], p=[0.5, 0.5])
        if category_choice == "SiaJob":
          new_job["category"] = "SiaJob"
          new_job["application"] = "imagenet"
        else:
          new_job["category"] = "SyntheticSinglePhaseJob"
          new_job["application"] = "synthetic_mixed_long"
      # <1000 GPU hrs: SyntheticSinglePhaseJob[synthetic_linear_long, synthetic_sublinear_long]
      elif row.gpu_time < 1000 * 3600:
        new_job["category"] = "SyntheticSinglePhaseJob"
        new_job["application"] = workload_rngs[2].choice(["synthetic_linear_long", "synthetic_sublinear_long"])
      else:
        raise ValueError(f"Job with {row.gpu_time} GPU seconds not supported")
      jobs.append(new_job)
  jobs.sort(key=lambda v: v["time"])

  for idx, rec in enumerate(jobs):
      rec["name"] = "{}-{}".format(rec["application"], idx)
  jobs_df = pandas.DataFrame(jobs, columns=("name", "time", "category", "application"))
  # ensure first job is submitted at t=1 sec
  jobs_df['time'] = (jobs_df['time'] - jobs_df['time'].min()) + 1
  return jobs_df


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--start", type=int, default=2, help="starting hour")
  parser.add_argument("--duration", type=int, default=8, help="total number of workload hours")
  parser.add_argument("--num-jobs", type=int, default=160, help="total number of jobs")
  parser.add_argument("--oversample", action="store_true", help="oversample the trace if number of jobs in the trace is less than num-jobs")
  parser.add_argument("--output", type=str, help="path to output the workload")
  parser.add_argument("--seed", type=int, default=0, help="random seed")
  parser.add_argument("--trace", type=str, default="philly", help="which trace to use [philly, helios-saturn, alibaba-pai]")
  args = parser.parse_args()
  workload = generate(args.num_jobs, start=args.start, duration=args.duration, \
                      seed=args.seed, trace=args.trace, oversample=args.oversample)
  csv = workload.set_index("name").to_csv(args.output)
  if csv:
    print(csv)
  print(workload.groupby(["category", "application"]).size().reset_index(name="count"))
