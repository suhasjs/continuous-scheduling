import pandas as pd
import csv
from argparse import ArgumentParser
from rich import print as rprint


argparser = ArgumentParser()
argparser.add_argument('--input-job-trace', type=str, default=None, help='Path to job trace file with adaptive jobs')
argparser.add_argument('--output-job-trace', type=str, default=None, help='Path to **output** job trace file')

args = argparser.parse_args()
input_job_trace_file = args.input_job_trace
output_job_trace_file = args.output_job_trace

with open(input_job_trace_file, 'r') as f:
  jobs_pd = pd.read_csv(f)

new_jobs_pd = pd.DataFrame(columns=['name', 'time', 'category', 'application', 'args'])
rows = []
for i, row in jobs_pd.iterrows():
  new_row = {'name': row['name'], 'time': row['time'], 'category': 'SiaJob', 
              'application': row['application'], 'args': None}
  rows.append(new_row)
new_jobs_pd = pd.DataFrame(rows)

with open(output_job_trace_file, 'w'):
  new_jobs_pd.to_csv(output_job_trace_file, index=False)