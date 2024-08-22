import numpy as np
import pandas as pd
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--job-trace', type=str, default=None, help='Path to job trace file')
argparser.add_argument('--round-duration', type=int, default=60, help='Duration of each round in seconds')
argparser.add_argument('--simulate-scheduler-delay', action='store_true', help='Whether to include scheduler latency in simulation: if True, the simulator will incorporate scheduler latency to next round duration [default: False]')

args = argparser.parse_args()
round_duration = args.round_duration
simulate_scheduler_delay = args.simulate_scheduler_delay
job_trace_file = args.job_trace

# load job trace
