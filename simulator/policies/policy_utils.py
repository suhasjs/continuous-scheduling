def round_allocations_largest(partial_allocations, cluster_free_gpus):
  # allocate the largest possible config to each job
  rounded_allocs = {}
  for jobname, partial_alloc in partial_allocations.items():
    alloced_gpus = 0
    for config, _ in partial_alloc:
      _, ngpus, cluster = config
      if cluster_free_gpus[cluster] >= ngpus:
        rounded_allocs[jobname] = config
        cluster_free_gpus[cluster] -= ngpus
        alloced_gpus = ngpus
        break
    if alloced_gpus == 0:
      rounded_allocs[jobname] = None
  return rounded_allocs