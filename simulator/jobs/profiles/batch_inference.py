imagenet_resnet50 = {
  "name" : "ImagenetResnet50",
  "num_iters" : 1431167, # 1.28M train, 50k validation + 100k test
  "gpu_profiles" : {
    "azure" : {"min_gpus": 1, "batch_size" : 256, "throughput": 635},
    "aws" : {"min_gpus": 1, "batch_size" : 128, "throughput": 401},
    "dgx-ext" : {"min_gpus": 1, "batch_size": 384, "throughput": 2343},
    "a100-pcie" : {"min_gpus": 1, "batch_size": 384, "throughput": 2343},
    "a10-pcie" : {"min_gpus": 1, "batch_size": 192, "throughput": 825},
    "quadro" : {"min_gpus": 1, "batch_size": 192, "throughput": 489},
    "rtx" : {"min_gpus": 1, "batch_size" : 64, "throughput": 440}
  },
  'sim_speedup' : 10 # 10x speedup relative to wall-clock time
}

llama_8b_wikipedia = {
  "name" : "Llama-8B-wiki",
  "num_iters" : 100*(10**6), # 100M tokens
  "gpu_profiles" : {
    "azure" : {"min_gpus": 2, "batch_size" : 32, "throughput": 5100},
    "aws" : {"min_gpus": 4, "batch_size" : 16, "throughput": 3800},
    "dgx-ext" : {"min_gpus": 1, "batch_size": 40, "throughput": 5424},
    "a100-pcie" : {"min_gpus": 1, "batch_size": 40, "throughput": 5400},
    "a10-pcie" : {"min_gpus": 2, "batch_size": 24, "throughput": 5400},
    "quadro" : {"min_gpus": 2, "batch_size": 24, "throughput": 4000},
    "rtx" : {"min_gpus": 4, "batch_size" : 11, "throughput": 8170}
  },
  'sim_speedup' : 1 # 10x speedup relative to wall-clock time
}

llama_8b_commoncrawl = {
  "name" : "Llama-8B-CC",
  "num_iters" : 1.2*(10**12), # 1.2T tokens
  "gpu_profiles" : {
    "azure" : {"min_gpus": 2, "batch_size" : 32, "throughput": 5100},
    "aws" : {"min_gpus": 4, "batch_size" : 16, "throughput": 3800},
    "dgx-ext" : {"min_gpus": 1, "batch_size": 40, "throughput": 5424},
    "a100-pcie" : {"min_gpus": 1, "batch_size": 40, "throughput": 5400},
    "a10-pcie" : {"min_gpus": 2, "batch_size": 24, "throughput": 5400},
    "quadro" : {"min_gpus": 2, "batch_size": 24, "throughput": 4000},
    "rtx" : {"min_gpus": 4, "batch_size" : 11, "throughput": 8170}
  },
  'sim_speedup' : 500 # 50,000x speedup relative to wall-clock time
}