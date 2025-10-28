import subprocess

def get_gpu_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                             '--format=csv,nounits,noheader'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')
    usage = [(i, int(u.split(',')[0]) / int(u.split(',')[1]) * 100) for i, u in enumerate(lines)]
    return {i: round(percent, 2) for i, percent in usage}

def select_gpus(config):
    usage = get_gpu_usage()
    print("GPU Usage:")
    for i, p in usage.items():
        print(f"  GPU {i}: {p}% used")

    selected = list(usage.keys())

    if config.get("filter_by_threshold", True) and config.get("threshold") is not None:
        selected = [i for i in selected if usage[i] < config["threshold"]]

    if config.get("gpu_ids") is not None:
        selected = [i for i in selected if i in config["gpu_ids"]]

    if not selected:
        print("No GPUs match the criteria. Defaulting to GPU 0.")
        selected = [0]

    print(f"Selected GPUs: {selected}")
    return selected, len(selected)

