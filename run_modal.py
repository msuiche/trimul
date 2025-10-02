#!/usr/bin/env python3
"""
Modal app for running GPU benchmarks remotely on Modal's cloud infrastructure.
"""
import modal
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import dataclasses

# Create Modal app with a clear name
app = modal.App(name="trimul-gpu-benchmark")

# Define the GPU-enabled image with all required dependencies
# Using CUDA 12.4 which is compatible with PyTorch's CUDA 12.x builds
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")  # GCC and build tools
    .pip_install(
        "torch",
        "triton",
        "pyyaml",
        "numpy",
        "ninja",  # Required for PyTorch C++ extension compilation
    )
)


@app.function(
    image=gpu_image,
    gpu="H100",  # Use NVIDIA A100 GPU - closer to H100 architecture
    timeout=1800,  # 30 minutes timeout
)
def run_remote_benchmark(mode: str = "benchmark", task_file_content: str = None, sources: dict = None, verbose: bool = False):
    """
    Run benchmarks on Modal's GPU infrastructure.
    
    Args:
        mode: Execution mode ('test', 'benchmark', 'profile', 'leaderboard')
        task_file_content: YAML content of task configuration
        sources: Dictionary of source files to write
        verbose: Enable verbose output
    
    Returns:
        Dict containing results
    """
    import os
    import sys
    import tempfile
    from pathlib import Path
    import torch

    # Set CUDA_HOME - try to find nvcc
    cuda_paths = [
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/lib/cuda',
        os.path.join(os.path.dirname(torch.__file__), '_C')
    ]

    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ['CUDA_HOME'] = cuda_path
            break

    # Print debug info
    print(f"CUDA_HOME set to: {os.environ.get('CUDA_HOME', 'NOT SET')}", file=sys.stderr)

    # Create a temporary directory for our workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        sys.path.insert(0, tmpdir)
        
        # Write all source files to the temporary directory
        if sources:
            for filename, content in sources.items():
                Path(filename).write_text(content)
        
        # Import the actual evaluation code after writing files
        from run_eval import run_config, FullResult, make_system_info
        import yaml
        
        # Parse task configuration
        task_config = yaml.safe_load(task_file_content)
        
        # Prepare configuration  
        config = {
            'lang': task_config.get('lang', 'py'),
            'mode': mode,
            'sources': sources,
            'main': task_config.get('config', {}).get('main', 'eval.py'),
            'tests': task_config.get('tests', []),
            'benchmarks': task_config.get('benchmarks', []),
            'seed': task_config.get('seed'),
            'ranking_by': task_config.get('ranking_by', 'last'),
            'test_timeout': task_config.get('test_timeout', 180),
            'benchmark_timeout': task_config.get('benchmark_timeout', 180),
            'ranked_timeout': task_config.get('ranked_timeout', 180),
            'multi_gpu': task_config.get('multi_gpu', False),
        }
        
        # Add optional configuration if present
        optional_keys = ['arch', 'defines', 'include_dirs', 'libraries', 'headers']
        for key in optional_keys:
            if key in task_config:
                config[key] = task_config[key]
        
        # Run the configuration
        result = run_config(config)
        
        # Convert result to dictionary for serialization
        return dataclasses.asdict(result)


@app.function(
    image=gpu_image,
    gpu="H100",  # Use A100 for consistency
    timeout=60,
)
def check_gpu_availability():
    """Check if GPU is available and return system information."""
    import torch
    
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    return info


@app.local_entrypoint()
def main():
    """
    Main entry point for running benchmarks on Modal.
    This follows the Modal pattern of using @app.local_entrypoint() 
    to define the function that starts locally when we invoke modal run.
    
    Since Modal CLI doesn't pass arguments, use environment variables:
    - MODAL_MODE: test, benchmark, profile, or leaderboard (default: benchmark)  
    - MODAL_VERBOSE: set to "true" for verbose output
    - MODAL_CHECK_GPU: set to "true" to just check GPU
    """
    import os
    
    # Get configuration from environment variables
    mode = os.environ.get('MODAL_MODE', 'benchmark')
    verbose = os.environ.get('MODAL_VERBOSE', '').lower() == 'true'
    check_gpu = os.environ.get('MODAL_CHECK_GPU', '').lower() == 'true'
    json_output = os.environ.get('MODAL_JSON', '').lower() == 'true'
    task_file = os.environ.get('MODAL_TASK', 'task.yml')
    
    # Print usage if help requested
    if os.environ.get('MODAL_HELP', '').lower() == 'true':
        print("Usage: modal run run_modal.py")
        print("\nEnvironment variables:")
        print("  MODAL_MODE=<test|benchmark|profile|leaderboard>  # default: benchmark")
        print("  MODAL_VERBOSE=true                               # enable verbose output")
        print("  MODAL_CHECK_GPU=true                             # just check GPU availability")
        print("  MODAL_JSON=true                                  # output as JSON")
        print("  MODAL_TASK=<path>                                # task file (default: task.yml)")
        print("\nExamples:")
        print("  modal run run_modal.py                           # run benchmark")
        print("  MODAL_MODE=test modal run run_modal.py           # run tests")
        print("  MODAL_CHECK_GPU=true modal run run_modal.py      # check GPU")
        return
    
    if check_gpu:
        print("✓ Checking GPU availability on Modal...")
        gpu_info = check_gpu_availability.remote()
        if json_output:
            print(json.dumps(gpu_info, indent=2))
        else:
            print("\n✓ GPU Information on Modal:")
            print(f"  PyTorch Version: {gpu_info['pytorch_version']}")
            print(f"  CUDA Available: {gpu_info['cuda_available']}")
            if gpu_info['cuda_available']:
                print(f"  CUDA Version: {gpu_info['cuda_version']}")
                print(f"  Device Count: {gpu_info['device_count']}")
                print(f"  Device Name: {gpu_info['device_name']}")
        print("✓ GPU check completed.")
        return
    
    # Load task configuration
    if not Path(task_file).exists():
        print(f"Error: Task file '{task_file}' not found")
        sys.exit(1)
    
    with open(task_file, 'r') as f:
        task_content = f.read()
    
    import yaml
    task_config = yaml.safe_load(task_content)
    
    # Prepare source files to send to Modal
    sources = {}
    
    # First, add run_eval.py as it's needed for imports
    with open('run_eval.py', 'r') as f:
        sources['run_eval.py'] = f.read()
    
    # Add all files specified in task configuration
    for file_info in task_config.get('files', []):
        file_name = file_info['name']
        file_source = file_info['source']
        
        # Read source files
        if not file_source.startswith('@') and Path(file_source).exists():
            with open(file_source, 'r') as f:
                sources[file_name] = f.read()
        elif file_source == '@SUBMISSION@':
            # Use submission.py if it exists
            if Path('submission.py').exists():
                with open('submission.py', 'r') as f:
                    sources[file_name] = f.read()
            else:
                print(f"Error: submission.py not found")
                sys.exit(1)
    
    print(f"✓ Initialized Modal app: trimul-gpu-benchmark")
    print(f"✓ Running {mode} benchmarks on Modal GPU...")
    print("  This may take a few minutes for the first run while Modal sets up the environment.\n")
    
    try:
        # Run benchmark remotely using .remote() call pattern
        result_dict = run_remote_benchmark.remote(
            mode=mode,
            task_file_content=task_content,
            sources=sources,
            verbose=verbose
        )
        
        if json_output:
            print(json.dumps(result_dict, indent=2, default=str))
        else:
            # Pretty print results
            print_modal_results(result_dict, verbose)
        
        print("✓ App completed.")
        
        # Exit with appropriate code
        if not result_dict['success']:
            print(f"\nError: {result_dict.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error running benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_modal_results(result: Dict, verbose: bool = False):
    """Pretty print the benchmark results from Modal."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS (Modal GPU)")
    print("="*60)
    
    # System information
    if 'system' in result and result['system']:
        system = result['system']
        print("\nSystem Information:")
        print(f"  GPU: {system.get('gpu', 'N/A')}")
        print(f"  Device Count: {system.get('device_count', 'N/A')}")
        print(f"  Runtime: {system.get('runtime', 'N/A')}")
        print(f"  Platform: {system.get('platform', 'N/A')}")
        if system.get('torch'):
            print(f"  PyTorch: {system.get('torch')}")
    
    # Results for each run mode
    if 'runs' in result:
        for mode, eval_result in result['runs'].items():
            if not eval_result:
                continue
                
            print(f"\n{mode.upper()} Results:")
            print("-" * 40)
            
            # Compilation results
            if 'compilation' in eval_result and eval_result['compilation']:
                comp = eval_result['compilation']
                print(f"  Compilation: {'✓ Success' if comp.get('success') else '✗ Failed'}")
                if verbose and comp.get('stderr'):
                    print(f"  Compiler output: {comp['stderr'][:200]}...")
            
            # Run results
            if 'run' in eval_result and eval_result['run']:
                run = eval_result['run']
                print(f"  Execution: {'✓ Success' if run.get('success') else '✗ Failed'}")
                print(f"  Tests Passed: {'✓ Yes' if run.get('passed') else '✗ No'}")
                print(f"  Execution Time: {run.get('duration', 0):.3f} seconds")
                
                if run.get('result'):
                    result_data = run['result']
                    # Parse benchmark results
                    if 'benchmark-count' in result_data:
                        count = int(result_data['benchmark-count'])
                        print(f"  Benchmarks Run: {count}")
                        means_ms = []
                        
                        for i in range(count):
                            prefix = f"benchmark.{i}"
                            if f"{prefix}.spec" in result_data:
                                print(f"\n  Benchmark #{i+1}: {result_data[f'{prefix}.spec']}")
                                if f"{prefix}.mean" in result_data:
                                    mean_ns = float(result_data[f"{prefix}.mean"])  # ns
                                    std_ns = float(result_data.get(f"{prefix}.std", 0))
                                    best_ns = float(result_data.get(f"{prefix}.best", mean_ns))
                                    worst_ns = float(result_data.get(f"{prefix}.worst", mean_ns))
                                    runs = int(result_data.get(f"{prefix}.runs", 1))
                                    
                                    mean_ms = mean_ns / 1e6
                                    means_ms.append(mean_ms)

                                    print(f"    Mean: {mean_ms:.3f} ms")
                                    print(f"    Std Dev: {std_ns/1e6:.3f} ms")
                                    print(f"    Best: {best_ns/1e6:.3f} ms")
                                    print(f"    Worst: {worst_ns/1e6:.3f} ms")
                                    print(f"    Runs: {runs}")
                                elif f"{prefix}.error" in result_data:
                                    print(f"    Error: {result_data[f'{prefix}.error']}")

                        # Print geometric mean if we gathered any means
                        if means_ms:
                            import math
                            # Filter out any non-positive means defensively
                            vals = [m for m in means_ms if m > 0]
                            if vals:
                                geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
                                print(f"\n  Geometric Mean: {geo:.3f} ms")
                    
                    # Parse test results
                    elif 'test-count' in result_data:
                        count = int(result_data['test-count'])
                        print(f"  Tests Run: {count}")
                        passed_count = 0
                        failed_count = 0
                        
                        for i in range(count):
                            prefix = f"test.{i}"
                            if f"{prefix}.status" in result_data:
                                status = result_data[f"{prefix}.status"]
                                if status == "pass":
                                    passed_count += 1
                                else:
                                    failed_count += 1
                                    if verbose:
                                        spec = result_data.get(f"{prefix}.spec", f"Test {i}")
                                        error = result_data.get(f"{prefix}.error", "Unknown error")
                                        print(f"\n  Failed Test: {spec}")
                                        print(f"    Error: {error}")
                        
                        print(f"  Passed: {passed_count}/{count}")
                        if failed_count > 0:
                            print(f"  Failed: {failed_count}/{count}")
                
                if verbose and run.get('stderr'):
                    stderr_out = run['stderr']
                    if len(stderr_out) > 5000:
                        print(f"\n  Error output (first 2500 chars):\n{stderr_out[:2500]}")
                        print(f"\n  ... ({len(stderr_out) - 5000} chars omitted) ...\n")
                        print(f"\n  Error output (last 2500 chars):\n{stderr_out[-2500:]}")
                    else:
                        print(f"\n  Error output:\n{stderr_out}")
    
    print("\n" + "="*60)
