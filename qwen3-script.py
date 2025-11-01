"""
Qwen3-8B-Q5_K_M GGUF Model Launcher
Optimized for CUDA 13.0 Update 2 and Python 3.13
qwen3-script.py
"""

# %% [1.0] Dependencies and Environment Check
import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Qwen3-8B GGUF Model Launcher - CUDA 13 Edition")
print("="*60)

# System info
print(f"\nSystem Information:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Platform: {platform.platform()}")
print(f"  Processor: {platform.processor()}")

# Check Python version
py_version = sys.version_info
if py_version.major == 3 and py_version.minor >= 13:
    print(f"  Python {py_version.major}.{py_version.minor} (optimized for 3.13+)")
elif py_version.major == 3 and py_version.minor >= 10:
    print(f"  Python {py_version.major}.{py_version.minor} (supported)")
else:
    print(f"  Warning: Python {py_version.major}.{py_version.minor} (upgrade to 3.13 recommended)")

# Check llama-cpp-python
try:
    from llama_cpp import Llama
    import llama_cpp
    print(f"  llama-cpp-python: installed {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else ''}")
    print("  Note: Using direct CUDA via cuBLAS (no PyTorch needed)")
except ImportError:
    print("  llama-cpp-python: NOT FOUND")
    print("\n  Requires llama-cpp-python compiled with CUDA 13 support")
    sys.exit(1)

# %% [1.1] GPU/CUDA Detection  
print(f"\n" + "="*60)
print("GPU/CUDA 13 Diagnostics")
print("="*60)

def get_nvidia_info():
    """Get comprehensive NVIDIA GPU and CUDA 13 information"""
    gpu_info = {}
    
    try:
        # Check CUDA 13 specific version
        cuda_check = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'],
            capture_output=True, text=True, shell=True
        )
        
        if cuda_check.returncode == 0:
            driver, cuda = cuda_check.stdout.strip().split(', ')
            gpu_info['driver'] = driver
            gpu_info['cuda'] = cuda
            
            # Check for CUDA 13 compatibility
            cuda_major = int(cuda.split('.')[0]) if '.' in cuda else 0
            if cuda_major >= 13:
                print(f"  CUDA {cuda} (13.0+ detected)")
                print(f"  Driver {driver} (580+ required for CUDA 13)")
            elif cuda_major == 12:
                print(f"  Warning: CUDA {cuda} (upgrade to 13.0 recommended)")
                print(f"  Driver: {driver}")
            else:
                print(f"  Warning: CUDA {cuda} (CUDA 13.0 strongly recommended)")
                print(f"  Driver: {driver}")
        
        # Get detailed GPU information using nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get additional details specific to newer architectures
            gpu_info['cuda_cores'] = pynvml.nvmlDeviceGetNumGpuCores(handle) if hasattr(pynvml, 'nvmlDeviceGetNumGpuCores') else 'N/A'
            
            # Get compute capability - important for CUDA 13 (needs 7.5+)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            gpu_info['compute_cap'] = f"{major}.{minor}"
            
            if major >= 7 and minor >= 5:
                print(f"  Compute Capability: {major}.{minor} (Turing or newer)")
            else:
                print(f"  Warning: Compute Capability: {major}.{minor} (may not support all CUDA 13 features)")
                
            gpu_info['pcie_gen'] = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle) if hasattr(pynvml, 'nvmlDeviceGetMaxPcieLinkGeneration') else 'N/A'
            gpu_info['pcie_width'] = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle) if hasattr(pynvml, 'nvmlDeviceGetMaxPcieLinkWidth') else 'N/A'
            
            pynvml.nvmlShutdown()
        except:
            # Try to get compute capability via nvidia-smi
            cc_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, shell=True
            )
            if cc_result.returncode == 0:
                compute_cap = cc_result.stdout.strip()
                gpu_info['compute_cap'] = compute_cap
                major, minor = map(int, compute_cap.split('.'))
                if major >= 7 and minor >= 5:
                    print(f"  Compute Capability: {compute_cap} (Turing or newer)")
                else:
                    print(f"  Warning: Compute Capability: {compute_cap} (upgrade GPU recommended)")
        
        # Get comprehensive GPU info via nvidia-smi with CUDA 13 focus
        queries = [
            ('Basic Info', 'name,compute_cap,driver_version,cuda_version', ['GPU Model', 'Compute Capability', 'Driver Version', 'CUDA Version']),
            ('Memory Info', 'memory.total,memory.free,memory.used', ['Total Memory', 'Free Memory', 'Used Memory']),
            ('Clock Speeds', 'clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory', 
             ['Current GPU Clock', 'Max GPU Clock', 'Current Mem Clock', 'Max Mem Clock']),
            ('Power/Thermal', 'power.draw,power.limit,temperature.gpu,temperature.memory', 
             ['Power Draw', 'Power Limit', 'GPU Temp', 'Memory Temp']),
            ('Utilization', 'utilization.gpu,utilization.memory', ['GPU Utilization', 'Memory Controller']),
            ('PCIe Info', 'pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max', 
             ['Current PCIe Gen', 'Max PCIe Gen', 'Current Width', 'Max Width']),
        ]
        
        for category, query, labels in queries:
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, shell=True
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                print(f"\n{category}:")
                for label, value in zip(labels, values):
                    # Format specific values
                    if 'Memory' in label and value.replace('.','').isdigit():
                        value_gb = float(value) / 1024 if float(value) > 1024 else float(value)
                        unit = 'GB' if float(value) > 1024 else 'MB'
                        print(f"  {label}: {value_gb:.2f} {unit} ({value} MiB)")
                    elif 'Clock' in label and value.replace('.','').isdigit():
                        print(f"  {label}: {value} MHz")
                    elif 'Power' in label and value.replace('.','').isdigit():
                        print(f"  {label}: {value} W")
                    elif 'Temp' in label:
                        print(f"  {label}: {value}°C")
                    elif 'Utilization' in label or 'Controller' in label:
                        print(f"  {label}: {value}%")
                    elif 'PCIe' in label and 'Width' in label:
                        print(f"  {label}: x{value}")
                    elif label == 'Compute Capability':
                        # Already handled above, skip duplicate
                        continue
                    else:
                        print(f"  {label}: {value}")
                
                # Store for later use
                for label, value in zip(labels, values):
                    gpu_info[label.lower().replace(' ', '_')] = value
        
        # Get memory bus width and calculate bandwidth - important for CUDA 13 memory ops
        bus_result = subprocess.run(
            ['nvidia-smi', '-q', '-d', 'MEMORY'],
            capture_output=True, text=True, shell=True
        )
        
        if bus_result.returncode == 0:
            output_lines = bus_result.stdout.split('\n')
            for line in output_lines:
                if 'Bus Width' in line:
                    bus_width = line.split(':')[1].strip()
                    print(f"\nMemory Architecture:")
                    print(f"  Memory Bus Width: {bus_width}")
                    gpu_info['memory_bus_width'] = bus_width
                    
                    # Calculate theoretical bandwidth
                    if 'current_mem_clock' in gpu_info and gpu_info['current_mem_clock'].isdigit():
                        mem_clock_mhz = float(gpu_info['current_mem_clock'])
                        bus_width_bits = int(bus_width.replace('bits', '').strip())
                        # Bandwidth (GB/s) = (memory_clock_MHz × bus_width_bits × 2) / 8 / 1000
                        bandwidth_gbps = (mem_clock_mhz * bus_width_bits * 2) / 8 / 1000
                        print(f"  Theoretical Bandwidth: {bandwidth_gbps:.1f} GB/s")
                        
                        # CUDA 13 specific: Check if bandwidth is sufficient for large models
                        if bandwidth_gbps > 500:
                            print(f"  High bandwidth suitable for large models")
                        elif bandwidth_gbps > 300:
                            print(f"  Good bandwidth for medium models")
                        else:
                            print(f"  Warning: Limited bandwidth may affect performance")
        
        # Performance state - CUDA 13 can better manage P-states
        pstate_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pstate', '--format=csv,noheader'],
            capture_output=True, text=True, shell=True
        )
        
        if pstate_result.returncode == 0:
            pstate = pstate_result.stdout.strip()
            print(f"\nPerformance State: {pstate}")
            if pstate == 'P0':
                print("  Maximum performance mode")
            elif pstate in ['P1', 'P2']:
                print("  High performance mode")
            else:
                print("  Warning: Power saving mode (may affect performance)")
        
        # Check for CUDA 13 specific features
        print(f"\nCUDA 13 Feature Support:")
        if gpu_info.get('compute_cap'):
            major, minor = map(int, gpu_info['compute_cap'].split('.'))
            
            # Blackwell (sm_120) support
            if major >= 12:
                print("  Blackwell architecture (full CUDA 13 features)")
            # Hopper (sm_90)
            elif major == 9:
                print("  Hopper architecture (full CUDA 13 support)")
            # Ada Lovelace (sm_89)
            elif major == 8 and minor >= 9:
                print("  Ada Lovelace (full CUDA 13 support)")
            # Ampere (sm_80, sm_86)
            elif major == 8 and minor >= 0:
                print("  Ampere (CUDA 13 supported)")
            # Turing (sm_75)
            elif major == 7 and minor >= 5:
                print("  Turing (CUDA 13 supported, some features limited)")
            else:
                print("  Warning: Older architecture (limited CUDA 13 support)")
            
    except Exception as e:
        print(f"\nWarning: Could not get complete GPU info")
        print(f"  Error: {e}")
        print("  Ensure NVIDIA drivers 580+ are installed for CUDA 13")
        gpu_info['error'] = str(e)
    
    # CUDA 13 compilation check for llama.cpp
    try:
        import llama_cpp
        if hasattr(llama_cpp, 'llama_backend_init'):
            print(f"\nllama.cpp CUDA backend available")
            print("  Using cuBLAS for acceleration (no PyTorch required)")
        else:
            print(f"\nWarning: llama.cpp may not have CUDA 13 support")
            print("  Recompile with CUDA 13 toolkit for best performance")
    except:
        pass
    
    return gpu_info

gpu_info = get_nvidia_info()

# Memory requirement estimation with CUDA 13 optimizations
print(f"\n" + "="*60)
print("Model Memory Requirements (CUDA 13 Optimized)")
print("="*60)
print(f"  Model: Qwen3-8B-Q5_K_M (5.85 GB file)")
print(f"  Context: 8192 tokens")
print(f"  Estimated VRAM: ~7-8 GB (CUDA 13 memory compression may reduce)")
print(f"  Recommended free VRAM: >8 GB")

if gpu_info.get('free_memory'):
    free_gb = float(gpu_info['free_memory']) / 1024
    if free_gb < 8:
        print(f"\nWarning: Only {free_gb:.2f} GB VRAM free")
        print("  CUDA 13 memory compression will be utilized")
    else:
        print(f"\n{free_gb:.2f} GB VRAM available - excellent")

if not gpu_info.get('gpu_model') and not gpu_info.get('error'):
    print("\nWarning: No NVIDIA GPU detected - CPU mode will be slow!")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

# %% [2.0] Configuration
class Config:
    # Model location
    MODEL_PATH = r"E:\ai\models\qwen3-8b-gguf\qwen3-8b-q5_k_m.gguf"
    
    # GPU/CPU Toggle
    USE_GPU = True           # Set to False to force CPU mode (not recommended)
    
    # Model settings optimized for CUDA 13
    N_CTX = 8192            # Context window
    N_GPU_LAYERS = -1       # -1 = all layers on GPU (CUDA 13 optimized)
    N_BATCH = 512           # Batch size (CUDA 13 can handle larger batches)
    N_THREADS = 16          # CPU threads (only if forced to CPU mode)
    
    # Generation defaults
    TEMPERATURE = 0.7
    TOP_P = 0.95
    TOP_K = 40
    REPEAT_PENALTY = 1.1
    MAX_TOKENS = 2048
    
    # Memory settings - CUDA 13 specific
    USE_MMAP = True         # Memory-mapped files
    USE_MLOCK = False       # Lock in RAM (set True if 32GB+ RAM)
    VERBOSE = False         # Show llama.cpp output
    
    # GPU specific settings for CUDA 13
    TENSOR_SPLIT = None     # For multi-GPU (e.g., [0.5, 0.5] for 2 GPUs)
    MAIN_GPU = 0           # Primary GPU index
    
    # CUDA 13 optimizations
    CUDA_GRAPHS = True      # Enable CUDA graphs for better performance
    FLASH_ATTENTION = True  # Use flash attention if available

# Apply GPU/CPU toggle
if not Config.USE_GPU:
    Config.N_GPU_LAYERS = 0
    print("\nWarning: CPU mode enabled - performance will be significantly reduced")
    print("  CUDA 13 GPU acceleration is strongly recommended")

print(f"\nModel Configuration (CUDA 13 Optimized):")
print(f"  Model: {Path(Config.MODEL_PATH).name}")
print(f"  Context: {Config.N_CTX} tokens")
print(f"  Batch Size: {Config.N_BATCH}")
print(f"  Mode: {'GPU (CUDA 13)' if Config.USE_GPU else 'CPU'}")
if Config.USE_GPU:
    print(f"  GPU Layers: {'All' if Config.N_GPU_LAYERS == -1 else Config.N_GPU_LAYERS}")
    print(f"  Main GPU: {Config.MAIN_GPU}")
    print(f"  CUDA Graphs: {'Enabled' if Config.CUDA_GRAPHS else 'Disabled'}")
    print(f"  Flash Attention: {'Enabled' if Config.FLASH_ATTENTION else 'Disabled'}")

# %% [3.0] Model Validation
def validate_model():
    """Check if model file exists"""
    model_path = Path(Config.MODEL_PATH)
    
    if not model_path.exists():
        print(f"\nModel not found at: {Config.MODEL_PATH}")
        print("\nTo fix:")
        print("1. Download from: https://huggingface.co/Qwen/Qwen3-8B-GGUF")
        print("2. Get file: Qwen3-8B-Q5_K_M.gguf (5.85 GB)")
        print(f"3. Save to: {Config.MODEL_PATH}")
        return False
    
    size_gb = model_path.stat().st_size / (1024**3)
    print(f"\nModel found ({size_gb:.2f} GB)")
    return True

if not validate_model():
    print("\nExiting - model file required")
    sys.exit(1)

# %% [4.0] Model Initialization
print("\n" + "="*60)
print("Loading Model")
print("="*60)

# Pre-initialization GPU memory check
if Config.USE_GPU and gpu_info.get('free_memory'):
    free_gb = float(gpu_info.get('free_memory', '0')) / 1024
    total_gb = float(gpu_info.get('total_memory', '0')) / 1024 if gpu_info.get('total_memory') else 0
    used_gb = total_gb - free_gb
    
    print(f"Pre-load VRAM Status:")
    print(f"  Available: {free_gb:.2f} GB / {total_gb:.2f} GB")
    print(f"  In Use: {used_gb:.2f} GB")
    
    # Visual bar for memory
    if total_gb > 0:
        usage_percent = (used_gb/total_gb)*100
        print(f"  Memory Usage: {usage_percent:.1f}%")

print(f"\nInitializing Qwen3-8B-Q5_K_M...")

try:
    # Build initialization parameters
    init_params = {
        'model_path': Config.MODEL_PATH,
        'n_ctx': Config.N_CTX,
        'n_gpu_layers': Config.N_GPU_LAYERS,
        'n_batch': Config.N_BATCH,
        'n_threads': Config.N_THREADS,
        'use_mmap': Config.USE_MMAP,
        'use_mlock': Config.USE_MLOCK,
        'verbose': Config.VERBOSE,
        'chat_format': "chatml",  # Qwen3 uses ChatML format
    }
    
    # Add GPU-specific parameters if using GPU
    if Config.USE_GPU and Config.N_GPU_LAYERS != 0:
        if Config.TENSOR_SPLIT:
            init_params['tensor_split'] = Config.TENSOR_SPLIT
        init_params['main_gpu'] = Config.MAIN_GPU
        print(f"  Offloading all layers to GPU...")
    
    llm = Llama(**init_params)
    print("Model loaded successfully!")
    
    # Post-load GPU memory check and stats
    if Config.USE_GPU:
        post_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
            capture_output=True, text=True, shell=True
        )
        
        if post_result.returncode == 0:
            used, free, util, temp = post_result.stdout.strip().split(', ')
            used_gb = float(used) / 1024
            free_gb = float(free) / 1024
            total_gb = used_gb + free_gb
            model_vram = used_gb - (total_gb - float(gpu_info.get('free_memory', '0')) / 1024) if gpu_info.get('free_memory') else used_gb
            
            print(f"\nPost-load VRAM Status:")
            print(f"  Model VRAM Usage: ~{model_vram:.2f} GB")
            print(f"  Total VRAM Used: {used_gb:.2f} GB / {total_gb:.2f} GB")
            print(f"  VRAM Available: {free_gb:.2f} GB")
            print(f"  GPU Utilization: {util}%")
            print(f"  GPU Temperature: {temp}°C")
            
            # Visual bar for memory after load
            usage_percent = (used_gb/total_gb)*100
            print(f"  Memory Usage: {usage_percent:.1f}%")
            
            # Performance assessment
            print(f"\nPerformance Assessment:")
            if free_gb > 4:
                print(f"  Excellent - {free_gb:.1f} GB free for generation")
            elif free_gb > 2:
                print(f"  Good - {free_gb:.1f} GB free for generation")
            elif free_gb > 1:
                print(f"  Adequate - {free_gb:.1f} GB free (may limit batch size)")
            else:
                print(f"  Warning: Low Memory - Only {free_gb:.1f} GB free (may cause slowdowns)")
                
except Exception as e:
    print(f"Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("- Check VRAM availability with nvidia-smi")
    print("- Try reducing N_CTX (current: {})".format(Config.N_CTX))
    print("- Try reducing N_BATCH (current: {})".format(Config.N_BATCH))
    print("- Set USE_GPU = False to force CPU mode")
    sys.exit(1)

# %% [5.0] Core Generation Functions
def generate(prompt: str, **kwargs) -> str:
    """Simple text generation"""
    params = {
        'max_tokens': Config.MAX_TOKENS,
        'temperature': Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'top_k': Config.TOP_K,
        'repeat_penalty': Config.REPEAT_PENALTY,
        'stream': False
    }
    params.update(kwargs)
    
    response = llm(prompt, **params)
    return response['choices'][0]['text']

def stream_generate(prompt: str, **kwargs) -> str:
    """Streaming text generation"""
    params = {
        'max_tokens': Config.MAX_TOKENS,
        'temperature': Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'top_k': Config.TOP_K,
        'repeat_penalty': Config.REPEAT_PENALTY,
        'stream': True
    }
    params.update(kwargs)
    
    output = ""
    for chunk in llm(prompt, **params):
        text = chunk['choices'][0]['text']
        print(text, end='', flush=True)
        output += text
    return output

def chat(messages: List[Dict[str, str]], stream: bool = True, **kwargs) -> str:
    """Chat completion with message history"""
    params = {
        'max_tokens': Config.MAX_TOKENS,
        'temperature': Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'stream': stream
    }
    params.update(kwargs)
    
    response = llm.create_chat_completion(messages=messages, **params)
    
    if stream:
        output = ""
        for chunk in response:
            if 'content' in chunk['choices'][0]['delta']:
                text = chunk['choices'][0]['delta']['content']
                print(text, end='', flush=True)
                output += text
        return output
    else:
        return response['choices'][0]['message']['content']

# %% [6.0] Quick Test
print("\n" + "="*60)
print("Testing Model")
print("="*60)

test_prompt = "Explain VRAM optimization in one sentence:"
print(f"Prompt: {test_prompt}\n")
print("Response: ", end='')
response = stream_generate(test_prompt, max_tokens=100)
print("\n")

# %% [6.1] Ready for Interactive Use
print("="*60)
print("Model Ready for Interactive Use!")
print("="*60)
print("\nQuick Commands:")
print("  chat_loop()    - Start interactive chat session")
print("  benchmark()    - Run performance test")
print("  model_info()   - Show detailed model/GPU info")
print("  gpu_monitor()  - Real-time GPU monitoring")
print("\n")

# %% [7.0] Interactive Chat Interface
def chat_loop():
    """Interactive chat session"""
    print("\n" + "="*60)
    print("Interactive Chat (type 'quit' to exit, 'clear' to reset)")
    print("="*60)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                messages = [messages[0]]
                print("Chat cleared.")
                continue
            elif not user_input:
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            print("\nAssistant: ", end='')
            response = chat(messages)
            print()
            
            messages.append({"role": "assistant", "content": response})
            
            # Keep last 10 exchanges
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]
                
        except KeyboardInterrupt:
            print("\n(Use 'quit' to exit)")
        except Exception as e:
            print(f"\nError: {e}")

# %% [8.0] Utility Functions
def benchmark():
    """Quick benchmark of generation speed with GPU monitoring"""
    import time
    
    prompt = "Write a haiku about AI:"
    
    print("\nBenchmarking...")
    
    # Pre-benchmark GPU state
    if Config.USE_GPU:
        pre_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, shell=True)
        if pre_result.returncode == 0:
            print(f"  Pre-benchmark GPU state: {pre_result.stdout.strip()}")
    
    start = time.time()
    tokens = 0
    
    for chunk in llm(prompt, max_tokens=50, stream=True):
        tokens += 1
    
    elapsed = time.time() - start
    
    # Post-benchmark GPU state
    if Config.USE_GPU:
        post_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, shell=True)
        if post_result.returncode == 0:
            print(f"  Post-benchmark GPU state: {post_result.stdout.strip()}")
    
    print(f"\nResults:")
    print(f"  Generated {tokens} tokens in {elapsed:.2f}s")
    print(f"  Speed: {tokens/elapsed:.1f} tokens/sec")
    
    # Estimate for different context sizes
    if tokens > 0:
        ms_per_token = (elapsed * 1000) / tokens
        print(f"  Latency: {ms_per_token:.1f} ms/token")
        print(f"  Estimated for 1K tokens: {(ms_per_token * 1000)/1000:.1f}s")
        print(f"  Estimated for 4K tokens: {(ms_per_token * 4000)/1000:.1f}s")

def model_info():
    """Display model and GPU information"""
    print("\nModel Information:")
    print(f"  Context length: {llm.n_ctx()}")
    print(f"  Vocabulary size: {llm.n_vocab()}")
    print(f"  Embedding dimensions: {llm.n_embd()}")
    print(f"  Model type: GGUF Q5_K_M quantization")
    
    if Config.USE_GPU:
        print("\nGPU Configuration:")
        print(f"  Layers offloaded: {Config.N_GPU_LAYERS if Config.N_GPU_LAYERS != -1 else 'All'}")
        print(f"  Main GPU index: {Config.MAIN_GPU}")
        
        # Current GPU status
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.free,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("\nCurrent GPU Status:")
            for i, line in enumerate(result.stdout.strip().split('\n')):
                parts = line.split(', ')
                if len(parts) >= 5:
                    name, util, used, free, temp = parts[:5]
                    print(f"  GPU {i} ({name}):")
                    print(f"    Utilization: {util}%")
                    print(f"    Memory: {used} MiB used, {free} MiB free")
                    print(f"    Temperature: {temp}°C")

def gpu_monitor():
    """Real-time GPU monitoring (updates every 2 seconds)"""
    if not Config.USE_GPU:
        print("GPU monitoring not available in CPU mode")
        return
    
    import time
    print("\nGPU Monitor (Ctrl+C to stop)")
    print("-" * 60)
    
    try:
        while True:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                # Clear previous lines
                print("\033[2K\033[1A" * 2, end='')
                
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        util, used, free, temp = parts[:4]
                        power = parts[4] if len(parts) > 4 else "N/A"
                        print(f"GPU {i}: Util {util}% | Mem {used}/{free} MiB | Temp {temp}°C | Power {power}W")
                
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

# %% [9.0] Main Entry Point
if __name__ == "__main__":
    # Show current GPU status at startup
    if Config.USE_GPU:
        print("Current GPU Status:")
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw', 
             '--format=csv,noheader,nounits'], 
            capture_output=True, text=True, shell=True
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            if len(values) >= 6:
                name, mem_used, mem_free, util, temp, power = values[:6]
                used_gb = float(mem_used) / 1024
                free_gb = float(mem_free) / 1024
                print(f"  {name}")
                print(f"  Memory: {used_gb:.1f}GB used, {free_gb:.1f}GB free")
                print(f"  Status: {util}% util, {temp}°C, {power}W")
    
    print("\n" + "="*60)
    print("Entering Interactive Chat Mode")
    print("Commands: 'quit' to exit, 'clear' to reset conversation")
    print("="*60)
    
    # Automatically start chat loop
    chat_loop()
