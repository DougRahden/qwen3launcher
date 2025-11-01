"""
Qwen3-8B-Q5_K_M GGUF Model Launcher
NVIDIA GPU Accelerated Version
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
print("Qwen3-8B GGUF Model Launcher - GPU Edition")
print("="*60)

# System info
print(f"\nSystem Information:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Platform: {platform.platform()}")
print(f"  Processor: {platform.processor()}")

# Check llama-cpp-python
try:
    from llama_cpp import Llama
    import llama_cpp
    print(f"  llama-cpp-python: ✓ {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'installed'}")
except ImportError:
    print("  llama-cpp-python: ✗ NOT FOUND")
    sys.exit(1)

# %% [1.1] GPU/CUDA Detection
print(f"\n" + "="*60)
print("GPU/CUDA Diagnostics")
print("="*60)

def get_nvidia_info():
    """Get comprehensive NVIDIA GPU and CUDA information"""
    gpu_info = {}
    
    try:
        # Get detailed GPU information using nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get additional details
            gpu_info['cuda_cores'] = pynvml.nvmlDeviceGetNumGpuCores(handle) if hasattr(pynvml, 'nvmlDeviceGetNumGpuCores') else 'N/A'
            gpu_info['pcie_gen'] = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle) if hasattr(pynvml, 'nvmlDeviceGetMaxPcieLinkGeneration') else 'N/A'
            gpu_info['pcie_width'] = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle) if hasattr(pynvml, 'nvmlDeviceGetMaxPcieLinkWidth') else 'N/A'
            
            pynvml.nvmlShutdown()
        except:
            pass  # pynvml not available, use nvidia-smi only
        
        # Get comprehensive GPU info via nvidia-smi
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
                    else:
                        print(f"  {label}: {value}")
                
                # Store for later use
                for label, value in zip(labels, values):
                    gpu_info[label.lower().replace(' ', '_')] = value
        
        # Get memory bus width and bandwidth (these require specific queries)
        bus_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, shell=True
        )
        
        if bus_result.returncode == 0:
            # Try to get additional architecture info
            arch_result = subprocess.run(
                ['nvidia-smi', '-q', '-d', 'MEMORY'],
                capture_output=True, text=True, shell=True
            )
            
            if arch_result.returncode == 0:
                output_lines = arch_result.stdout.split('\n')
                for line in output_lines:
                    if 'Bus Width' in line:
                        bus_width = line.split(':')[1].strip()
                        print(f"\nMemory Architecture:")
                        print(f"  Memory Bus Width: {bus_width}")
                        gpu_info['memory_bus_width'] = bus_width
                        
                        # Calculate theoretical bandwidth if we have clock speed
                        if 'current_mem_clock' in gpu_info and gpu_info['current_mem_clock'].isdigit():
                            mem_clock_mhz = float(gpu_info['current_mem_clock'])
                            bus_width_bits = int(bus_width.replace('bits', '').strip())
                            # Bandwidth (GB/s) = (memory_clock_MHz × bus_width_bits × 2) / 8 / 1000
                            bandwidth_gbps = (mem_clock_mhz * bus_width_bits * 2) / 8 / 1000
                            print(f"  Theoretical Bandwidth: {bandwidth_gbps:.1f} GB/s")
        
        # Performance state
        pstate_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pstate', '--format=csv,noheader'],
            capture_output=True, text=True, shell=True
        )
        
        if pstate_result.returncode == 0:
            print(f"\nPerformance State: {pstate_result.stdout.strip()}")
            
    except Exception as e:
        print(f"\n⚠ Warning: Could not get complete GPU info")
        print(f"  Error: {e}")
        print("  Ensure NVIDIA drivers are installed and nvidia-smi is accessible")
        gpu_info['error'] = str(e)
    
    # CUDA compilation check
    try:
        import llama_cpp
        if hasattr(llama_cpp, 'llama_backend_init'):
            print(f"\n✓ llama.cpp CUDA backend available")
        else:
            print(f"\n⚠ llama.cpp may not have CUDA support")
    except:
        pass
    
    return gpu_info

gpu_info = get_nvidia_info()

# Memory requirement estimation
print(f"\n" + "="*60)
print("Model Memory Requirements (Estimated)")
print("="*60)
print(f"  Model: Qwen3-8B-Q5_K_M (5.85 GB file)")
print(f"  Context: {8192} tokens")
print(f"  Estimated VRAM needed: ~7-8 GB")
print(f"  Recommended free VRAM: >8 GB")

if gpu_info.get('free_memory'):
    free_gb = float(gpu_info['free_memory']) / 1024
    if free_gb < 8:
        print(f"\n⚠ Warning: Only {free_gb:.2f} GB VRAM free, model may not fit entirely on GPU")

if not gpu_info.get('gpu_model') and not gpu_info.get('error'):
    print("\n⚠ No NVIDIA GPU detected - model will run slowly!")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

# %% [2.0] Configuration
class Config:
    # Model location
    MODEL_PATH = r"E:\ai\models\qwen3-8b-gguf\qwen3-8b-q5_k_m.gguf"
    
    # GPU/CPU Toggle
    USE_GPU = True           # Set to False to force CPU mode
    
    # Model settings
    N_CTX = 8192            # Context window
    N_GPU_LAYERS = -1       # -1 = all layers on GPU (when USE_GPU=True)
    N_BATCH = 512           # Batch size
    N_THREADS = 16          # CPU threads (only used if forced to CPU)
    
    # Generation defaults
    TEMPERATURE = 0.7
    TOP_P = 0.95
    TOP_K = 40
    REPEAT_PENALTY = 1.1
    MAX_TOKENS = 2048
    
    # Memory
    USE_MMAP = True         # Memory-mapped files
    USE_MLOCK = False       # Lock in RAM
    VERBOSE = False         # Show llama.cpp output
    
    # GPU specific settings
    TENSOR_SPLIT = None     # For multi-GPU systems (e.g., [0.5, 0.5] for 2 GPUs)
    MAIN_GPU = 0           # Primary GPU index for scratch buffer

# Apply GPU/CPU toggle
if not Config.USE_GPU:
    Config.N_GPU_LAYERS = 0
    print("\n⚠ GPU disabled by config - using CPU mode")

print(f"\nModel Configuration:")
print(f"  Model: {Path(Config.MODEL_PATH).name}")
print(f"  Context: {Config.N_CTX} tokens")
print(f"  Batch Size: {Config.N_BATCH}")
print(f"  Mode: {'GPU' if Config.USE_GPU else 'CPU'}")
if Config.USE_GPU:
    print(f"  GPU Layers: {'All' if Config.N_GPU_LAYERS == -1 else Config.N_GPU_LAYERS}")
    print(f"  Main GPU: {Config.MAIN_GPU}")

# %% [3.0] Model Validation
def validate_model():
    """Check if model file exists"""
    model_path = Path(Config.MODEL_PATH)
    
    if not model_path.exists():
        print(f"\n✗ Model not found at: {Config.MODEL_PATH}")
        print("\nTo fix:")
        print("1. Download from: https://huggingface.co/Qwen/Qwen3-8B-GGUF")
        print("2. Get file: Qwen3-8B-Q5_K_M.gguf (5.85 GB)")
        print(f"3. Save to: {Config.MODEL_PATH}")
        return False
    
    size_gb = model_path.stat().st_size / (1024**3)
    print(f"\n✓ Model found ({size_gb:.2f} GB)")
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
        bar_width = 40
        used_bars = int((used_gb / total_gb) * bar_width)
        free_bars = bar_width - used_bars
        print(f"  [{'█' * used_bars}{'░' * free_bars}] {(used_gb/total_gb)*100:.1f}% used")

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
    print("✓ Model loaded successfully!")
    
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
            bar_width = 40
            used_bars = int((used_gb / total_gb) * bar_width)
            free_bars = bar_width - used_bars
            print(f"  [{'█' * used_bars}{'░' * free_bars}] {(used_gb/total_gb)*100:.1f}% used")
            
            # Performance assessment
            print(f"\nPerformance Assessment:")
            if free_gb > 4:
                print(f"  ✓ Excellent - {free_gb:.1f} GB free for generation")
            elif free_gb > 2:
                print(f"  ✓ Good - {free_gb:.1f} GB free for generation")
            elif free_gb > 1:
                print(f"  ⚠ Adequate - {free_gb:.1f} GB free (may limit batch size)")
            else:
                print(f"  ⚠ Low Memory - Only {free_gb:.1f} GB free (may cause slowdowns)")
                
except Exception as e:
    print(f"✗ Failed to load model: {e}")
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
    """Benchmark generation speed with GPU monitoring"""
    import time
    
    prompt = "Write a haiku about AI:"
    
    print("\nBenchmarking...")
    
    # Pre-benchmark GPU state
    if Config.USE_GPU:
        pre_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], 
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
        post_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], 
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
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("\nCurrent GPU Status:")
            for i, line in enumerate(result.stdout.strip().split('\n')):
                parts = line.split(', ')
                if len(parts) >= 5:
                    name, util, used, free, temp = parts[:5]
                    print(f"  GPU {i} ({name}):")
                    print(f"    Utilization: {util}")
                    print(f"    Memory: {used} used, {free} free")
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
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                # Clear previous lines
                print("\033[2K\033[1A" * 2, end='')
                
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        util, used, free, temp = parts[:4]
                        power = parts[4] if len(parts) > 4 else "N/A"
                        print(f"GPU {i}: Util {util} | Mem {used}/{free} | Temp {temp}°C | Power {power}")
                
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
