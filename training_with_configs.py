import subprocess
import os
from datetime import datetime

# Function to launch training on a specific GPU with a given config file
def run_training_on_gpus(prompts):
    
    processes = []
    
    try:
        for _, prompt in enumerate(prompts):
            # Generate config file for this GPU setting
            config_file_dir = 'exp/tabr/sait/' + prompt['name'] + '.toml'
            which_gpu = prompt['which_gpu']
        
            command = ["python", "bin/go.py", config_file_dir, "--force"]
            os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
            output_file_dir = 'exp/tabr/sait/' + prompt['name'] + ".txt"
            
            out = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append((out, output_file_dir, which_gpu, config_file_dir))
            
        # Now collect the results
        for out, output_file_dir, which_gpu, config_file_dir in processes:
            try:
                with open(output_file_dir, 'w') as f:
                    # Run the command and capture the output
                    stdout, stderr = out.communicate()

                    # If the script ran successfully, write the output to the file
                    if out.returncode == 0:
                        f.write(stdout)
                    else:
                        f.write(f"{which_gpu}// Error in {config_file_dir}:\n{stderr}")
                        return f"{which_gpu}// Error in {config_file_dir}:\n{stderr}"

            except Exception as e:
                return f"Exception while running {config_file_dir}: {str(e)}"
    
    except KeyboardInterrupt:
        # Terminate all subprocesses when a KeyboardInterrupt is caught (e.g., Ctrl+C)
        print("\nAborting... Terminating all subprocesses.")
        for out, _, _, _ in processes:
            if out.poll() is None:  # Check if the process is still running
                out.terminate()  # Gracefully terminate the process
                try:
                    out.wait(timeout=5)  # Wait for process to terminate
                except subprocess.TimeoutExpired:
                    out.kill()  # Force kill if not terminated after timeout
        print("All subprocesses terminated.")
        
    return "All training processes completed successfully."

if __name__ == "__main__":
    # Example settings for multiple GPUs
        
    prompts = [
        {"name": "2-config01-plr-lite-tuning", "which_gpu": "0"},
        {"name": "2-config03-plr-lite-tuning", "which_gpu": "1"},
    ]
    
    
    # Launch the training on the specified GPU with the generated config file
    run_training_on_gpus(prompts)