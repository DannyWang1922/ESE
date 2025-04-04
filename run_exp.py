import subprocess

cmd_list = [
    'python train_self.py --model_name_or_path BAAI/bge-base-en-v1.5  --apply_ese 0 --save_dir result_beg',
    'python train_self.py --model_name_or_path BAAI/bge-base-en-v1.5  --apply_ese 1 --save_dir result_beg_ese',
]

for cmd in cmd_list:
    print(f"\nRunning cmd: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)


        
