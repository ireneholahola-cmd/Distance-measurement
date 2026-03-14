import subprocess

def run_detection_script():
    script_path = "detect.py"  # 这里替换成你的原始脚本的路径

    # 使用subprocess调用Python脚本
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running script:", e)

if __name__ == "__main__":
    run_detection_script()
