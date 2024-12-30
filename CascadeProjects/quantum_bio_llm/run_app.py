import os
import subprocess
import sys

def check_node_installed():
    try:
        subprocess.run([r"C:\Program Files\nodejs\node.exe", "-v"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_packages():
    try:
        subprocess.run([r"C:\Program Files\nodejs\npm.cmd", "install"], check=True, cwd="c:/Users/Windows-500GB/CascadeProjects/quantum_bio_llm")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def start_react_app():
    try:
        subprocess.Popen([r"C:\Program Files\nodejs\node.exe", r"C:\Program Files\nodejs\npm.cmd", "start"], cwd="c:/Users/Windows-500GB/CascadeProjects/quantum_bio_llm")
    except Exception as e:
        print(f"Error starting React app: {e}")
        sys.exit(1)

def run_python_script(script_name):
    try:
        subprocess.run([sys.executable, f"c:/Users/Windows-500GB/CascadeProjects/quantum_bio_llm/examples/{script_name}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Python script {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not check_node_installed():
        print("Node.js is not installed. Please install Node.js to continue.")
        sys.exit(1)

    install_packages()
    start_react_app()
    run_python_script("rag_memory.py")  # Change this to the script you want to run
