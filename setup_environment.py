import subprocess
import sys

def install_dependencies():
    """Install or upgrade the required dependencies."""
    print("Installing required dependencies...")
    
    # List of required packages with version constraints
    requirements = [
        "numpy>=1.24.0",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "shap",
        "numba",
        "joblib",
        "ipywidgets",  # Required for SHAP visualizations in notebooks
    ]
    
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install or upgrade each package
    for package in requirements:
        print(f"Installing/upgrading {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            
    print("\nDependency installation completed.")
    print("You may need to restart your Python environment for changes to take effect.")
    
if __name__ == "__main__":
    install_dependencies()
