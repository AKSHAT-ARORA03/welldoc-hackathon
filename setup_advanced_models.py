import subprocess
import sys

def install_advanced_model_dependencies():
    """Install required packages for advanced models"""
    print("Installing dependencies for advanced models...")
    
    requirements = [
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "shap>=0.41.0",
        "streamlit>=1.15.0",
        "plotly>=5.10.0",
        "joblib>=1.1.0"
    ]
    
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install each package
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Error installing {package}, trying without version constraint...")
            package_name = package.split(">=")[0]
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    
    print("Installation completed successfully!")

if __name__ == "__main__":
    install_advanced_model_dependencies()
