"""
Setup script for College AI Assistant.
Automates initial setup and configuration.
"""

import os
import shutil
import sys
from pathlib import Path


def setup_project():
    """Run complete project setup."""
    print("=" * 60)
    print("🎓 College AI Assistant - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = ["data", "logs", "exports", ".streamlit"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ {directory}/")
    
    # Copy .env if not exists
    print("\n⚙️ Configuration setup...")
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            print("  ✓ Created .env from .env.example")
            print("  ⚠️  Please update .env with your Groq API key")
        else:
            with open(".env", "w") as f:
                f.write('GROQ_API_KEY=your_key_here\n')
            print("  ✓ Created .env (update with your API key)")
    else:
        print("  ✓ .env already exists")
    
    # Check/create Streamlit config
    print("\n🎨 Streamlit configuration...")
    streamlit_config_path = Path(".streamlit/config.toml")
    if not streamlit_config_path.exists():
        streamlit_config_path.parent.mkdir(exist_ok=True)
        with open(streamlit_config_path, "w") as f:
            f.write("""[theme]
primaryColor = "#2196F3"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#000000"
font = "sans serif"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[server]
maxUploadSize = 200
enableXsrfProtection = true
""")
        print("  ✓ Created Streamlit config")
    else:
        print("  ✓ Streamlit config exists")
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  ✓ Packages installed successfully")
        else:
            print("  ⚠️  Package installation had warnings (see above)")
    except Exception as e:
        print(f"  ❌ Failed to install packages: {e}")
        print("  Run manually: pip install -r requirements.txt")
    
    # Download transformer model
    print("\n🤖 Downloading emotion detection model...")
    try:
        from transformers import pipeline
        print("  ⏳ Downloading transformer model (this may take a minute)...")
        pipeline("sentiment-analysis")
        print("  ✓ Model downloaded successfully")
    except Exception as e:
        print(f"  ⚠️  Could not download model: {e}")
        print("  This will be downloaded on first run")
    
    # Train intent model
    print("\n🧠 Training intent classifier...")
    try:
        from intent_model import IntentClassifier
        classifier = IntentClassifier()
        if not classifier.is_trained:
            result = classifier.train()
            if result.get("success"):
                print(f"  ✓ Model trained ({result.get('intents_count')} intents)")
            else:
                print(f"  ⚠️  Training had issues: {result.get('error')}")
        else:
            print("  ✓ Model already trained")
    except Exception as e:
        print(f"  ⚠️  Could not train model: {e}")
        print("  Model will be trained on first run")
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update .env with your Groq API key (from https://console.groq.com)")
    print("2. Edit data/college_data.json with your institution info")
    print("3. Run: streamlit run app.py")
    print("\nFor CLI mode, run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        setup_project()
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)
