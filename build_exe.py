#!/usr/bin/env python
"""
Build script for creating a single executable of the cursor_clicker application.

This script uses Python's built-in venv module to create a virtual environment,
installs the package and its dependencies, and then uses PyInstaller to create
a single executable file.

Requirements:
- uv tool installed and available in PATH
- PyInstaller
"""

import os
import sys
import shutil
import subprocess
import platform
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, env=None):
    """Run a command and return its output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, 
        cwd=cwd, 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    return result.stdout

def build_executable(mode="run", output_name=None):
    """Build the executable for the specified mode."""
    # Get absolute paths
    root_dir = os.path.abspath(os.path.dirname(__file__))
    build_dir = os.path.join(root_dir, "build")
    dist_dir = os.path.join(root_dir, "dist")
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)
    
    # Determine the entry point and script path
    if mode == "run":
        entry_point = "cursor_clicker.cli:main"
        default_name = "cursor_clicker"
        script_path = os.path.join(root_dir, "src", "cursor_clicker", "cli.py")
    elif mode == "monitor":
        entry_point = "cursor_clicker.continuous_monitor:main"
        default_name = "continuous_monitor"
        script_path = os.path.join(root_dir, "src", "cursor_clicker", "continuous_monitor.py")
    else:
        print(f"Invalid mode: {mode}")
        sys.exit(1)
    
    # Verify the script path exists
    if not os.path.exists(script_path):
        print(f"Error: Script path '{script_path}' does not exist.")
        sys.exit(1)
    
    # Determine output name
    if output_name is None:
        output_name = default_name
    
    # Add .exe extension on Windows
    if platform.system() == "Windows" and not output_name.endswith(".exe"):
        output_name += ".exe"
    
    # Install the package and PyInstaller
    print("Installing dependencies using uv...")
    run_command(["uv", "pip", "install", "-e", "."], cwd=root_dir)
    run_command(["uv", "pip", "install", "pyinstaller"], cwd=root_dir)
    
    # Create the spec file
    spec_file = os.path.join(build_dir, f"{output_name}.spec")
    
    print(f"Creating PyInstaller spec file: {spec_file}")
    with open(spec_file, "w") as f:
        f.write(f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    [r'{script_path.replace("\\", "\\\\")}'],
    pathex=[r'{root_dir.replace("\\", "\\\\")}'],
    binaries=[],
    datas=[],
    hiddenimports=["PIL._tkinter_finder"],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{output_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
""")
    
    # Build the executable
    print(f"Building executable for {mode} mode as {output_name}...")
    run_command(["pyinstaller", "--clean", spec_file], cwd=root_dir)
    
    # Verify the executable was created
    executable_path = os.path.join(dist_dir, output_name)
    if os.path.exists(executable_path):
        print(f"Successfully built executable: {executable_path}")
    else:
        print(f"Failed to build executable: {executable_path}")
        sys.exit(1)
    
    return executable_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build cursor_clicker executable")
    parser.add_argument(
        "--mode", 
        choices=["run", "monitor"], 
        default="run",
        help="Which mode to build (run: full application, monitor: continuous monitor only)"
    )
    parser.add_argument(
        "--output", 
        help="Name of the output executable (without extension)"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Build the executable
    executable_path = build_executable(
        mode=args.mode,
        output_name=args.output
    )
    
    print(f"\nBuild completed successfully!")
    print(f"Executable: {executable_path}")
    print(f"Run with: {executable_path}")

if __name__ == "__main__":
    main() 