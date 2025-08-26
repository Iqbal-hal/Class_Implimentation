#!/usr/bin/env python3
"""
Usage Example: Block Diagram Generator Demo

This script demonstrates the key features of the block diagram generator
for the algorithmic trading system.

Run this to see all capabilities in action.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"{'='*60}")
    print(f"🔧 Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                # Print only last few lines to avoid clutter
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"   {line}")
        else:
            print("❌ Error!")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}...")
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Demonstrate block diagram generator features."""
    print("🎯 Block Diagram Generator - Feature Demonstration")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("Enhanced_stock_trading_V8.py").exists():
        print("❌ Please run this from the project root directory")
        sys.exit(1)
    
    # 1. Show help
    run_command("python diagram_cli.py --help", 
                "Display CLI Help and Options")
    
    # 2. Generate system architecture only
    run_command("python diagram_cli.py --system-only --format png", 
                "Generate System Architecture Diagrams (PNG)")
    
    # 3. Generate trading pipeline
    run_command("python diagram_cli.py --pipeline-only", 
                "Generate Trading Pipeline Diagram")
    
    # 4. Generate class method diagram
    run_command("python diagram_cli.py --class-only FilteringAndBacktesting", 
                "Generate Class Method Diagram")
    
    # 5. Generate interactive HTML diagrams
    run_command("python diagram_cli.py --system-only --format html --layout circular", 
                "Generate Interactive HTML Diagrams")
    
    # 6. Create comprehensive suite
    run_command("python diagram_cli.py --create-index", 
                "Create HTML Index with All Diagrams")
    
    # 7. Show final results
    print(f"\n{'='*60}")
    print("📊 FINAL RESULTS")
    print(f"{'='*60}")
    
    output_dir = Path("diagram_output")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        print(f"✅ Generated {len(files)} files in diagram_output/:")
        
        for file in sorted(files):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    size_str = f"{size_mb:.1f}MB"
                else:
                    size_str = f"{file.stat().st_size / 1024:.1f}KB"
                print(f"   📄 {file.name} ({size_str})")
        
        index_file = output_dir / "index.html"
        if index_file.exists():
            print(f"\n🌐 Open the HTML index to view all diagrams:")
            print(f"   file://{index_file.absolute()}")
    else:
        print("❌ No output directory found")
    
    print(f"\n{'='*60}")
    print("🎉 DEMONSTRATION COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Open diagram_output/index.html in your browser")
    print("2. Explore the interactive HTML diagrams")
    print("3. Review the comprehensive PDF report")
    print("4. Use the Mermaid diagram in your GitHub documentation")

if __name__ == "__main__":
    main()