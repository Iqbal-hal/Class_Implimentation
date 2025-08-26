#!/usr/bin/env python3
"""
Master Block Diagram Generator CLI for Algorithmic Trading System

This is the main command-line interface that provides access to all
block diagram generation features for the trading system.

Features:
- Complete system architecture diagrams
- Trading pipeline visualization
- Class method flow diagrams
- Data flow analysis
- Multiple output formats (PNG, SVG, HTML, PDF, Mermaid)
- Interactive and static diagrams
- Comprehensive PDF reports

Usage Examples:
    python diagram_cli.py --all                    # Generate all diagrams
    python diagram_cli.py --system-only           # System architecture only
    python diagram_cli.py --pipeline-only         # Trading pipeline only
    python diagram_cli.py --format png html       # Specific formats
    python diagram_cli.py --output ./my_diagrams  # Custom output directory

Author: AI Assistant
Created: 2024
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Import our diagram generators
from block_diagram_generator import DiagramGenerator
from enhanced_diagram_features import EnhancedDiagramGenerator


class MasterDiagramCLI:
    """Master CLI for all diagram generation features."""
    
    def __init__(self, project_root: str = '.', output_dir: str = None):
        self.project_root = Path(project_root)
        self.enhanced_generator = EnhancedDiagramGenerator(str(self.project_root))
        
        if output_dir:
            self.enhanced_generator.output_dir = Path(output_dir)
            self.enhanced_generator.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
    
    def generate_system_diagrams(self, layouts: List[str], formats: List[str]) -> Dict[str, str]:
        """Generate system architecture diagrams."""
        print("🏗️  Generating system architecture diagrams...")
        return self.enhanced_generator.generate_all_diagrams(layouts=layouts, formats=formats)
    
    def generate_pipeline_diagrams(self) -> Dict[str, str]:
        """Generate trading pipeline diagrams."""
        print("📈 Generating trading pipeline diagrams...")
        results = {}
        results['trading_pipeline'] = self.enhanced_generator.generate_trading_pipeline_diagram()
        return results
    
    def generate_enhanced_diagrams(self) -> Dict[str, str]:
        """Generate enhanced analysis diagrams."""
        print("🔬 Generating enhanced analysis diagrams...")
        return self.enhanced_generator.generate_enhanced_analysis()
    
    def generate_class_diagrams(self, class_name: str = "FilteringAndBacktesting") -> Dict[str, str]:
        """Generate class-specific diagrams."""
        print(f"🎯 Generating class diagrams for {class_name}...")
        results = {}
        results['class_methods'] = self.enhanced_generator.generate_class_method_diagram(class_name)
        return results
    
    def generate_data_flow_diagrams(self) -> Dict[str, str]:
        """Generate data flow diagrams."""
        print("📊 Generating data flow diagrams...")
        results = {}
        results['data_flow'] = self.enhanced_generator.generate_data_flow_diagram()
        return results
    
    def generate_pdf_reports(self) -> Dict[str, str]:
        """Generate PDF reports."""
        print("📄 Generating PDF reports...")
        results = {}
        results['pdf_report'] = self.enhanced_generator.generate_pdf_report()
        return results
    
    def generate_all_diagrams(self, layouts: List[str], formats: List[str]) -> Dict[str, Any]:
        """Generate all types of diagrams."""
        print("🚀 Generating comprehensive diagram suite...")
        
        all_results = {}
        
        # System architecture diagrams
        if 'png' in formats or 'svg' in formats or 'html' in formats or 'mermaid' in formats:
            system_results = self.generate_system_diagrams(layouts, formats)
            all_results.update(system_results)
        
        # Enhanced analysis
        enhanced_results = self.generate_enhanced_diagrams()
        all_results.update(enhanced_results)
        
        return all_results
    
    def create_index_html(self) -> str:
        """Create an HTML index page linking to all generated diagrams."""
        print("🌐 Creating HTML index page...")
        
        # Collect all generated files
        diagram_files = []
        for file_path in self.enhanced_generator.output_dir.glob("*"):
            if file_path.is_file():
                file_info = {
                    'name': file_path.name,
                    'path': file_path.name,  # Relative path for HTML
                    'size': file_path.stat().st_size,
                    'type': file_path.suffix.lower()
                }
                diagram_files.append(file_info)
        
        # Sort files by type and name
        diagram_files.sort(key=lambda x: (x['type'], x['name']))
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithmic Trading System - Block Diagrams</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .file-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .file-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #f8f9fa;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .file-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .file-name {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .file-type {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .type-png {{ background: #e74c3c; color: white; }}
        .type-svg {{ background: #9b59b6; color: white; }}
        .type-html {{ background: #e67e22; color: white; }}
        .type-pdf {{ background: #c0392b; color: white; }}
        .type-md {{ background: #27ae60; color: white; }}
        .file-size {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 5px;
        }}
        .file-link {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
        }}
        .file-link:hover {{
            background: #2980b9;
        }}
        .overview {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 5px;
        }}
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ Algorithmic Trading System</h1>
        <p class="subtitle">Block Diagram Analysis & Visualization Suite</p>
        
        <div class="overview">
            <h3>📊 Overview</h3>
            <p>This comprehensive visualization suite provides multiple perspectives on the algorithmic trading system architecture, 
               including system dependencies, trading pipeline flow, class relationships, and data transformations.</p>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{len([f for f in diagram_files if f['type'] == '.png'])}</div>
                    <div class="stat-label">PNG Diagrams</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len([f for f in diagram_files if f['type'] == '.html'])}</div>
                    <div class="stat-label">Interactive HTML</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len([f for f in diagram_files if f['type'] == '.svg'])}</div>
                    <div class="stat-label">SVG Diagrams</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len([f for f in diagram_files if f['type'] == '.pdf'])}</div>
                    <div class="stat-label">PDF Reports</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Generated Diagrams</h2>
            <div class="file-grid">
        """
        
        # Add file cards
        for file_info in diagram_files:
            file_size_kb = file_info['size'] / 1024
            size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb/1024:.1f} MB"
            
            type_class = f"type-{file_info['type'][1:]}" if file_info['type'] else "type-unknown"
            
            description = self._get_file_description(file_info['name'])
            
            html_content += f"""
                <div class="file-card">
                    <div class="file-name">{file_info['name']}</div>
                    <div class="file-type {type_class}">{file_info['type'][1:].upper() if file_info['type'] else 'FILE'}</div>
                    <div class="file-size">{size_str}</div>
                    <p style="font-size: 14px; color: #5d6d7e; margin: 10px 0;">{description}</p>
                    <a href="{file_info['path']}" class="file-link" target="_blank">Open Diagram</a>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Diagram Types Explained</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div style="padding: 15px; border-left: 4px solid #e74c3c;">
                    <h4>🏗️ System Architecture</h4>
                    <p>Shows module dependencies and relationships across the entire trading system.</p>
                </div>
                <div style="padding: 15px; border-left: 4px solid #3498db;">
                    <h4>📈 Trading Pipeline</h4>
                    <p>Visualizes the data flow from input through processing to output generation.</p>
                </div>
                <div style="padding: 15px; border-left: 4px solid #27ae60;">
                    <h4>🎯 Class Methods</h4>
                    <p>Details the methods and flow within key trading classes like FilteringAndBacktesting.</p>
                </div>
                <div style="padding: 15px; border-left: 4px solid #f39c12;">
                    <h4>📊 Data Flow</h4>
                    <p>Interactive visualization of how data transforms through different system stages.</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📖 Usage Instructions</h2>
            <ol>
                <li><strong>Static Diagrams (PNG/SVG):</strong> Best for documentation, presentations, and printed materials</li>
                <li><strong>Interactive Diagrams (HTML):</strong> Open in web browser for interactive exploration with zoom and hover details</li>
                <li><strong>PDF Reports:</strong> Comprehensive analysis with multiple diagrams combined for formal documentation</li>
                <li><strong>Mermaid Diagrams:</strong> Use in GitHub README files and Markdown documentation</li>
            </ol>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            <p>Generated by Block Diagram Generator for Algorithmic Trading System</p>
            <p>Created on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save index HTML
        index_file = self.enhanced_generator.output_dir / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML index created: {index_file}")
        return str(index_file)
    
    def _get_file_description(self, filename: str) -> str:
        """Get description for a diagram file."""
        descriptions = {
            'system_diagram_hierarchical.png': 'Hierarchical view of system module dependencies',
            'system_diagram_circular.png': 'Circular layout showing module relationships',
            'interactive_diagram_hierarchical.html': 'Interactive hierarchical system diagram',
            'interactive_diagram_circular.html': 'Interactive circular system diagram',
            'trading_pipeline_diagram.png': 'Trading system data flow pipeline visualization',
            'class_methods_FilteringAndBacktesting.png': 'Method flow within FilteringAndBacktesting class',
            'data_flow_diagram.html': 'Interactive data transformation flow analysis',
            'mermaid_diagram.md': 'GitHub-compatible Mermaid diagram format',
            'comprehensive_analysis_report.pdf': 'Complete PDF report with all diagrams',
            'analysis_report.md': 'Markdown summary of system analysis'
        }
        
        return descriptions.get(filename, 'System diagram file')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Master Block Diagram Generator for Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagram_cli.py --all                           # Generate everything
  python diagram_cli.py --system-only                   # System architecture only  
  python diagram_cli.py --pipeline-only                 # Trading pipeline only
  python diagram_cli.py --enhanced-only                 # Enhanced analysis only
  python diagram_cli.py --class-only FilteringAndBacktesting  # Class diagrams only
  python diagram_cli.py --format png html               # Specific formats
  python diagram_cli.py --layout hierarchical circular  # Multiple layouts
  python diagram_cli.py --output ./my_diagrams          # Custom output directory
  python diagram_cli.py --create-index                  # Create HTML index page
        """
    )
    
    # Main options
    parser.add_argument('--project-root', '-p', default='.', 
                       help='Root directory of the project (default: current directory)')
    parser.add_argument('--output', '-o', help='Output directory for diagrams')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Generation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--all', action='store_true', 
                           help='Generate all types of diagrams (default)')
    mode_group.add_argument('--system-only', action='store_true',
                           help='Generate only system architecture diagrams')
    mode_group.add_argument('--pipeline-only', action='store_true',
                           help='Generate only trading pipeline diagrams')
    mode_group.add_argument('--enhanced-only', action='store_true',
                           help='Generate only enhanced analysis diagrams')
    mode_group.add_argument('--class-only', metavar='CLASS_NAME',
                           help='Generate only class method diagrams for specified class')
    mode_group.add_argument('--data-flow-only', action='store_true',
                           help='Generate only data flow diagrams')
    mode_group.add_argument('--pdf-only', action='store_true',
                           help='Generate only PDF reports')
    
    # Format and layout options
    parser.add_argument('--format', '-f', 
                       choices=['png', 'svg', 'html', 'pdf', 'mermaid'],
                       nargs='+', default=['png', 'html', 'mermaid'],
                       help='Output formats (default: png html mermaid)')
    parser.add_argument('--layout', '-l',
                       choices=['hierarchical', 'circular', 'force-directed'],
                       nargs='+', default=['hierarchical'],
                       help='Layout algorithms (default: hierarchical)')
    
    # Additional options
    parser.add_argument('--create-index', action='store_true',
                       help='Create HTML index page for all generated diagrams')
    parser.add_argument('--export-config', metavar='FILE',
                       help='Export generation configuration to JSON file')
    parser.add_argument('--import-config', metavar='FILE',
                       help='Import generation configuration from JSON file')
    
    args = parser.parse_args()
    
    # Handle config import
    if args.import_config:
        try:
            with open(args.import_config, 'r') as f:
                config = json.load(f)
            # Override args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            print(f"📥 Configuration imported from {args.import_config}")
        except Exception as e:
            print(f"❌ Error importing config: {e}")
            return 1
    
    # Initialize CLI
    cli = MasterDiagramCLI(args.project_root, args.output)
    
    print(f"🎯 Project root: {cli.project_root}")
    print(f"📁 Output directory: {cli.enhanced_generator.output_dir}")
    print(f"🎨 Layouts: {', '.join(args.layout)}")
    print(f"📄 Formats: {', '.join(args.format)}")
    
    # Determine what to generate
    if args.system_only:
        results = cli.generate_system_diagrams(args.layout, args.format)
    elif args.pipeline_only:
        results = cli.generate_pipeline_diagrams()
    elif args.enhanced_only:
        results = cli.generate_enhanced_diagrams()
    elif args.class_only:
        results = cli.generate_class_diagrams(args.class_only)
    elif args.data_flow_only:
        results = cli.generate_data_flow_diagrams()
    elif args.pdf_only:
        results = cli.generate_pdf_reports()
    else:  # Default: generate all
        results = cli.generate_all_diagrams(args.layout, args.format)
    
    # Create HTML index if requested or if generating all
    if args.create_index or args.all or not any([args.system_only, args.pipeline_only, 
                                                args.enhanced_only, args.class_only,
                                                args.data_flow_only, args.pdf_only]):
        index_file = cli.create_index_html()
        results['html_index'] = index_file
    
    # Export config if requested
    if args.export_config:
        config_data = {
            'project_root': args.project_root,
            'output': args.output,
            'format': args.format,
            'layout': args.layout,
            'verbose': args.verbose
        }
        try:
            with open(args.export_config, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"📤 Configuration exported to {args.export_config}")
        except Exception as e:
            print(f"❌ Error exporting config: {e}")
    
    # Print results
    print("\n🎉 Diagram generation completed successfully!")
    print("\n📊 Generated files:")
    for key, file_path in results.items():
        if file_path:
            print(f"  - {key}: {file_path}")
    
    if 'html_index' in results:
        print(f"\n🌐 Open the HTML index to view all diagrams: {results['html_index']}")
    
    return 0


if __name__ == "__main__":
    import pandas as pd  # For timestamp formatting
    sys.exit(main())