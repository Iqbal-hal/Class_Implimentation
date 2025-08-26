#!/usr/bin/env python3
"""
Block Diagram Generator for Algorithmic Trading System

This module provides comprehensive block diagram generation capabilities that visualize
the program flow and module interconnections of the algorithmic trading portfolio 
management system.

Features:
- Automatic code analysis to extract dependencies and relationships
- Multiple output formats: PNG/SVG, interactive HTML, PDF, Mermaid
- Different layout algorithms: hierarchical, circular, force-directed
- CLI interface for easy execution
- Integration with existing project structure

Author: AI Assistant
Created: 2024
"""

import ast
import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
import traceback

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import graphviz
import seaborn as sns

# Data handling
import pandas as pd
import numpy as np


class CodeAnalyzer:
    """Analyzes Python code to extract structural information for diagram generation."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.modules = {}
        self.dependencies = defaultdict(set)
        self.classes = defaultdict(dict)
        self.functions = defaultdict(list)
        self.imports = defaultdict(set)
        self.method_calls = defaultdict(set)
        
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the entire project structure."""
        print("🔍 Starting comprehensive code analysis...")
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        print(f"📁 Found {len(python_files)} Python files to analyze")
        
        for py_file in python_files:
            if self._should_skip_file(py_file):
                continue
                
            try:
                self._analyze_file(py_file)
            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")
                
        return self._compile_results()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            '__pycache__',
            '.git',
            'test_',
            '_test',
            '.pyc',
            'backup',
            'reference'
        ]
        
        file_str = str(file_path).lower()
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                content = f.read()
                
            # Clean BOM and other problematic characters
            content = content.encode('utf-8', errors='ignore').decode('utf-8')
                
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Get relative module path
            rel_path = file_path.relative_to(self.project_root)
            module_name = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
            
            # Store module info
            self.modules[module_name] = {
                'file_path': str(file_path),
                'relative_path': str(rel_path),
                'size': len(content),
                'lines': len(content.splitlines())
            }
            
            # Analyze AST nodes
            visitor = self._create_ast_visitor(module_name)
            visitor.visit(tree)
            
        except Exception as e:
            print(f"⚠️  Failed to parse {file_path}: {e}")
    
    def _create_ast_visitor(self, module_name: str):
        """Create AST visitor for analyzing code structure."""
        
        class CodeVisitor(ast.NodeVisitor):
            def __init__(self, analyzer, module_name):
                self.analyzer = analyzer
                self.module_name = module_name
                self.current_class = None
                
            def visit_Import(self, node):
                for alias in node.names:
                    self.analyzer.imports[self.module_name].add(alias.name)
                    self.analyzer.dependencies[self.module_name].add(alias.name)
                
            def visit_ImportFrom(self, node):
                if node.module:
                    self.analyzer.imports[self.module_name].add(node.module)
                    self.analyzer.dependencies[self.module_name].add(node.module)
                    
            def visit_ClassDef(self, node):
                self.current_class = node.name
                class_info = {
                    'name': node.name,
                    'methods': [],
                    'bases': [base.id if hasattr(base, 'id') else str(base) for base in node.bases],
                    'line_number': node.lineno
                }
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'line_number': item.lineno,
                            'is_private': item.name.startswith('_'),
                            'is_property': any(isinstance(d, ast.Name) and d.id == 'property' 
                                             for d in item.decorator_list)
                        }
                        class_info['methods'].append(method_info)
                
                self.analyzer.classes[self.module_name][node.name] = class_info
                self.generic_visit(node)
                self.current_class = None
                
            def visit_FunctionDef(self, node):
                if self.current_class is None:  # Module-level function
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line_number': node.lineno,
                        'is_private': node.name.startswith('_')
                    }
                    self.analyzer.functions[self.module_name].append(func_info)
                self.generic_visit(node)
                
            def visit_Call(self, node):
                # Track method/function calls
                if hasattr(node.func, 'attr'):
                    call_name = node.func.attr
                    self.analyzer.method_calls[self.module_name].add(call_name)
                elif hasattr(node.func, 'id'):
                    call_name = node.func.id
                    self.analyzer.method_calls[self.module_name].add(call_name)
                self.generic_visit(node)
        
        return CodeVisitor(self, module_name)
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile analysis results into a structured format."""
        return {
            'modules': dict(self.modules),
            'dependencies': {k: list(v) for k, v in self.dependencies.items()},
            'classes': dict(self.classes),
            'functions': dict(self.functions),
            'imports': {k: list(v) for k, v in self.imports.items()},
            'method_calls': {k: list(v) for k, v in self.method_calls.items()},
            'statistics': {
                'total_modules': len(self.modules),
                'total_classes': sum(len(classes) for classes in self.classes.values()),
                'total_functions': sum(len(funcs) for funcs in self.functions.values()),
                'total_dependencies': sum(len(deps) for deps in self.dependencies.values())
            }
        }


class LayoutManager:
    """Manages different layout algorithms for diagram generation."""
    
    @staticmethod
    def create_hierarchical_layout(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout using NetworkX."""
        try:
            return nx.spring_layout(graph, k=3, iterations=50)
        except:
            return nx.random_layout(graph)
    
    @staticmethod
    def create_circular_layout(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create circular layout."""
        return nx.circular_layout(graph)
    
    @staticmethod
    def create_force_directed_layout(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Create force-directed layout."""
        try:
            return nx.spring_layout(graph, k=2, iterations=100)
        except:
            return nx.random_layout(graph)


class DiagramGenerator:
    """Main class for generating block diagrams in various formats."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = CodeAnalyzer(project_root)
        self.analysis_data = None
        self.output_dir = self.project_root / "diagram_output"
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_all_diagrams(self, layouts=['hierarchical'], formats=['png', 'html', 'mermaid']):
        """Generate all types of diagrams."""
        print("🚀 Starting comprehensive diagram generation...")
        
        # Analyze code first
        self.analysis_data = self.analyzer.analyze_project()
        
        results = {}
        
        for layout in layouts:
            for format_type in formats:
                try:
                    if format_type == 'png':
                        result = self._generate_matplotlib_diagram(layout)
                    elif format_type == 'svg':
                        result = self._generate_graphviz_diagram(layout, 'svg')
                    elif format_type == 'html':
                        result = self._generate_plotly_diagram(layout)
                    elif format_type == 'pdf':
                        result = self._generate_graphviz_diagram(layout, 'pdf')
                    elif format_type == 'mermaid':
                        result = self._generate_mermaid_diagram()
                    else:
                        continue
                        
                    key = f"{layout}_{format_type}"
                    results[key] = result
                    print(f"✅ Generated {key} diagram: {result}")
                    
                except Exception as e:
                    print(f"❌ Error generating {layout} {format_type}: {e}")
                    traceback.print_exc()
        
        # Generate summary report
        self._generate_summary_report()
        
        return results
    
    def _create_networkx_graph(self) -> nx.DiGraph:
        """Create NetworkX graph from analysis data."""
        G = nx.DiGraph()
        
        # Add nodes for modules
        for module_name, module_info in self.analysis_data['modules'].items():
            node_type = self._classify_module(module_name)
            G.add_node(module_name, 
                      type=node_type,
                      size=module_info['lines'],
                      path=module_info['relative_path'])
        
        # Add edges for dependencies
        for module, deps in self.analysis_data['dependencies'].items():
            for dep in deps:
                if dep in self.analysis_data['modules']:
                    G.add_edge(module, dep)
        
        return G
    
    def _classify_module(self, module_name: str) -> str:
        """Classify module type for visualization."""
        name_lower = module_name.lower()
        
        if 'config' in name_lower:
            return 'config'
        elif 'dashboard' in name_lower:
            return 'visualization'
        elif 'enhanced_stock_trading' in name_lower:
            return 'core_engine'
        elif 'support_files' in name_lower:
            return 'support'
        elif 'gui' in name_lower or 'streamlit' in name_lower:
            return 'interface'
        elif 'test' in name_lower:
            return 'test'
        else:
            return 'utility'
    
    def _generate_matplotlib_diagram(self, layout: str) -> str:
        """Generate diagram using matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Create graph
        G = self._create_networkx_graph()
        
        # Choose layout
        if layout == 'hierarchical':
            pos = LayoutManager.create_hierarchical_layout(G)
        elif layout == 'circular':
            pos = LayoutManager.create_circular_layout(G)
        else:
            pos = LayoutManager.create_force_directed_layout(G)
        
        # Color mapping for different module types
        color_map = {
            'core_engine': '#FF6B6B',    # Red
            'visualization': '#4ECDC4',  # Teal
            'support': '#45B7D1',        # Blue
            'config': '#96CEB4',         # Green
            'interface': '#FFEAA7',      # Yellow
            'utility': '#DDA0DD',        # Plum
            'test': '#FFB347'            # Orange
        }
        
        # Draw nodes with different colors based on type
        for node_type in color_map.keys():
            nodes = [n for n in G.nodes() if G.nodes[n].get('type') == node_type]
            if nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=color_map[node_type],
                                     node_size=800, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                              arrowsize=20, alpha=0.6, ax=ax)
        
        # Draw labels
        labels = {node: node.split('.')[-1] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        # Add title and legend
        ax.set_title(f'Algorithmic Trading System - Module Dependencies ({layout} layout)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Create legend
        legend_elements = [patches.Patch(color=color, label=type_name.replace('_', ' ').title()) 
                          for type_name, color in color_map.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save diagram
        output_file = self.output_dir / f"system_diagram_{layout}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _generate_plotly_diagram(self, layout: str) -> str:
        """Generate interactive HTML diagram using plotly."""
        G = self._create_networkx_graph()
        
        # Choose layout
        if layout == 'hierarchical':
            pos = LayoutManager.create_hierarchical_layout(G)
        elif layout == 'circular':
            pos = LayoutManager.create_circular_layout(G)
        else:
            pos = LayoutManager.create_force_directed_layout(G)
        
        # Extract node coordinates
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Color mapping
        color_map = {
            'core_engine': '#FF6B6B',
            'visualization': '#4ECDC4',
            'support': '#45B7D1',
            'config': '#96CEB4',
            'interface': '#FFEAA7',
            'utility': '#DDA0DD',
            'test': '#FFB347'
        }
        
        node_colors = [color_map.get(G.nodes[node].get('type', 'utility'), '#DDA0DD') 
                      for node in G.nodes()]
        
        # Create traces
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=1, color='gray'),
                               hoverinfo='none',
                               mode='lines')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=[node.split('.')[-1] for node in G.nodes()],
                               textposition="middle center",
                               marker=dict(size=30, color=node_colors, line=dict(width=2, color='white')))
        
        # Add hover text
        node_info = []
        for node in G.nodes():
            module_info = self.analysis_data['modules'].get(node, {})
            info = f"Module: {node}<br>"
            info += f"Type: {G.nodes[node].get('type', 'unknown')}<br>"
            info += f"Lines: {module_info.get('lines', 'unknown')}<br>"
            info += f"Dependencies: {len(self.analysis_data['dependencies'].get(node, []))}"
            node_info.append(info)
        
        node_trace.hovertext = node_info
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'Interactive Algorithmic Trading System Diagram ({layout} layout)',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Click and drag to interact with the diagram",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # Save HTML file
        output_file = self.output_dir / f"interactive_diagram_{layout}.html"
        fig.write_html(str(output_file))
        
        return str(output_file)
    
    def _generate_graphviz_diagram(self, layout: str, format_type: str) -> str:
        """Generate diagram using graphviz."""
        dot = graphviz.Digraph(comment='Trading System Architecture')
        dot.attr(rankdir='TB', size='16,12', dpi='300')
        
        # Color mapping
        color_map = {
            'core_engine': '#FF6B6B',
            'visualization': '#4ECDC4',
            'support': '#45B7D1',
            'config': '#96CEB4',
            'interface': '#FFEAA7',
            'utility': '#DDA0DD',
            'test': '#FFB347'
        }
        
        # Add nodes
        for module_name in self.analysis_data['modules'].keys():
            module_type = self._classify_module(module_name)
            short_name = module_name.split('.')[-1]
            
            dot.node(module_name, 
                    short_name,
                    style='filled',
                    fillcolor=color_map.get(module_type, '#DDA0DD'),
                    fontsize='10')
        
        # Add edges
        for module, deps in self.analysis_data['dependencies'].items():
            for dep in deps:
                if dep in self.analysis_data['modules']:
                    dot.edge(module, dep)
        
        # Save diagram
        output_file = self.output_dir / f"graphviz_diagram_{layout}"
        dot.render(str(output_file), format=format_type, cleanup=True)
        
        return f"{output_file}.{format_type}"
    
    def _generate_mermaid_diagram(self) -> str:
        """Generate Mermaid format diagram for GitHub integration."""
        mermaid_content = ["```mermaid", "graph TD"]
        
        # Color mapping for Mermaid
        color_classes = {
            'core_engine': 'class-core',
            'visualization': 'class-viz', 
            'support': 'class-support',
            'config': 'class-config',
            'interface': 'class-interface',
            'utility': 'class-utility',
            'test': 'class-test'
        }
        
        # Add nodes
        node_mapping = {}
        for i, module_name in enumerate(self.analysis_data['modules'].keys()):
            short_name = module_name.split('.')[-1]
            node_id = f"node{i}"
            node_mapping[module_name] = node_id
            
            module_type = self._classify_module(module_name)
            class_name = color_classes.get(module_type, 'class-utility')
            
            mermaid_content.append(f"    {node_id}[{short_name}]")
            mermaid_content.append(f"    class {node_id} {class_name}")
        
        # Add edges
        for module, deps in self.analysis_data['dependencies'].items():
            if module in node_mapping:
                for dep in deps:
                    if dep in node_mapping:
                        mermaid_content.append(f"    {node_mapping[module]} --> {node_mapping[dep]}")
        
        # Add style definitions
        mermaid_content.extend([
            "",
            "    classDef class-core fill:#FF6B6B,stroke:#000,stroke-width:2px",
            "    classDef class-viz fill:#4ECDC4,stroke:#000,stroke-width:2px", 
            "    classDef class-support fill:#45B7D1,stroke:#000,stroke-width:2px",
            "    classDef class-config fill:#96CEB4,stroke:#000,stroke-width:2px",
            "    classDef class-interface fill:#FFEAA7,stroke:#000,stroke-width:2px",
            "    classDef class-utility fill:#DDA0DD,stroke:#000,stroke-width:2px",
            "    classDef class-test fill:#FFB347,stroke:#000,stroke-width:2px",
            "```"
        ])
        
        # Save Mermaid file
        output_file = self.output_dir / "mermaid_diagram.md"
        with open(output_file, 'w') as f:
            f.write('\n'.join(mermaid_content))
        
        return str(output_file)
    
    def _generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        report_content = [
            "# Algorithmic Trading System - Block Diagram Analysis Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Overview",
            f"- **Total Modules**: {self.analysis_data['statistics']['total_modules']}",
            f"- **Total Classes**: {self.analysis_data['statistics']['total_classes']}",
            f"- **Total Functions**: {self.analysis_data['statistics']['total_functions']}",
            f"- **Total Dependencies**: {self.analysis_data['statistics']['total_dependencies']}",
            "",
            "## Module Classification",
        ]
        
        # Count modules by type
        type_counts = defaultdict(int)
        for module_name in self.analysis_data['modules'].keys():
            module_type = self._classify_module(module_name)
            type_counts[module_type] += 1
        
        for module_type, count in sorted(type_counts.items()):
            report_content.append(f"- **{module_type.replace('_', ' ').title()}**: {count} modules")
        
        report_content.extend([
            "",
            "## Key Components",
            "",
            "### Core Trading Engine",
            "- `Enhanced_stock_trading_V8.py`: Main portfolio management system",
            "- `FilteringAndBacktesting` class: Core trading logic and backtesting",
            "",
            "### Support Files Module", 
            "- Data extraction and processing utilities",
            "- Technical indicators computation",
            "- File I/O operations and logging",
            "",
            "### Visualization Dashboard",
            "- `dashboard_integration.py`: Trading dashboard creation",
            "- Interactive HTML generation and browser integration",
            "",
            "### Configuration Management",
            "- Strategy configuration and risk parameters",
            "- Filter management and selection",
            "",
            "## Generated Diagrams",
            "",
            "The following diagram files have been generated in the `diagram_output/` directory:",
            "",
        ])
        
        # List generated files
        for file in sorted(self.output_dir.glob("*")):
            if file.is_file():
                report_content.append(f"- `{file.name}`: {file.suffix.upper()[1:]} format diagram")
        
        report_content.extend([
            "",
            "## Usage Instructions",
            "",
            "1. **Static Diagrams**: Open PNG files for documentation and presentations",
            "2. **Interactive Diagrams**: Open HTML files in a web browser for interactive exploration",
            "3. **Mermaid Diagrams**: Use the `.md` file for GitHub integration",
            "4. **PDF Diagrams**: Use for formal reports and documentation",
            "",
            "## Integration with Existing Codebase",
            "",
            "The block diagram generator seamlessly integrates with the existing project structure:",
            "- Analyzes all Python files automatically",
            "- Respects the current module organization", 
            "- Generates diagrams that reflect actual code relationships",
            "- Provides multiple output formats for different use cases",
        ])
        
        # Save report
        output_file = self.output_dir / "analysis_report.md"
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"📋 Summary report generated: {output_file}")
        return str(output_file)


def main():
    """Command-line interface for the block diagram generator."""
    parser = argparse.ArgumentParser(
        description="Block Diagram Generator for Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python block_diagram_generator.py                    # Generate all diagrams with default settings
  python block_diagram_generator.py --layout circular # Use circular layout
  python block_diagram_generator.py --format html     # Generate only HTML format
  python block_diagram_generator.py --output ./diagrams # Custom output directory
        """
    )
    
    parser.add_argument(
        '--project-root', '-p',
        type=str,
        default='.',
        help='Root directory of the project to analyze (default: current directory)'
    )
    
    parser.add_argument(
        '--layout', '-l',
        choices=['hierarchical', 'circular', 'force-directed'],
        nargs='+',
        default=['hierarchical'],
        help='Layout algorithm(s) to use (default: hierarchical)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['png', 'svg', 'html', 'pdf', 'mermaid'],
        nargs='+',
        default=['png', 'html', 'mermaid'],
        help='Output format(s) to generate (default: png html mermaid)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for generated diagrams (default: PROJECT_ROOT/diagram_output)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DiagramGenerator(args.project_root)
    
    # Set custom output directory if provided
    if args.output:
        generator.output_dir = Path(args.output)
        generator.output_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Project root: {generator.project_root}")
    print(f"📁 Output directory: {generator.output_dir}")
    print(f"🎨 Layouts: {', '.join(args.layout)}")
    print(f"📄 Formats: {', '.join(args.format)}")
    
    # Generate diagrams
    try:
        results = generator.generate_all_diagrams(
            layouts=args.layout,
            formats=args.format
        )
        
        print("\n🎉 Diagram generation completed successfully!")
        print("\n📊 Generated files:")
        for key, file_path in results.items():
            print(f"  - {key}: {file_path}")
            
        print(f"\n📋 Check the analysis report at: {generator.output_dir / 'analysis_report.md'}")
        
    except Exception as e:
        print(f"❌ Error during diagram generation: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()