#!/usr/bin/env python3
"""
Enhanced Block Diagram Features for Algorithmic Trading System

This module provides additional features for the block diagram generator:
- Module-specific flow diagrams
- Class method call flow visualization
- Data flow analysis
- Trading pipeline visualization
- Enhanced PDF generation with ReportLab

Author: AI Assistant
Created: 2024
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import json

# Additional imports for enhanced features
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Try to import ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Import our main generator
from block_diagram_generator import DiagramGenerator, CodeAnalyzer


class EnhancedDiagramGenerator(DiagramGenerator):
    """Enhanced diagram generator with additional visualization features."""
    
    def __init__(self, project_root: str):
        super().__init__(project_root)
        self.trading_pipeline = None
        
    def generate_trading_pipeline_diagram(self) -> str:
        """Generate a specific diagram showing the trading pipeline flow."""
        print("📈 Generating trading pipeline diagram...")
        
        # Define the trading pipeline based on code analysis
        pipeline_stages = [
            {
                'name': 'Data Input',
                'modules': ['input_data', 'File_IO'],
                'description': 'CSV data files\n(Nif50_5y_1w.csv)',
                'color': '#E8F4FD'
            },
            {
                'name': 'Configuration',
                'modules': ['updated_config', 'config'],
                'description': 'Strategy configuration\nRisk parameters',
                'color': '#D4E6F1'
            },
            {
                'name': 'Data Processing',
                'modules': ['scrip_extractor', 'compute_indicators_helper'],
                'description': 'Data extraction\nTechnical indicators',
                'color': '#D5DBDB'
            },
            {
                'name': 'Core Trading Engine',
                'modules': ['Enhanced_stock_trading_V8'],
                'description': 'FilteringAndBacktesting\nPortfolio allocation\nBacktesting',
                'color': '#FADBD8'
            },
            {
                'name': 'Visualization',
                'modules': ['dashboard_integration'],
                'description': 'Interactive dashboard\nCharts and reports',
                'color': '#D5F4E6'
            },
            {
                'name': 'Output',
                'modules': ['output_data'],
                'description': 'Excel reports\nHTMLdashboard\nLog files',
                'color': '#FCF3CF'
            }
        ]
        
        # Create the pipeline diagram
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Calculate positions
        stage_width = 2.2
        stage_height = 1.5
        spacing = 0.5
        total_width = len(pipeline_stages) * stage_width + (len(pipeline_stages) - 1) * spacing
        start_x = -total_width / 2
        
        y_pos = 0
        
        # Draw pipeline stages
        for i, stage in enumerate(pipeline_stages):
            x_pos = start_x + i * (stage_width + spacing)
            
            # Create rounded rectangle for stage
            box = FancyBboxPatch(
                (x_pos, y_pos), stage_width, stage_height,
                boxstyle="round,pad=0.1",
                facecolor=stage['color'],
                edgecolor='#2C3E50',
                linewidth=2
            )
            ax.add_patch(box)
            
            # Add stage name
            ax.text(x_pos + stage_width/2, y_pos + stage_height - 0.3,
                   stage['name'], 
                   ha='center', va='center',
                   fontsize=12, fontweight='bold')
            
            # Add description
            ax.text(x_pos + stage_width/2, y_pos + stage_height/2 - 0.1,
                   stage['description'],
                   ha='center', va='center',
                   fontsize=9, style='italic')
            
            # Add module names
            module_text = '\n'.join(stage['modules'][:2])  # Limit to 2 modules
            if len(stage['modules']) > 2:
                module_text += f'\n+{len(stage["modules"])-2} more'
            
            ax.text(x_pos + stage_width/2, y_pos + 0.2,
                   module_text,
                   ha='center', va='center',
                   fontsize=8, color='#2C3E50')
            
            # Draw arrow to next stage
            if i < len(pipeline_stages) - 1:
                arrow_start_x = x_pos + stage_width
                arrow_end_x = start_x + (i + 1) * (stage_width + spacing)
                
                ax.annotate('', xy=(arrow_end_x, y_pos + stage_height/2),
                           xytext=(arrow_start_x, y_pos + stage_height/2),
                           arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))
        
        # Add title and styling
        ax.set_title('Algorithmic Trading System - Data Flow Pipeline',
                    fontsize=18, fontweight='bold', pad=30)
        
        # Set axis limits and remove axes
        ax.set_xlim(start_x - 0.5, start_x + total_width + 0.5)
        ax.set_ylim(-0.5, stage_height + 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save diagram
        output_file = self.output_dir / "trading_pipeline_diagram.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Trading pipeline diagram saved: {output_file}")
        return str(output_file)
    
    def generate_class_method_diagram(self, class_name: str = "FilteringAndBacktesting") -> str:
        """Generate a diagram showing class methods and their relationships."""
        print(f"🔍 Generating class method diagram for {class_name}...")
        
        if not self.analysis_data:
            self.analysis_data = self.analyzer.analyze_project()
        
        # Find the class in analysis data
        target_class = None
        for module_name, classes in self.analysis_data['classes'].items():
            if class_name in classes:
                target_class = classes[class_name]
                break
        
        if not target_class:
            print(f"⚠️  Class {class_name} not found in analysis data")
            return ""
        
        # Create method flow diagram
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        methods = target_class['methods']
        
        # Categorize methods
        public_methods = [m for m in methods if not m['is_private']]
        private_methods = [m for m in methods if m['is_private']]
        
        # Calculate layout
        methods_per_row = 4
        method_width = 2.5
        method_height = 0.8
        spacing_x = 0.3
        spacing_y = 0.5
        
        # Draw public methods
        y_start = 4
        ax.text(0, y_start + 1, f'{class_name} - Public Methods', 
               fontsize=16, fontweight='bold', ha='left')
        
        for i, method in enumerate(public_methods):
            row = i // methods_per_row
            col = i % methods_per_row
            
            x = col * (method_width + spacing_x)
            y = y_start - row * (method_height + spacing_y)
            
            # Method box
            box = FancyBboxPatch(
                (x, y), method_width, method_height,
                boxstyle="round,pad=0.05",
                facecolor='#E8F6F3',
                edgecolor='#16A085',
                linewidth=1.5
            )
            ax.add_patch(box)
            
            # Method name
            ax.text(x + method_width/2, y + method_height/2,
                   method['name'],
                   ha='center', va='center',
                   fontsize=10, fontweight='bold')
            
            # Parameters
            params_text = f"({', '.join(method['args'][:3])})"
            if len(method['args']) > 3:
                params_text = f"({', '.join(method['args'][:3])}, ...)"
            
            ax.text(x + method_width/2, y + 0.1,
                   params_text,
                   ha='center', va='center',
                   fontsize=8, style='italic')
        
        # Draw private methods
        private_y_start = y_start - len(public_methods) // methods_per_row * (method_height + spacing_y) - 1.5
        if private_methods:
            ax.text(0, private_y_start + 0.5, f'{class_name} - Private Methods', 
                   fontsize=14, fontweight='bold', ha='left', color='#7F8C8D')
            
            for i, method in enumerate(private_methods[:8]):  # Limit to 8 private methods
                row = i // methods_per_row
                col = i % methods_per_row
                
                x = col * (method_width + spacing_x)
                y = private_y_start - row * (method_height + spacing_y)
                
                # Method box
                box = FancyBboxPatch(
                    (x, y), method_width, method_height,
                    boxstyle="round,pad=0.05",
                    facecolor='#FDEDEC',
                    edgecolor='#E74C3C',
                    linewidth=1.5
                )
                ax.add_patch(box)
                
                # Method name
                ax.text(x + method_width/2, y + method_height/2,
                       method['name'],
                       ha='center', va='center',
                       fontsize=9, fontweight='bold')
        
        # Add method flow arrows for key methods
        key_flow = [
            ('apply_filter', 'calculate_stock_score'),
            ('calculate_stock_score', 'allocate_portfolio'),
            ('allocate_portfolio', 'backtest_strategy'),
            ('backtest_strategy', 'backtested_global_summary')
        ]
        
        method_positions = {}
        for i, method in enumerate(public_methods):
            row = i // methods_per_row
            col = i % methods_per_row
            x = col * (method_width + spacing_x) + method_width/2
            y = y_start - row * (method_height + spacing_y) + method_height/2
            method_positions[method['name']] = (x, y)
        
        # Draw flow arrows
        for source, target in key_flow:
            if source in method_positions and target in method_positions:
                source_pos = method_positions[source]
                target_pos = method_positions[target]
                
                ax.annotate('', xy=target_pos, xytext=source_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='#3498DB',
                                         connectionstyle="arc3,rad=0.1"))
        
        # Set limits and styling
        max_x = max(len(public_methods), len(private_methods)) // methods_per_row * (method_width + spacing_x)
        ax.set_xlim(-0.5, max(max_x, 8))
        ax.set_ylim(private_y_start - 2, y_start + 2)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save diagram
        output_file = self.output_dir / f"class_methods_{class_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Class method diagram saved: {output_file}")
        return str(output_file)
    
    def generate_data_flow_diagram(self) -> str:
        """Generate a data flow diagram showing how data moves through the system."""
        print("📊 Generating data flow diagram...")
        
        # Create plotly diagram showing data transformations
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Input Data', 'Processing', 'Analysis', 
                          'Backtesting', 'Visualization', 'Output'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Define data flow stages
        stages = [
            {
                'name': 'CSV Input',
                'data': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
                'row': 1, 'col': 1
            },
            {
                'name': 'Technical Indicators',
                'data': ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger'],
                'row': 1, 'col': 2
            },
            {
                'name': 'Filtered Signals',
                'data': ['Buy_Signal', 'Sell_Signal', 'Score'],
                'row': 1, 'col': 3
            },
            {
                'name': 'Portfolio Data',
                'data': ['Allocation', 'Entry_Price', 'Exit_Price', 'PnL'],
                'row': 2, 'col': 1
            },
            {
                'name': 'Dashboard Data',
                'data': ['Charts', 'Statistics', 'Transactions'],
                'row': 2, 'col': 2
            },
            {
                'name': 'Reports',
                'data': ['Excel', 'HTML', 'JSON', 'Logs'],
                'row': 2, 'col': 3
            }
        ]
        
        # Add data boxes to each subplot
        for stage in stages:
            y_positions = list(range(len(stage['data'])))
            x_positions = [0] * len(stage['data'])
            
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode='markers+text',
                    text=stage['data'],
                    textposition="middle right",
                    marker=dict(size=20, color='lightblue'),
                    showlegend=False,
                    name=stage['name']
                ),
                row=stage['row'], col=stage['col']
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Algorithmic Trading System - Data Flow Analysis",
                font=dict(size=16)
            ),
            height=800,
            showlegend=False
        )
        
        # Hide axes for all subplots
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
        
        # Save diagram
        output_file = self.output_dir / "data_flow_diagram.html"
        fig.write_html(str(output_file))
        
        print(f"✅ Data flow diagram saved: {output_file}")
        return str(output_file)
    
    def generate_pdf_report(self) -> str:
        """Generate a comprehensive PDF report with all diagrams."""
        if not REPORTLAB_AVAILABLE:
            print("⚠️  ReportLab not available. Skipping PDF generation.")
            return ""
            
        print("📄 Generating comprehensive PDF report...")
        
        output_file = self.output_dir / "comprehensive_analysis_report.pdf"
        doc = SimpleDocTemplate(str(output_file), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Algorithmic Trading System", title_style))
        story.append(Paragraph("Block Diagram Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Introduction
        intro_text = """
        This comprehensive report provides a detailed analysis of the algorithmic trading 
        portfolio management system through various block diagrams and visualizations. 
        The system implements intelligent portfolio allocation, backtesting, and 
        visualization capabilities.
        """
        story.append(Paragraph(intro_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # System Statistics
        if self.analysis_data:
            stats_data = [
                ['Metric', 'Count'],
                ['Total Modules', str(self.analysis_data['statistics']['total_modules'])],
                ['Total Classes', str(self.analysis_data['statistics']['total_classes'])],
                ['Total Functions', str(self.analysis_data['statistics']['total_functions'])],
                ['Total Dependencies', str(self.analysis_data['statistics']['total_dependencies'])]
            ]
            
            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("System Overview", styles['Heading2']))
            story.append(stats_table)
            story.append(Spacer(1, 20))
        
        # Add images if they exist
        image_files = [
            ('System Architecture (Hierarchical)', 'system_diagram_hierarchical.png'),
            ('System Architecture (Circular)', 'system_diagram_circular.png'),
            ('Trading Pipeline', 'trading_pipeline_diagram.png'),
            ('Class Methods', 'class_methods_FilteringAndBacktesting.png')
        ]
        
        for title, filename in image_files:
            image_path = self.output_dir / filename
            if image_path.exists():
                story.append(Paragraph(title, styles['Heading2']))
                story.append(Spacer(1, 10))
                
                # Add image with proper scaling
                img = Image(str(image_path))
                img_width, img_height = img.imageWidth, img.imageHeight
                aspect = img_height / float(img_width)
                
                # Scale to fit page width
                display_width = 6 * inch
                display_height = display_width * aspect
                
                if display_height > 8 * inch:  # Too tall, scale down
                    display_height = 8 * inch
                    display_width = display_height / aspect
                
                img.drawWidth = display_width
                img.drawHeight = display_height
                
                story.append(img)
                story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {output_file}")
        return str(output_file)
    
    def generate_enhanced_analysis(self) -> Dict[str, str]:
        """Generate all enhanced diagrams and analysis."""
        print("🚀 Starting enhanced diagram analysis...")
        
        # First run the basic analysis
        if not self.analysis_data:
            self.analysis_data = self.analyzer.analyze_project()
        
        results = {}
        
        try:
            # Generate enhanced diagrams
            results['trading_pipeline'] = self.generate_trading_pipeline_diagram()
            results['class_methods'] = self.generate_class_method_diagram()
            results['data_flow'] = self.generate_data_flow_diagram()
            results['pdf_report'] = self.generate_pdf_report()
            
            print("✅ Enhanced analysis completed!")
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
        
        return results


def main():
    """CLI for enhanced diagram features."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Block Diagram Features")
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--class-name', default='FilteringAndBacktesting', 
                       help='Class name for method diagram')
    
    args = parser.parse_args()
    
    generator = EnhancedDiagramGenerator(args.project_root)
    results = generator.generate_enhanced_analysis()
    
    print("\n📊 Enhanced Analysis Results:")
    for key, file_path in results.items():
        if file_path:
            print(f"  - {key}: {file_path}")


if __name__ == "__main__":
    main()