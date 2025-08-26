# Block Diagram Generator for Algorithmic Trading System

A comprehensive visualization suite that automatically analyzes and generates block diagrams for the algorithmic trading portfolio management system.

## 🌟 Features

### Visual Diagram Generation
- **System Architecture Diagrams**: Module dependencies and relationships
- **Trading Pipeline Flow**: Data flow from input to output 
- **Class Method Diagrams**: Method relationships within trading classes
- **Data Flow Analysis**: Interactive visualization of data transformations

### Multiple Output Formats
- **PNG/SVG**: High-quality static diagrams for documentation
- **Interactive HTML**: Clickable, zoomable diagrams with hover details
- **PDF Reports**: Comprehensive analysis documents
- **Mermaid**: GitHub-compatible diagram format
- **HTML Index**: Master page linking all generated diagrams

### Layout Algorithms
- **Hierarchical**: Top-down structure showing dependencies
- **Circular**: Radial layout highlighting interconnections
- **Force-Directed**: Physics-based layout for natural clustering

### Automatic Code Analysis
- **Import Dependencies**: Tracks module imports and usage
- **Class Relationships**: Analyzes inheritance and composition
- **Method Call Flows**: Maps function and method interactions
- **Data Transformations**: Identifies data flow patterns

## 🚀 Quick Start

### Basic Usage
```bash
# Generate all diagrams with default settings
python diagram_cli.py

# Generate specific diagram types
python diagram_cli.py --system-only
python diagram_cli.py --pipeline-only
python diagram_cli.py --enhanced-only

# Use specific formats and layouts
python diagram_cli.py --format png html --layout hierarchical circular

# Custom output directory
python diagram_cli.py --output ./my_diagrams
```

### Individual Generators
```bash
# Basic system diagrams
python block_diagram_generator.py --layout hierarchical --format png html mermaid

# Enhanced features
python enhanced_diagram_features.py

# Create HTML index
python diagram_cli.py --create-index
```

## 📊 Generated Diagrams

### System Architecture
- **Purpose**: Visualizes module dependencies and system structure
- **Files**: `system_diagram_*.png`, `interactive_diagram_*.html`
- **Use Cases**: Documentation, onboarding, architecture reviews

### Trading Pipeline
- **Purpose**: Shows data flow through the trading system
- **Files**: `trading_pipeline_diagram.png`
- **Use Cases**: Understanding system workflow, process optimization

### Class Methods
- **Purpose**: Details method relationships within key classes
- **Files**: `class_methods_*.png`
- **Use Cases**: Code reviews, refactoring planning, debugging

### Data Flow
- **Purpose**: Interactive analysis of data transformations
- **Files**: `data_flow_diagram.html`
- **Use Cases**: Performance analysis, data lineage tracking

## 🏗️ System Architecture

The block diagram generator analyzes the following key components:

### Input Layer
- **CSV Data Files**: `Nif50_5y_1w.csv` and other market data
- **Configuration Files**: `updated_config.py` with strategy parameters

### Support Files Module
- **scrip_extractor.py**: Stock data extraction utilities
- **compute_indicators_helper.py**: Technical indicator calculations
- **File_IO.py**: File input/output operations
- **dual_logger.py**: Logging system implementation
- **updated_config.py**: Strategy and risk configuration

### Core Trading Engine
- **Enhanced_stock_trading_V8.py**: Main trading system
- **FilteringAndBacktesting Class**:
  - `apply_filter()`: Stock screening and filtering
  - `calculate_stock_score()`: Multi-factor scoring algorithm
  - `allocate_portfolio()`: Intelligent capital allocation
  - `backtest_strategy()`: Strategy backtesting engine
  - `backtested_global_summary()`: Performance analysis

### Visualization Dashboard
- **dashboard_integration.py**: Dashboard creation and management
- **TradingDashboard Class**:
  - `prepare_dashboard_data()`: Data processing for visualization
  - `create_dashboard_html()`: HTML dashboard generation
  - `launch_dashboard()`: Browser integration
  - `export_dashboard_data()`: Data export capabilities

### Web Interface
- **streamlit_dashboard.py**: Configuration UI components
- Filter selection and risk parameter configuration
- Strategy configuration editor
- Backtest execution triggers

### Output Layer
- **Excel Reports**: `backtested_scrips.xlsx` and analysis files
- **Interactive Dashboard**: `trading_dashboard.html`
- **Log Files**: `portfolio_trading_log.txt`
- **JSON Exports**: `dashboard_data.json`

## 🛠️ Technical Implementation

### Code Analysis Engine
```python
class CodeAnalyzer:
    - Parses Python AST to extract structural information
    - Identifies imports, classes, functions, and method calls
    - Handles Unicode BOM and encoding issues
    - Generates dependency graphs and relationship maps
```

### Diagram Generation
```python
class DiagramGenerator:
    - Creates NetworkX graphs from code analysis
    - Supports multiple layout algorithms
    - Generates diagrams in various formats
    - Provides consistent styling and color coding
```

### Enhanced Features
```python
class EnhancedDiagramGenerator:
    - Trading-specific pipeline visualization
    - Class method flow diagrams
    - Data transformation analysis
    - PDF report generation with ReportLab
```

## 📈 Module Classification

The system automatically classifies modules into categories:

- **🔴 Core Engine**: Main trading logic (`Enhanced_stock_trading_V8`)
- **🟦 Support**: Utilities and helpers (`support_files/`)
- **🟢 Configuration**: Settings and parameters (`*config*`)
- **🟡 Interface**: User interfaces (`gui/`, `streamlit_dashboard`)
- **🟠 Visualization**: Charts and dashboards (`dashboard_integration`)
- **🟣 Utility**: General utilities and tools
- **🟤 Test**: Testing and validation code

## 🎨 Customization

### Color Schemes
Each module type has a distinctive color for easy identification:
```python
color_map = {
    'core_engine': '#FF6B6B',     # Red
    'visualization': '#4ECDC4',   # Teal
    'support': '#45B7D1',         # Blue
    'config': '#96CEB4',          # Green
    'interface': '#FFEAA7',       # Yellow
    'utility': '#DDA0DD',         # Plum
    'test': '#FFB347'             # Orange
}
```

### Layout Options
- **Hierarchical**: Best for showing clear dependencies
- **Circular**: Good for highlighting interconnected systems
- **Force-Directed**: Natural clustering of related components

### Export Formats
- **PNG**: High-resolution images for presentations
- **SVG**: Scalable vector graphics for web and print
- **HTML**: Interactive diagrams with zoom and pan
- **PDF**: Professional reports with multiple diagrams
- **Mermaid**: GitHub-integrated diagrams

## 🔧 CLI Reference

### Master CLI (`diagram_cli.py`)
```bash
# Generation modes
--all                    # Generate all diagram types (default)
--system-only           # System architecture only
--pipeline-only         # Trading pipeline only
--enhanced-only         # Enhanced analysis only
--class-only CLASS      # Class method diagrams
--data-flow-only        # Data flow diagrams
--pdf-only              # PDF reports only

# Format options
--format png svg html pdf mermaid    # Output formats
--layout hierarchical circular       # Layout algorithms

# Utility options
--create-index          # Generate HTML index page
--export-config FILE    # Save configuration
--import-config FILE    # Load configuration
--output DIR            # Custom output directory
```

### Basic Generator (`block_diagram_generator.py`)
```bash
--project-root PATH     # Project directory
--layout LAYOUT         # Layout algorithm
--format FORMAT         # Output format
--output DIR            # Output directory
--verbose               # Detailed output
```

## 📁 File Structure

```
diagram_output/
├── index.html                           # Master index page
├── system_diagram_hierarchical.png      # System architecture (hierarchical)
├── system_diagram_circular.png          # System architecture (circular)
├── interactive_diagram_hierarchical.html # Interactive system diagram
├── trading_pipeline_diagram.png         # Trading pipeline flow
├── class_methods_FilteringAndBacktesting.png # Class method diagram
├── data_flow_diagram.html               # Interactive data flow
├── mermaid_diagram.md                   # GitHub-compatible diagram
├── comprehensive_analysis_report.pdf    # Complete PDF report
└── analysis_report.md                   # Markdown summary
```

## 🔄 Integration with Existing Codebase

The block diagram generator seamlessly integrates with the existing project:

1. **No Code Modification Required**: Analyzes existing Python files without changes
2. **Respects Project Structure**: Follows the established module organization
3. **Automatic Discovery**: Finds and analyzes all Python files recursively
4. **Error Handling**: Gracefully handles encoding issues and parsing errors
5. **Minimal Dependencies**: Uses standard scientific Python libraries

## 🎯 Use Cases

### Development Team
- **Code Reviews**: Visualize system architecture before reviews
- **Onboarding**: Help new team members understand the codebase
- **Refactoring**: Identify tightly coupled modules for improvement
- **Documentation**: Generate up-to-date architecture diagrams

### Project Management
- **Progress Tracking**: Visualize system complexity and growth
- **Risk Assessment**: Identify critical dependencies and bottlenecks
- **Planning**: Understand impact of proposed changes
- **Communication**: Present system architecture to stakeholders

### Quality Assurance
- **Test Planning**: Understand component relationships for test coverage
- **Bug Analysis**: Trace dependencies to find root causes
- **Performance**: Identify potential performance bottlenecks
- **Maintenance**: Plan maintenance windows based on dependencies

## 🚀 Future Enhancements

- **Real-time Updates**: Monitor code changes and update diagrams automatically
- **Dependency Analysis**: Detect circular dependencies and suggest improvements
- **Performance Metrics**: Integrate code complexity and performance data
- **Integration APIs**: RESTful API for integration with CI/CD pipelines
- **Custom Themes**: User-defined color schemes and styling options
- **3D Visualizations**: Three-dimensional system architecture views

## 📝 Requirements

### Python Packages
```bash
pip install matplotlib plotly networkx graphviz seaborn pandas numpy reportlab
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/
```

## 🤝 Contributing

1. **Fork the Repository**: Create your own fork for modifications
2. **Create Feature Branch**: Develop new features in separate branches
3. **Add Tests**: Include tests for new functionality
4. **Update Documentation**: Keep documentation current with changes
5. **Submit Pull Request**: Submit changes for review

## 📄 License

This block diagram generator is part of the algorithmic trading system project. Please refer to the main project license for usage terms and conditions.

---

*Generated by Block Diagram Generator for Algorithmic Trading System*