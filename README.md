# 🤖 Ollama Data Analysis Agent

> **AI-Powered Local Data Analysis with Professional PDF Reports**

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/ollama-local%20LLM-green.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

A powerful, privacy-first data analysis agent that runs completely locally using Ollama. Ask questions about your data in natural language, get AI-generated Python code, and receive comprehensive PDF reports - all without your data ever leaving your machine.

<p align="center">
  <img src="https://img.shields.io/badge/AI-Generative%20AI-blueviolet" />
  <img src="https://img.shields.io/badge/Privacy-100%25%20Local-brightgreen" />
  <img src="https://img.shields.io/badge/Cost-$0-success" />
</p>

---

## ✨ Features

### 🧠 **AI-Powered Analysis**
- **Natural Language Queries**: Ask questions like "What are the peak hours for rides?"
- **Automatic Code Generation**: AI writes pandas/numpy code for you
- **Context-Aware**: Understands your data structure and generates relevant analysis
- **Error Recovery**: Automatically fixes common syntax errors

### 🔒 **100% Local & Private**
- **No Cloud**: Everything runs on your machine
- **Zero Data Leakage**: Your sensitive data stays completely private
- **Offline Capable**: Works without internet after setup
- **No API Costs**: Free forever, no usage limits

### 📄 **Professional Reports**
- **Automated PDF Generation**: Business-ready formatted reports
- **4 Key Sections**: Overview, Statistics, Insights, Quality Assessment
- **Clean Formatting**: No markdown artifacts, just professional output
- **Actual Results**: Shows real analysis, not just code

### 💻 **Developer Friendly**
- **Interactive CLI**: Simple command-line interface
- **Conversation Memory**: Tracks your analysis history
- **Extensible**: Easy to add new features
- **Well Documented**: Clear code with comprehensive guides

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Ollama** installed
- **8GB+ RAM** (16GB recommended for larger models)

### Installation
```bash
# 1. Install Ollama
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - Download from https://ollama.com/download

# 2. Start Ollama and pull a model
ollama serve
ollama pull llama3.2  # Recommended for balance of speed/quality

# 3. Clone and setup
git clone https://github.com/yourusername/ollama-data-agent.git
cd ollama-data-agent

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the agent!
python ollama_data_agent.py
```

---

## 📖 Usage

### Basic Commands
```bash
🤖 Agent> load data.csv          # Load your dataset
🤖 Agent> summary               # Get AI-powered overview
🤖 Agent> suggest               # Get analysis recommendations
🤖 Agent> ask [question]        # Natural language query
🤖 Agent> report                # Generate comprehensive PDF
🤖 Agent> history               # View past analyses
🤖 Agent> quit                  # Exit
```

### Example Workflow
```bash
# Start the agent
python ollama_data_agent.py

# Load your Uber dataset
🤖 Agent> load uber_rides.csv
📊 Data loaded successfully!
   Shape: (150000, 21)
   Columns: ['Booking_ID', 'Vehicle_Type', 'Pickup_Location', ...]

# Get AI-powered summary
🤖 Agent> summary
🤔 Analyzing data with AI...

DATASET OVERVIEW:
Large ride-sharing dataset with comprehensive booking information...

KEY PATTERNS:
- High incomplete ride rate at 30%
- UPI payment dominates with 45,909 transactions
- Peak hours are 8-10 AM and 6-8 PM

# Ask specific questions
🤖 Agent> ask What are the top 5 vehicle types by revenue?
🧠 Generating analysis...
[AI generates pandas groupby code]

🔧 Execute the generated code? (y/n): y
⚡ Executing code...
📋 Result:
Vehicle_Type    Total_Revenue
Premium         $2,450,000
Comfort         $1,980,000
Economy         $1,650,000
...

📋 Add this analysis to PDF report? (y/n): y
✅ Added to report queue

# Generate final report
🤖 Agent> report
🔍 Generating comprehensive analysis report...
📊 Generating data overview...
📈 Generating statistical summary...
💡 Generating key insights...
🔍 Generating data quality assessment...
📄 PDF report generated: analysis_outputs/reports/report_20241201_143022.pdf
```

---

## 📊 Sample Questions You Can Ask

### For Uber/Ride-Sharing Data
```
- What are the peak hours for rides?
- Which vehicle types generate the most revenue?
- What are the top 10 pickup locations by ride volume?
- How do customer ratings correlate with ride distance?
- What percentage of rides are cancelled and why?
- Show me the average ride duration by vehicle type
- What's the distribution of payment methods?
```

### For General Business Data
```
- What's the month-over-month growth trend?
- Can you segment customers by usage patterns?
- Which products have the highest profit margins?
- Show me sales by region for Q4
- What are the common reasons for customer churn?
```

---


## 🛠️ Configuration

### Choose Your Model

Edit the agent initialization:
```python
# Fast and efficient (recommended for most tasks)
agent = OllamaDataAgent(model_name="llama3.2")

# Better for complex code generation
agent = OllamaDataAgent(model_name="codellama")

# Most powerful (requires 16GB+ RAM)
agent = OllamaDataAgent(model_name="llama3.1")
```

### Custom Ollama Server
```python
agent = OllamaDataAgent(
    model_name="llama3.2",
    base_url="http://your-server:11434"
)
```

---

## 📋 Requirements
```txt
pandas>=2.0.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
requests>=2.28.0       # API communication
tabulate>=0.9.0        # Table formatting
openpyxl>=3.1.0        # Excel support
reportlab>=4.0.0       # PDF generation
```

---


## 🎯 Use Cases

### Perfect For:
- 📊 **Data Analysts** - Quick exploratory analysis
- 🏢 **Business Analysts** - Automated reporting
- 👔 **Executives** - Business intelligence insights
- 🔒 **Privacy-Conscious Users** - Sensitive data analysis
- 🎓 **Students/Learners** - Understanding data analysis
- 💼 **Consultants** - Client data analysis

### Industries:
- 🚕 Ride-sharing & Transportation
- 🛒 E-commerce & Retail
- 🏥 Healthcare (HIPAA-compliant)
- 💰 Finance & Banking
- 📱 SaaS & Technology
- 🏭 Manufacturing & Operations

---


## 📊 Example Output

### Generated PDF Report Includes:
1. **Title Page** - Dataset metadata and info
2. **Data Overview** - AI-powered summary and patterns
3. **Statistical Analysis** - Descriptive stats and insights
4. **Business Insights** - Key findings and recommendations
5. **Data Quality** - Completeness and issue assessment

### Sample Insights:
```
KEY PATTERNS IDENTIFIED:
- 30% incomplete ride rate indicates service delivery issues
- UPI payment method dominates (45,909 transactions)
- Peak demand at 8-10 AM and 6-8 PM (40% higher)
- Premium vehicles generate 3x revenue vs Economy

RECOMMENDATIONS:
- Implement automated vehicle maintenance alerts
- Optimize driver allocation during peak hours
- Investigate incomplete ride causes in high-frequency zones
- Expand premium vehicle fleet in high-revenue locations
```


