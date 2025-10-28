#!/usr/bin/env python3
"""
Ollama ML Data Analysis Agent with Prediction Capabilities
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import traceback
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

class OllamaMLAgent:
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        """Initialize the Ollama ML Data Analysis Agent"""
        self.model_name = model_name
        self.base_url = base_url
        self.data_context = {}
        self.analysis_history = []
        self.models = {}
        
        # Create directories
        self.output_dir = Path("analysis_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        self.report_sections = []
        
        print(f"Initializing Ollama ML Agent with model: {model_name}")
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                print(f"Connected to Ollama. Available models: {available_models}")
            else:
                raise Exception("Failed to connect")
        except Exception as e:
            print(f"Cannot connect to Ollama at {self.base_url}")
            print("Make sure Ollama is running with: ollama serve")
            sys.exit(1)
    
    def load_data(self, file_path):
        """Load dataset"""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.df = pd.read_json(file_path)
            else:
                raise ValueError("Supported formats: CSV, JSON")
            
            self.data_context = {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict(),
                'missing_values': self.df.isnull().sum().to_dict(),
                'file_path': file_path
            }
            
            print(f"Data loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def prepare_time_features(self, date_column):
        """Extract time-based features from date column"""
        if not hasattr(self, 'df'):
            return "No data loaded."
        
        try:
            self.df[date_column] = pd.to_datetime(self.df[date_column])
            
            # Extract features
            self.df['hour'] = self.df[date_column].dt.hour
            self.df['day_of_week'] = self.df[date_column].dt.dayofweek
            self.df['day'] = self.df[date_column].dt.day
            self.df['month'] = self.df[date_column].dt.month
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            
            # Peak hours classification
            self.df['time_category'] = pd.cut(
                self.df['hour'], 
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            print(f"Time features extracted from {date_column}")
            print("New columns: hour, day_of_week, day, month, is_weekend, time_category")
            return True
            
        except Exception as e:
            print(f"Error preparing time features: {str(e)}")
            return f"Error: {str(e)}"
    
    def analyze_peak_patterns(self, date_column, value_column):
        """Analyze and predict peak demand patterns"""
        if not hasattr(self, 'df'):
            return "No data loaded."
        
        try:
            print("Analyzing peak patterns...")
            
            # Ensure time features exist
            if 'hour' not in self.df.columns:
                self.prepare_time_features(date_column)
            
            # Hourly analysis
            hourly_demand = self.df.groupby('hour')[value_column].agg([
                'count', 'mean', 'sum', 'std'
            ]).round(2)
            hourly_demand.columns = ['num_records', 'avg_value', 'total_value', 'std_dev']
            
            # Identify peak hours
            threshold = hourly_demand['num_records'].quantile(0.75)
            peak_hours = hourly_demand[hourly_demand['num_records'] > threshold].index.tolist()
            
            # Day of week analysis
            dow_demand = self.df.groupby('day_of_week')[value_column].agg([
                'count', 'mean', 'sum'
            ]).round(2)
            dow_demand.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_demand.columns = ['num_records', 'avg_value', 'total_value']
            
            results = f"""
Peak Pattern Analysis Results

PEAK HOURS IDENTIFIED:
{', '.join([f'{h}:00' for h in peak_hours])}

HOURLY DEMAND PATTERN:
{hourly_demand.to_string()}

DAY OF WEEK PATTERN:
{dow_demand.to_string()}

KEY INSIGHTS:
- Busiest Hour: {hourly_demand['num_records'].idxmax()}:00 ({hourly_demand['num_records'].max():.0f} records)
- Quietest Hour: {hourly_demand['num_records'].idxmin()}:00 ({hourly_demand['num_records'].min():.0f} records)
- Busiest Day: {dow_demand['num_records'].idxmax()}
            """
            
            print(results)
            
            # Add to report
            self.add_to_report(
                "Peak Time Pattern Analysis",
                results,
                {
                    'hourly_demand': hourly_demand,
                    'day_of_week_demand': dow_demand
                }
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error in peak analysis: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg
    
    def train_demand_prediction(self, target_column, feature_columns=None):
        """Train a demand/volume prediction model"""
        if not hasattr(self, 'df'):
            return "No data loaded."
        
        try:
            print(f"Training demand prediction model for: {target_column}")
            
            # Auto-select features if not provided
            if feature_columns is None:
                time_features = ['hour', 'day_of_week', 'day', 'month', 'is_weekend']
                numerical_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                
                # Remove duplicates and target column
                all_features = time_features + numerical_cols
                feature_columns = []
                seen = set()
                for col in all_features:
                    if col in self.df.columns and col != target_column and col not in seen:
                        feature_columns.append(col)
                        seen.add(col)
            
            print(f"Using {len(feature_columns)} features: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")
            
            # Prepare data
            X = self.df[feature_columns].fillna(0)
            y = self.df[target_column].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            print("Training Random Forest model...")
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save model
            model_name = f"demand_model_{target_column}"
            model_path = self.model_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            
            self.models[model_name] = {
                'model': model,
                'features': feature_columns,
                'target': target_column,
                'metrics': {
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'rmse': rmse,
                    'mae': mae
                }
            }
            
            results = f"""
Model Training Complete!

Target Variable: {target_column}
Features Used: {len(feature_columns)}

Performance Metrics:
- Training R²: {train_score:.4f}
- Testing R²: {test_score:.4f}
- RMSE: {rmse:.2f}
- MAE: {mae:.2f}

Top 5 Important Features:
{feature_importance.head().to_string()}

Model saved to: {model_path}
            """
            
            print(results)
            
            # Add to report
            self.add_to_report(
                f"Demand Prediction Model - {target_column}",
                results,
                {'feature_importance': feature_importance.head(10)}
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error training model: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg
    
    def predict_demand(self, model_name, input_data):
        """Make predictions using trained model - ALWAYS returns a dict"""
        try:
            if model_name not in self.models:
                return {
                    'error': f"Model {model_name} not found",
                    'available_models': list(self.models.keys())
                }
            
            model_info = self.models[model_name]
            model = model_info['model']
            features = model_info['features']
            
            # Prepare input
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()
            
            # Ensure all features present
            for feat in features:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            
            # Make prediction
            prediction = model.predict(input_df[features])
            
            return {
                'prediction': float(prediction[0]),
                'model': model_name,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }
    
    def run_ml_pipeline(self, date_column, target_column):
        """Run complete ML analysis pipeline"""
        print("Starting comprehensive ML analysis pipeline...\n")
        
        # Step 1: Feature engineering
        print("Step 1: Feature Engineering")
        result = self.prepare_time_features(date_column)
        if isinstance(result, str) and result.startswith("Error"):
            print(f"Failed at Step 1: {result}")
            return result
        
        # Step 2: Peak pattern analysis
        print("\nStep 2: Peak Pattern Analysis")
        peak_results = self.analyze_peak_patterns(date_column, target_column)
        
        # Step 3: Train prediction model
        print("\nStep 3: Training Prediction Model")
        model_results = self.train_demand_prediction(target_column)
        
        # Step 4: Generate predictions for next 24 hours
        print("\nStep 4: Generating Future Predictions")
        model_name = f"demand_model_{target_column}"
        
        if model_name not in self.models:
            error_msg = f"Model {model_name} was not created. Check Step 3 for errors."
            print(error_msg)
            return error_msg
        
        # Get feature means for prediction inputs
        model_features = self.models[model_name]['features']
        feature_means = {}
        for feat in model_features:
            if feat in self.df.columns:
                feature_means[feat] = float(self.df[feat].mean())
        
        future_predictions = []
        
        for hour in range(24):
            # Create input with all required features
            pred_input = {
                'hour': hour,
                'day_of_week': 1,
                'is_weekend': 0,
                'month': datetime.now().month,
                'day': datetime.now().day
            }
            
            # Add feature means for other columns
            pred_input.update(feature_means)
            
            # Make prediction
            pred_result = self.predict_demand(model_name, pred_input)
            
            if pred_result.get('success'):
                future_predictions.append({
                    'hour': f"{hour:02d}:00",
                    'predicted_demand': round(pred_result['prediction'], 2)
                })
            else:
                print(f"Prediction failed for hour {hour}: {pred_result.get('error')}")
                future_predictions.append({
                    'hour': f"{hour:02d}:00",
                    'predicted_demand': 0.0
                })
        
        pred_df = pd.DataFrame(future_predictions)
        
        summary = f"""
Complete ML Analysis Pipeline Executed

Steps Completed:
1. Time-based feature engineering
2. Peak pattern identification
3. Demand prediction model training
4. 24-hour demand forecasting

Next 24 Hours Forecast:
{pred_df.to_string()}

Peak Predicted Hours:
{pred_df.nlargest(5, 'predicted_demand').to_string()}

All results saved to analysis outputs.
        """
        
        print(summary)
        
        # Add predictions to report
        self.add_to_report(
            "24-Hour Demand Forecast",
            "Predicted demand for the next 24 hours based on trained ML model",
            {'predictions': pred_df}
        )
        
        return summary
    
    def add_to_report(self, title, content, tables=None):
        """Add analysis section to report"""
        section = {
            'title': title,
            'content': content,
            'tables': tables or {},
            'timestamp': datetime.now().isoformat()
        }
        self.report_sections.append(section)
        print(f"Added '{title}' to report queue")
    
    def generate_pdf_report(self, filename=None):
        """Generate PDF report"""
        if not self.report_sections:
            return "No analysis sections to include in report."
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_analysis_report_{timestamp}.pdf"
        
        report_path = self.reports_dir / filename
        
        try:
            doc = SimpleDocTemplate(str(report_path), pagesize=A4, leftMargin=1*inch, rightMargin=1*inch)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=20,
                textColor=colors.HexColor('#1f4e79'),
                alignment=1
            )
            
            section_style = ParagraphStyle(
                'SectionTitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.HexColor('#2E86AB')
            )
            
            content = []
            
            # Title
            content.append(Paragraph("ML Data Analysis Report", title_style))
            content.append(Spacer(1, 20))
            
            # Dataset info
            if hasattr(self, 'data_context'):
                dataset_info = f"""
                <b>Dataset Information:</b><br/>
                File: {self.data_context.get('file_path', 'Unknown')}<br/>
                Rows: {self.data_context['shape'][0]:,}<br/>
                Columns: {self.data_context['shape'][1]:,}<br/>
                Analysis Date: {datetime.now().strftime('%B %d, %Y')}
                """
                content.append(Paragraph(dataset_info, styles['Normal']))
                content.append(Spacer(1, 30))
            
            # Add sections
            for i, section in enumerate(self.report_sections, 1):
                content.append(Paragraph(f"{i}. {section['title']}", section_style))
                content.append(Spacer(1, 10))
                
                # Add content
                for line in section['content'].split('\n'):
                    if line.strip():
                        content.append(Paragraph(line, styles['Normal']))
                        content.append(Spacer(1, 5))
                
                # Add tables
                if section['tables']:
                    for table_name, table_df in section['tables'].items():
                        content.append(Paragraph(f"Table: {table_name}", styles['Heading4']))
                        
                        display_df = table_df.head(15) if len(table_df) > 15 else table_df
                        
                        table_data = [display_df.columns.tolist()]
                        for _, row in display_df.iterrows():
                            formatted_row = []
                            for val in row:
                                if isinstance(val, float):
                                    formatted_row.append(f"{val:.2f}" if pd.notna(val) else "N/A")
                                else:
                                    formatted_row.append(str(val) if pd.notna(val) else "N/A")
                            table_data.append(formatted_row)
                        
                        pdf_table = Table(table_data, hAlign='LEFT')
                        pdf_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ]))
                        
                        content.append(pdf_table)
                        content.append(Spacer(1, 15))
                
                content.append(PageBreak())
            
            doc.build(content)
            
            print(f"PDF report generated: {report_path}")
            return f"Report saved to: {report_path}"
            
        except Exception as e:
            return f"Error generating PDF: {str(e)}"
    
    def interactive_session(self):
        """Start interactive session"""
        print("\nStarting Interactive ML Data Analysis Session")
        print("\nCommands:")
        print("  load <file_path> - Load dataset")
        print("  ml_pipeline <date_col> <target_col> - Run complete ML pipeline")
        print("  report - Generate PDF report")
        print("  info - Show dataset info")
        print("  quit - Exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nAgent> ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                elif user_input.startswith('load '):
                    file_path = user_input[5:].strip()
                    self.load_data(file_path)
                
                elif user_input.startswith('ml_pipeline '):
                    params = user_input[12:].strip().split()
                    if len(params) >= 2:
                        date_col, target_col = params[0], params[1]
                        result = self.run_ml_pipeline(date_col, target_col)
                        print(f"\n{result}")
                    else:
                        print("Usage: ml_pipeline <date_column> <target_column>")
                
                elif user_input.lower() == 'report':
                    result = self.generate_pdf_report()
                    print(f"\n{result}")
                
                elif user_input.lower() == 'info':
                    if hasattr(self, 'df'):
                        print(f"\nDataset Info:")
                        print(f"Shape: {self.df.shape}")
                        print(f"Columns: {list(self.df.columns)}")
                        print(f"\nFirst 5 rows:")
                        print(self.df.head())
                    else:
                        print("No data loaded.")
                
                else:
                    print("Unknown command. Type a valid command or 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n{traceback.format_exc()}")

def main():
    """Main function"""
    print("Ollama ML Data Analysis Agent")
    print("=" * 50)
    
    agent = OllamaMLAgent()
    agent.interactive_session()

if __name__ == "__main__":
    main()