import pandas as pd
import numpy as np
import logging
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ====================== PIPELINE FUNCTIONS ======================
# Define expected schemas for each dataset
SCHEMAS = {
    'event_logs': {
        'required': ['user_id', 'event_type', 'event_time', 'product_id', 'amount'],
        'dtypes': {
            'user_id': 'str',
            'event_type': 'str',
            'event_time': 'datetime64[ns]',
            'product_id': 'str',
            'amount': 'float'
        }
    },
    'trend_report': {
        'required': ['week', 'avg_users', 'sales_growth_rate'],
        'dtypes': {
            'week': 'datetime64[ns]',
            'avg_users': 'int',
            'sales_growth_rate': 'float'
        }
    },
    'marketing_summary': {
        'required': ['date', 'users_active', 'total_sales', 'new_customers'],
        'dtypes': {
            'date': 'datetime64[ns]',
            'users_active': 'int',
            'total_sales': 'float',
            'new_customers': 'int'
        }
    }
}

# Fallback defaults for critical columns
FALLBACK_VALUES = {
    'amount': 0.0,
    'total_sales': 0.0,
    'avg_users': 0,
    'users_active': 0,
    'new_customers': 0,
    'sales_growth_rate': 0.0
}

def validate_and_repair(df, dataset_name):
    """Validate schema and repair missing/corrupted columns"""
    schema = SCHEMAS[dataset_name]
    validation_report = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'original_columns': list(df.columns),
        'missing_columns': [],
        'repaired_columns': [],
        'type_issues': {},
        'critical_failure': False
    }
    
    # 1. Check for missing columns
    missing_cols = [col for col in schema['required'] if col not in df.columns]
    validation_report['missing_columns'] = missing_cols
    
    if missing_cols:
        logging.warning(f"Missing columns in {dataset_name}: {missing_cols}")
        for col in missing_cols:
            if col in FALLBACK_VALUES:
                df[col] = FALLBACK_VALUES[col]
                logging.info(f"Created fallback column: {col} with default values")
                validation_report['repaired_columns'].append(col)
            else:
                logging.error(f"Unrecoverable missing column: {col}")
                validation_report['critical_failure'] = True
    
    # 2. Validate data types
    for col, expected_type in schema['dtypes'].items():
        if col not in df.columns:
            continue
            
        current_type = str(df[col].dtype)
        type_match = current_type == expected_type
        
        # Special handling for datetime columns
        if expected_type == 'datetime64[ns]' and not pd.api.types.is_datetime64_any_dtype(df[col]):
            type_match = False
            
        if not type_match:
            validation_report['type_issues'][col] = {
                'expected': expected_type,
                'actual': current_type
            }
            logging.warning(f"Type mismatch in {dataset_name}.{col}: {current_type} vs {expected_type}")
            
            try:
                # Handle datetime conversion
                if expected_type == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Check if conversion created too many NaTs
                    if df[col].isna().mean() > 0.5:
                        logging.error(f"Over 50% date conversion failures for {col}")
                        raise ValueError("Excessive date conversion errors")
                else:
                    df[col] = df[col].astype(expected_type)
                    
                logging.info(f"Successfully converted {col} to {expected_type}")
                validation_report['type_issues'][col]['repaired'] = True
                
            except (TypeError, ValueError) as e:
                logging.error(f"Type conversion failed for {col}: {str(e)}")
                if col in FALLBACK_VALUES:
                    df[col] = FALLBACK_VALUES[col]
                    logging.warning(f"Reset corrupted column: {col} to default values")
                    validation_report['type_issues'][col]['fallback_used'] = True
                else:
                    validation_report['type_issues'][col]['repaired'] = False
    
    return df, validation_report

def clean_event_logs(df):
    """Cleaning logic for event logs"""
    # Remove undefined columns
    df = df[SCHEMAS['event_logs']['required']].copy()
    
    # Handle missing values
    df['event_type'] = df['event_type'].fillna('unknown')
    df['user_id'] = df['user_id'].fillna('UNK').astype(str)
    
    # Product/amount handling: Only relevant for orders
    order_mask = df['event_type'].str.contains('order|purchase', case=False, na=False)
    df.loc[order_mask, 'product_id'] = df.loc[order_mask, 'product_id'].fillna('PROD_UNK')
    df.loc[order_mask, 'amount'] = df.loc[order_mask, 'amount'].fillna(0)
    df.loc[~order_mask, 'product_id'] = df.loc[~order_mask, 'product_id'].fillna('N/A')
    df.loc[~order_mask, 'amount'] = 0
    
    return df

def clean_trend_report(df):
    """Cleaning logic for trend report"""
    # Keep only relevant columns
    df = df[SCHEMAS['trend_report']['required']].copy()
    
    # Handle dates
    if not pd.api.types.is_datetime64_any_dtype(df['week']):
        df['week'] = pd.to_datetime(df['week'], errors='coerce')
    
    # Sort before forward-filling
    df.sort_values('week', inplace=True)
    
    # Fill numeric metrics
    df['avg_users'] = df['avg_users'].fillna(0)
    df['sales_growth_rate'] = df['sales_growth_rate'].ffill().fillna(0)
    
    return df

def clean_marketing_summary(df):
    """Cleaning logic for marketing summary"""
    # Keep relevant columns
    df = df[SCHEMAS['marketing_summary']['required']].copy()
    
    # Date conversion
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fill sequential data
    df.sort_values('date', inplace=True)
    for col in ['users_active', 'total_sales', 'new_customers']:
        df[col] = df[col].ffill().fillna(0)
    
    return df

def process_dataset(file_path, dataset_name):
    """Load, validate, and clean dataset with error handling"""
    logging.info(f"Starting processing: {dataset_name}")
    validation_result = None
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} records from {dataset_name}")
        
        # Schema validation and repair
        df, validation_result = validate_and_repair(df, dataset_name)
        
        if validation_result.get('critical_failure', False):
            logging.error(f"Critical validation failure for {dataset_name}. Aborting processing.")
            return None, validation_result
        
        # Dataset-specific cleaning
        if dataset_name == 'event_logs':
            df = clean_event_logs(df)
        elif dataset_name == 'trend_report':
            df = clean_trend_report(df)
        elif dataset_name == 'marketing_summary':
            df = clean_marketing_summary(df)
            
        logging.info(f"Successfully processed {dataset_name}")
        return df, validation_result
        
    except Exception as e:
        logging.exception(f"Fatal error processing {dataset_name}")
        return None, validation_result

# ====================== GUI APPLICATION ======================
class PipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FinMark Data Pipeline")
        self.root.geometry("1000x700")
        
        # Configure logging to text widget
        self.log_box = tk.Text(root, height=15, state='disabled')
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Setup logging handler
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.config(state='normal')
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
                self.text_widget.config(state='disabled')
        
        text_handler = TextHandler(self.log_box)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        # File selection frame
        file_frame = ttk.LabelFrame(root, text="Dataset Selection")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.file_vars = {
            'event_logs': tk.StringVar(),
            'trend_report': tk.StringVar(),
            'marketing_summary': tk.StringVar()
        }
        
        for i, name in enumerate(self.file_vars):
            ttk.Label(file_frame, text=f"{name.replace('_', ' ').title()}:").grid(row=i, column=0, padx=5, pady=2, sticky='w')
            ttk.Entry(file_frame, textvariable=self.file_vars[name], width=50).grid(row=i, column=1, padx=5, pady=2)
            ttk.Button(file_frame, text="Browse", 
                      command=lambda n=name: self.browse_file(n)).grid(row=i, column=2, padx=5, pady=2)
        
        # Control buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Run Pipeline", command=self.run_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_frame = ttk.LabelFrame(root, text="Processing Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.tree = ttk.Treeview(self.results_frame, columns=('Dataset', 'Status', 'Records', 'Details'), show='headings')
        self.tree.heading('Dataset', text='Dataset')
        self.tree.heading('Status', text='Status')
        self.tree.heading('Records', text='Records')
        self.tree.heading('Details', text='Details')
        self.tree.column('Dataset', width=150)
        self.tree.column('Status', width=80)
        self.tree.column('Records', width=80)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization area
        self.viz_frame = ttk.LabelFrame(root, text="Data Visualization")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
    def browse_file(self, dataset_name):
        filepath = filedialog.askopenfilename(
            title=f"Select {dataset_name} file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            self.file_vars[dataset_name].set(filepath)
    
    def clear_logs(self):
        self.log_box.config(state='normal')
        self.log_box.delete(1.0, tk.END)
        self.log_box.config(state='disabled')
    
    def run_pipeline(self):
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Prepare datasets
        datasets = {}
        for name, var in self.file_vars.items():
            path = var.get()
            if path:
                datasets[name] = path
            else:
                logging.warning(f"No file selected for {name}")
        
        if not datasets:
            messagebox.showwarning("Input Error", "Please select at least one dataset file")
            return
        
        # Run in separate thread to keep GUI responsive
        threading.Thread(target=self._execute_pipeline, args=(datasets,), daemon=True).start()
    
    def _execute_pipeline(self, datasets):
        results = {}
        validation_reports = {}
        
        logging.info("=== Starting Pipeline Execution ===")
        
        for name, path in datasets.items():
            logging.info(f"Processing {name}...")
            cleaned, report = process_dataset(path, name)
            validation_reports[name] = report
            
            if cleaned is not None:
                results[name] = cleaned
                cleaned.to_csv(f'cleaned_{name}.csv', index=False)
                status = "✅ Success"
                details = f"Saved to cleaned_{name}.csv"
                self._update_results(name, status, len(cleaned), details)
                self._visualize_data(name, cleaned)
            else:
                status = "❌ Failed"
                details = "Check logs for errors"
                self._update_results(name, status, 0, details)
        
        logging.info(f"Pipeline completed. Successful datasets: {len(results)}/{len(datasets)}")
    
    def _update_results(self, dataset, status, records, details):
        self.root.after(0, lambda: self.tree.insert(
            '', 'end', 
            values=(dataset, status, records, details)
        ))
    
    def _visualize_data(self, dataset_name, df):
        self.root.after(0, self._clear_viz_frame)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        try:
            if dataset_name == 'marketing_summary' and 'date' in df.columns and 'total_sales' in df.columns:
                df = df.sort_values('date')
                ax.plot(df['date'], df['total_sales'], label='Total Sales')
                ax.set_title('Sales Over Time')
                ax.legend()
            
            elif dataset_name == 'trend_report' and 'week' in df.columns and 'avg_users' in df.columns:
                df = df.sort_values('week')
                ax.bar(df['week'], df['avg_users'], label='Active Users')
                ax.set_title('Weekly Active Users')
            
            elif dataset_name == 'event_logs' and 'event_type' in df.columns:
                # Sample: Event count by type
                event_counts = df['event_type'].value_counts().head(5)  # Show top 5
                ax.pie(event_counts, labels=event_counts.index, autopct='%1.1f%%')
                ax.set_title('Event Type Distribution')
            
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        except Exception as e:
            logging.error(f"Visualization error: {str(e)}")
    
    def _clear_viz_frame(self):
        for widget in self.viz_frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineApp(root)
    root.mainloop()