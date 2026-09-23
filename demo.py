import time
import pandas as pd
import joblib
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich import box
import random
import warnings

warnings.filterwarnings('ignore')

# --- Load Models & Data ---
console = Console()
with console.status("[bold green]Loading Models and Unseen Data..."):
    rf_model = joblib.load('model.pkl')
    try:
        dt_model = joblib.load('dt_model.pkl')
        lr_model = joblib.load('lr_model.pkl')
    except:
        dt_model = rf_model # Fallback if not ready
        lr_model = rf_model
        
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl')
    
    # Load unseen test data (already has 'Label' column)
    test_df = pd.read_csv('test_data_Bot.csv')
    
    # Shuffle the test data to simulate a realistic random feed
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- State Variables ---
models = {
    "Random Forest (Ensemble)": rf_model,
    "Decision Tree (Rules)": dt_model,
    "Logistic Regression (Linear)": lr_model
}

# Metrics storage: model_name -> {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
metrics = {m: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for m in models.keys()}

# Ring buffer for live feed display
MAX_FEED_ROWS = 15
live_feed = []

def generate_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1)
    )
    layout["main"].split_row(
        Layout(name="feed", ratio=2),
        Layout(name="stats", ratio=3)
    )
    layout["stats"].split_column(
        Layout(name="predictions", ratio=1),
        Layout(name="matrices", ratio=2)
    )
    return layout

def update_metrics(model_name, y_true, y_pred):
    is_anomaly_true = y_true.lower() != 'benign'
    is_anomaly_pred = y_pred.lower() != 'benign'
    
    if is_anomaly_true and is_anomaly_pred:
        metrics[model_name]['TP'] += 1
    elif not is_anomaly_true and not is_anomaly_pred:
        metrics[model_name]['TN'] += 1
    elif not is_anomaly_true and is_anomaly_pred:
        metrics[model_name]['FP'] += 1
    elif is_anomaly_true and not is_anomaly_pred:
        metrics[model_name]['FN'] += 1

def build_feed_panel() -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Timestamp", style="cyan")
    table.add_column("Dst Port", style="magenta")
    table.add_column("Protocol", style="magenta")
    table.add_column("Tot Fwd Pkts", justify="right")
    table.add_column("Fwd Pkt Len Mean", justify="right")
    table.add_column("True Label", style="bold white")

    for row in live_feed:
        # Determine color for true label
        label_color = "[bold green]" if row['Label'].lower() == 'benign' else "[bold red]"
        
        table.add_row(
            row['Time'],
            str(int(row['Dst Port'])) if 'Dst Port' in row else "?",
            str(int(row['Protocol'])) if 'Protocol' in row else "?",
            str(int(row['Tot Fwd Pkts'])) if 'Tot Fwd Pkts' in row else "?",
            f"{row['Fwd Pkt Len Mean']:.2f}" if 'Fwd Pkt Len Mean' in row else "?",
            f"{label_color}{row['Label']}[/]"
        )
    return Panel(Align.center(table, vertical="top"), title="[bold blue]Live Network Feed (Unseen Data)[/]", border_style="blue")

def build_predictions_panel(latest_preds) -> Panel:
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Model", style="cyan", ratio=2)
    table.add_column("Prediction", justify="center", ratio=1)
    table.add_column("Status", justify="center", ratio=1)

    for m_name, pred in latest_preds.items():
        if pred.lower() == 'benign':
            pred_styled = f"[bold green]{pred}[/]"
            status = "[bold green]✅ CLEAR[/]"
        else:
            pred_styled = f"[bold red blink]{pred}[/]"
            status = "[bold red]🚨 ALERT[/]"
            
        table.add_row(m_name, pred_styled, status)
        
    return Panel(Align.center(table, vertical="middle"), title="[bold yellow]Live Inference Engine[/]", border_style="yellow")

def build_matrices_panel() -> Panel:
    # We will create a layout of matrices
    matrix_table = Table(box=None, expand=True, show_header=False)
    matrix_table.add_column("M1", ratio=1)
    matrix_table.add_column("M2", ratio=1)
    matrix_table.add_column("M3", ratio=1)
    
    panels = []
    for m_name, mets in metrics.items():
        total = sum(mets.values())
        acc = (mets['TP'] + mets['TN']) / total if total > 0 else 0
        
        t = Table(title=f"{m_name}\n[white]Accuracy: {acc*100:.1f}%[/]", box=box.MINIMAL_DOUBLE_HEAD)
        t.add_column(" ", style="dim")
        t.add_column("Pred: Norm")
        t.add_column("Pred: Atk")
        t.add_row("True: Norm", f"[green]{mets['TN']}[/]", f"[red]{mets['FP']}[/]")
        t.add_row("True: Atk", f"[yellow]{mets['FN']}[/]", f"[red bold]{mets['TP']}[/]")
        
        panels.append(Panel(t, border_style="green" if acc > 0.95 else "yellow"))
        
    matrix_table.add_row(*panels)
    
    return Panel(Align.center(matrix_table, vertical="top"), title="[bold magenta]Live Confusion Matrices & Accuracy[/]", border_style="magenta")

def main():
    layout = generate_layout()
    layout["header"].update(Panel(Align.center(Text("🛡️ ENTERPRISE INTRUSION DETECTION SYSTEM (LIVE DEMO) 🛡️", style="bold white on blue")), style="blue"))

    # We iterate over the unseen test set
    with Live(layout, refresh_per_second=10, screen=True) as live:
        for idx, row in test_df.iterrows():
            # 1. Prepare Data
            y_true = row['Label']
            X_raw = pd.DataFrame([row.drop('Label')])
            
            # Ensure columns match model
            for col in model_features:
                if col not in X_raw.columns:
                    X_raw[col] = 0.0
            X_live = X_raw[model_features]
            X_live = X_live.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
            
            # Scale
            X_scaled = scaler.transform(X_live)
            
            # 2. Run Inference
            latest_preds = {}
            for m_name, model in models.items():
                pred = model.predict(X_scaled)[0]
                latest_preds[m_name] = pred
                update_metrics(m_name, y_true, pred)
                
            # 3. Update Feed
            feed_row = row.to_dict()
            feed_row['Time'] = time.strftime("%H:%M:%S")
            live_feed.insert(0, feed_row)
            if len(live_feed) > MAX_FEED_ROWS:
                live_feed.pop()
                
            # 4. Render
            layout["feed"].update(build_feed_panel())
            layout["predictions"].update(build_predictions_panel(latest_preds))
            layout["matrices"].update(build_matrices_panel())
            
            # Simulate real-time delay (randomized for realism)
            time.sleep(random.uniform(0.05, 0.15))

if __name__ == "__main__":
    main()
