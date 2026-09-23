import argparse
import time
import requests
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import threading
import json

class FlowHandler(FileSystemEventHandler):
    def __init__(self, file_path, endpoint):
        self.file_path = file_path
        self.endpoint = endpoint
        self.last_position = 0
        self.headers = None
        
    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.read_new_lines()
            
    def read_new_lines(self):
        try:
            with open(self.file_path, 'r') as f:
                # Read headers if we haven't
                if self.headers is None:
                    first_line = f.readline().strip()
                    if first_line:
                        self.headers = first_line.split(',')
                        self.last_position = f.tell()
                        
                f.seek(self.last_position)
                lines = f.readlines()
                self.last_position = f.tell()
                
                if not lines:
                    return
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    values = line.split(',')
                    if len(values) == len(self.headers):
                        # Construct JSON payload
                        payload = dict(zip(self.headers, values))
                        
                        # Send to FastAPI
                        try:
                            # Convert numerical values where possible
                            for k, v in payload.items():
                                try:
                                    payload[k] = float(v)
                                except ValueError:
                                    pass
                                    
                            requests.post(self.endpoint, json=payload, timeout=2)
                        except requests.exceptions.RequestException as e:
                            print(f"Error sending flow to API: {e}")
        except Exception as e:
            print(f"File reading error: {e}")

def run_cicflowmeter(interface, output_file):
    print(f"Starting CICFlowMeter on interface {interface}...")
    cmd = ["cicflowmeter", "-i", interface, "-c", output_file]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Sniff and send flows to IDS API")
    parser.add_argument("-i", "--interface", required=True, help="Network interface (e.g., eth0)")
    parser.add_argument("-e", "--endpoint", default="http://localhost:8000/predict", help="API Endpoint")
    parser.add_argument("-o", "--output", default="live_flows.csv", help="Temp CSV file for CICFlowMeter")
    args = parser.parse_args()

    # Create empty file first
    open(args.output, 'w').close()

    # Start cicflowmeter in background
    t = threading.Thread(target=run_cicflowmeter, args=(args.interface, args.output), daemon=True)
    t.start()

    # Watch the CSV file for changes
    event_handler = FlowHandler(f"./{args.output}", args.endpoint)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    print(f"Listening for flows on {args.interface} and forwarding to {args.endpoint}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping...")
    
    observer.join()

if __name__ == "__main__":
    main()
