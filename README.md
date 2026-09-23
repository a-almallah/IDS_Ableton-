# IDS_Ableton

This repo will attempt using different methods of doing anomaly detection to find malicous packets on the network and block them.

### Setup

we use `uv` for lightning-fast python package management. here is how to get it running:

1. create the virtual env and activate it:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```
2. install all the dependencies from the requirements file:
   ```bash
   uv pip install -r requirements.txt
   ```

### How to run

you'll find a few different scripts to play around with:

- **train:** train the model on a specific attack (e.g. Bot, DDoS) and mix in normal traffic to avoid overfitting.
  ```bash
  uv run train.py --attack "Bot"
  ```

- **test:** evaluate the model accuracy on unseen test data.
  ```bash
  uv run test.py --attack "Bot"
  ```

- **live demo:** watch the terminal UI feed unseen network traffic through 3 different models live! make sure your terminal is full screen.
  ```bash
  uv run demo.py
  ```

> [!WARNING]  
> For the sake of full transperency AI agents may have been used to review, fix, and debug the code. :)
