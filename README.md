# DP-LET

Official implementation for the paper:  [DP-LET: An Efficient Spatio-Temporal Network Traffic Prediction Framework](https://arxiv.org/abs/2504.03792)

---

## Overall Architecture

Refer to the figure below (`overall.jpg`) for an overview of the DP-LET framework.

---

## Usage

1. Create and activate a virtual environment
```bash
conda create -n DP-LET python=3.8
conda activate DP-LET
```

2. Install dependencies
```bash
cd framework
pip install -r requirements.txt
```

3. Prepare the dataset

Download the **Telecom Italia Milan Internet Traffic Dataset** from:  
[https://doi.org/10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)
You can choose any desired time duration and number of cells, and save the file as:
```
MILAN/Milan_Internet_[cell_number]_10min.csv
```
For example:
```
MILAN/Milan_Internet_100_10min.csv
```

4. Run the training script
```bash
python run.py
```

