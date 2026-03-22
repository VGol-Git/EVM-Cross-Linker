## Setup

### Install required tools

```
pip install -r requirements.txt
```

### Enable virtual environment as a kernel for laptops

```
python -m ipykernel install --user --name cross-chain-wallet-profiling --display-name "Python (cross-chain-wallet-profiling)"
```

### Check that everything is ok

```
python -c "import requests, pandas, numpy, matplotlib; print('ok')"
```

## Set up a key for the explorer API

### You need an Etherscan account https://etherscan.io/

```

$env:ETHERSCAN_API_KEY="YOUR_KEY"
$env:PROJECT_ROOT="C:\path\to\project"

echo $env:ETHERSCAN_API_KEY
echo $env:PROJECT_ROOT
```
