from src.config import load_config
from src.api_client import EtherscanClient

cfg = load_config()

with EtherscanClient(cfg.api) as client:
    eth_chain = cfg.chains["ethereum"]
    latest_block = client.get_latest_block_number(eth_chain)
    print("Latest Ethereum block:", latest_block)