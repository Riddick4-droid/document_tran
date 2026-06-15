import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import kagglehub

from src.logger import get_logger
from src.exceptions import ProjectException

logger = get_logger(__name__)

def download_datasets(config: Dict[str,Any], force:bool=False)->None:
    """
    Download datasets specified in the config (idempotent by default).

    Args:
        config: Full project configuration (parsed YAML).
        force: If True, re‑download even if the dataset directory already exists.
    """
    data_root = Path(config["data"]["root"])
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = config["data"]["datasets"]
    for purpose, ds_cfg in datasets.items():
        ds_name = ds_cfg["name"]
        local_dir = data_root / ds_cfg["local_dir"]
        _download_one(ds_name, local_dir,purpose, force)

def _download_one(ds_name: str, local_dir:Path, purpose: str, force:bool)->None:
    if local_dir.exists() and any(local_dir.iterdir()) and not force:
        logger.info(f"{purpose} dataset already present at {local_dir}")
        return
    logger.info(f"Downloading {purpose} dataset '{ds_name}' from kagglehub....")
    try:
        downloaded_path = kagglehub.dataset_download(ds_name)
        logger.info(f"Downloaded to {downloaded_path}")
        if local_dir.exists():
            shutil.rmtree(local_dir)
        shutil.copytree(downloaded_path, local_dir)
        logger.info(f"copied to {local_dir}")
    except Exception as e:
        raise ProjectException(
            f"failed to download dataset '{ds_name}' for {purpose}",
            error_detail=e
        )

#standalone
if __name__ == "__main__":
    from src.utils import load_config

    config = load_config(config_path="./config/configs.yaml")

    download_datasets(config=config)