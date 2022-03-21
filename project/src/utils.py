from pathlib import Path

def get_proj_root() -> Path:
    return Path(__file__).parent.parent


    