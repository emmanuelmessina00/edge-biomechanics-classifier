import kagglehub
import shutil
from pathlib import Path

# Download latest version
cache_path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")

# Define local data directory
local_data_dir = Path(__file__).parent.parent / "data"
local_data_dir.mkdir(parents=True, exist_ok=True)

# Copy dataset to local directory
for item in Path(cache_path).iterdir():
    dest = local_data_dir / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)

print("Path to dataset files:", local_data_dir)