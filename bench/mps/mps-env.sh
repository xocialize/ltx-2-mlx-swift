#!/bin/zsh
# AB-P-0004 step 0 — PyTorch-MPS env for the vendor arm.
# Uses the venv's OWN pip, not `uv pip`: the repo's [tool.uv.index] points every platform at
# download.pytorch.org/whl/cu132, which has no macOS wheels. pip ignores uv config entirely.
set -x
cd /Volumes/Satechi/Development/LTX-2 || exit 1
echo "=== 1. provision python 3.12 venv ==="
uv venv .venv-mps --python 3.12 --seed || exit 10
P=.venv-mps/bin/python
$P -V || exit 11
echo "=== 2. torch + torchaudio from DEFAULT PyPI (macOS arm64 = MPS) ==="
$P -m pip install -q --upgrade pip || exit 12
$P -m pip install torch torchaudio || exit 13
$P -c "import torch;print('TORCH', torch.__version__, 'MPS', torch.backends.mps.is_available(), torch.backends.mps.is_built())" || exit 14
echo "=== 3. first-party packages (editable) ==="
$P -m pip install -e packages/ltx-core || exit 15
$P -m pip install -e packages/ltx-pipelines || exit 16
echo "=== 4. verify the import surface the comparison needs ==="
$P -c "
import torch, ltx_core, ltx_pipelines
from ltx_core.devices import get_preferred_device
print('DEVICE', get_preferred_device())
try:
    import mps_sdpa; print('MPS_SDPA present')
except Exception as e:
    print('MPS_SDPA MISSING:', e)
from ltx_pipelines.distilled import DistilledPipeline
print('DistilledPipeline importable')
" || exit 17
echo "=== STEP0 OK ==="
