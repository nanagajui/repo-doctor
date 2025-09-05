# WorldGen Test Repository

This is a test repository that simulates the WorldGen project dependencies.

## Requirements

- Python 3.8+
- CUDA 11.8+ (for flash-attn compatibility)
- PyTorch with CUDA support
- At least 16GB VRAM for optimal performance

## Installation

```bash
pip install -r requirements.txt
```

## Known Issues

- flash-attn requires specific CUDA versions
- May not work on newer GPU architectures (sm_120+)
- Requires compatible PyTorch CUDA version
