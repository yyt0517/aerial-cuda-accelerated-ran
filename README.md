# NVIDIA Aerial™ CUDA-Accelerated RAN

## Overview
NVIDIA Aerial™ CUDA-Accelerated RAN is a part of [NVIDIA AI Aerial™](https://developer.nvidia.com/industries/telecommunications/ai-aerial), a portfolio of accelerated computing platforms, software and tools to build, train, simulate, and deploy AI-native wireless networks.

Documentation for AI Aerial™ can be found [here](https://docs.nvidia.com/aerial/index.html).

The following AI Aerial™ software is available as open source:
- NVIDIA Aerial™ CUDA-Accelerated RAN (this repository)
- [NVIDIA Aerial™ Framework](https://github.com/NVIDIA/aerial-framework)

Updates on new software releases, NVIDIA 6G events and technical training for AI Aerial™ are available via the [NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program).

The **Aerial CUDA-Accelerated RAN** SDK includes:

- **GPU-Accelerated 5G PHY (cuPHY)**: CUDA-based physical layer processing for 5G NR including channel coding (LDPC, Polar), modulation/demodulation, MIMO processing, and channel estimation
- **GPU-Accelerated MAC Scheduler (cuMAC)**: High-performance L2 scheduler acceleration for resource allocation and scheduling
- **Python API (pyAerial)**: Python bindings for AI/ML research and integration with frameworks like TensorFlow and Sionna
- **5G Reference Models (5GModel)**: MATLAB-based 5G waveform generation and test vector creation based on 3GPP specifications
- **Containerized Environment**: Docker-based development and deployment with pre-built containers

### Repository Structure

```
aerial-cuda-accelerated-ran/
├── cuPHY/              # CUDA-accelerated Physical Layer (L1)
├── cuPHY-CP/           # Control Plane and integration components
│   ├── aerial-fh-driver/    # Fronthaul driver for O-RAN interfaces
│   ├── cuphycontroller/     # PHY controller
│   ├── cuphydriver/         # PHY driver
│   ├── cuphyl2adapter/      # L2 adapter
│   ├── ru-emulator/         # Radio Unit emulator
│   ├── testMAC/            # Test MAC implementation
│   └── container/          # Container build scripts and recipes
├── cuMAC/              # CUDA-accelerated L2 Layer
├── cuMAC-CP/           # MAC Control Plane components
├── pyaerial/           # Python API and ML/AI tools
├── 5GModel/            # TV generation for cuPHY and cuBB verification
├── testBenches/        # Test benches and performance measurement tools
├── testVectors/        # Test vectors for validation
└── cubb_scripts/       # Build and automation scripts
```

## Getting Started

### Using Pre-Built Container (Recommended)

```bash
# Clone repository
git clone https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git --recurse-submodules
cd aerial-cuda-accelerated-ran

# Enable git LFS (if needed for large files)
git lfs install
git lfs pull

# Pull the Aerial container from NGC
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb

# Start interactive development container
./cuPHY-CP/container/run_aerial.sh

# Inside container: Build SDK
./testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

- Container versions available at [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/aerial/containers/aerial-cuda-accelerated-ran)

### Local Stage-B/HGraph Experiments

Before running the Stage-B HGraph training, evaluation, or comparison scripts in this workspace, enter the Aerial container and activate the Python virtual environment:

```bash
./cuPHY-CP/container/attach_aerial.sh
source .venv/bin/activate
```

Run the legacy Stage-B training and evaluation commands from the activated shell. The current-policy launcher described below runs on the host and enters the existing CUDA container itself.

Current Stage-B research status (updated 2026-11-25):

- [Project documentation](Doc/README.md)
- [Current progress and evidence](Doc/stageB_current_progress.md)
- [Frozen experimental configuration](Doc/current_stageB_effective_configuration.md)

The registered research policy is PFQ + CTBS + causal deficit-weighted physical local search with a causal dequeue floor of rho=0.90. Five-seed, ten-second comparisons against PFQ are complete for boundary/rate0.8 and uniform/rate1.0. Its [version registry](training/stageb_hgraph/current_algorithm.json) and [launcher](cuMAC/scripts/run_stageB_current_algorithm.sh) identify the boundary entrypoint, which requires the existing CUDA container and research worktree.

The latest research delivery evaluates DA-PLS (M-LWDF-PF weights with exact upper-bound-pruned local search). Across ten seeds and matching ten-second arrival cohorts, Boundary/rate0.8 complete-packet payload improves 67.78% over PFQ, successful mean delay falls 97.84%, and observed expiry is zero. This result does not extend unchanged to seven cells or reduced neighbor information. The [November evidence archive](Doc/evidence/2026-11-25/README.md) includes the final report's [minimal source and result package](Doc/evidence/2026-11-25/minimal_delivery/README.md), numerical verification and an offline replay sample, plus generalization, seven-cell and RSRP studies. This research policy has no trained model and has not replaced the registered default. Full GPU replay still requires the frozen local worktree, container and raw artifacts; build runtime libraries from the supplied sources.

### Further Information

Visit the full documentation at [NVIDIA Docs Hub](https://docs.nvidia.com/aerial/)

## Contribution Guidelines
- Aerial is not accepting contributions at this time.

## Security
- Vulnerability disclosure: [SECURITY.md](SECURITY.md)
- **Do not file public issues for security reports.**

## Support
- **Level**: Maintained
- **How to get help**:
  - File issues on GitHub for bugs and feature requests
  - Join discussions for questions and community support

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Note**: Some dependencies may have different licenses. See [ATTRIBUTION.rst](ATTRIBUTION.rst) for third-party attributions in the source repository.

## Citation

If you use NVIDIA Aerial™ CUDA-Accelerated RAN in your research, please cite:

```bibtex
@software{nvidia_aerial_cuda_accelerated_ran,
  title = {NVIDIA Aerial™ CUDA-Accelerated RAN},
  author = {NVIDIA Corporation},
  year = {2025},
  url = {https://github.com/NVIDIA/aerial-cuda-accelerated-ran}
}
```
