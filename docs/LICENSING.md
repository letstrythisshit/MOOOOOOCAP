# Licensing & Attribution

## Project License

This project is licensed under the **Apache License 2.0**.

You may:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Sublicense
- ✅ Private use

You must:
- Include license and copyright notice
- State changes
- Include NOTICE file if provided

You cannot:
- Hold liable
- Use trademarks

## Component Licenses

### Models

All models are from **OpenMMLab** with **Apache 2.0** license:

#### RTMDet (Detection)
- **Source**: https://github.com/open-mmlab/mmdetection
- **License**: Apache 2.0
- **Paper**: https://arxiv.org/abs/2212.07784
- **Models Used**:
  - rtmdet-nano
  - rtmdet-s
  - rtmdet-m

#### RTMPose (Pose Estimation)
- **Source**: https://github.com/open-mmlab/mmpose
- **License**: Apache 2.0
- **Paper**: https://arxiv.org/abs/2303.07399
- **Models Used**:
  - rtmpose-t
  - rtmpose-s
  - rtmpose-m
  - rtmpose-l
  - rtmpose-x
  - rtmpose-m-wholebody
  - rtmpose-l-wholebody
  - rtmpose-x-wholebody

### Algorithms

#### ByteTrack (Multi-Object Tracking)
- **Paper**: https://arxiv.org/abs/2110.06864
- **Original Implementation**: https://github.com/ifzhang/ByteTrack (MIT)
- **This Implementation**: Clean-room reimplementation based on paper
- **License**: Compatible with Apache 2.0

#### One-Euro Filter (Temporal Smoothing)
- **Paper**: CHI 2012 - "€1 Filter: A Simple Speed-based Low-pass Filter"
- **License**: Public Domain
- **Implementation**: Based on published algorithm

### Dependencies

| Package | License | Commercial Use | Notes |
|---------|---------|----------------|-------|
| **PyTorch** | BSD-3-Clause | ✅ Yes | Deep learning framework |
| **ONNX Runtime** | MIT | ✅ Yes | Cross-platform inference |
| **OpenCV** | Apache 2.0 | ✅ Yes | Computer vision |
| **NumPy** | BSD-3-Clause | ✅ Yes | Numerical computing |
| **SciPy** | BSD-3-Clause | ✅ Yes | Scientific computing |
| **PySide6** | LGPL-3.0 | ✅ Yes* | Qt for Python |
| **lap** | BSD-2-Clause | ✅ Yes | Linear assignment |
| **rich** | MIT | ✅ Yes | Terminal output |
| **tqdm** | MPL-2.0/MIT | ✅ Yes | Progress bars |
| **PyYAML** | MIT | ✅ Yes | YAML parser |

\* **PySide6 Note**: LGPL allows commercial use as long as you:
1. Dynamically link to PySide6 (don't statically compile it in)
2. Allow users to replace the PySide6 library
3. This is standard practice with pip-installed packages

## Explicitly Not Used

### MediaPipe
- **Reason**: Avoided per user requirements
- **Alternative**: RTMPose + custom hand analysis

### GPL-Licensed Code
- **Reason**: Incompatible with commercial use requirements
- **Alternatives**: Apache 2.0 or MIT licensed equivalents used throughout

## Commercial Use Compliance

This project is **fully commercial-use friendly**:

✅ **All models**: Apache 2.0 from OpenMMLab
✅ **All algorithms**: Apache 2.0 compatible implementations
✅ **All dependencies**: Permissive licenses (MIT, BSD, Apache 2.0, LGPL*)

No GPL or restrictive licenses in the entire stack!

## Citation Requirements

While not legally required, we encourage citing the original works:

### RTMPose
```bibtex
@misc{jiang2023rtmpose,
  title={RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  author={Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  journal={arXiv preprint arXiv:2303.07399},
  year={2023}
}
```

### ByteTrack
```bibtex
@article{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

### RTMDet
```bibtex
@misc{lyu2022rtmdet,
  title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
  author={Lyu, Chengqi and Zhang, Wenwei and Huang, Haian and Zhou, Yue and Wang, Yudong and Liu, Yanyi and Zhang, Shilong and Chen, Kai},
  journal={arXiv preprint arXiv:2212.07784},
  year={2022}
}
```

## Attribution

### OpenMMLab
We thank the OpenMMLab team for their excellent work on:
- MMDetection (detection framework)
- MMPose (pose estimation framework)
- MMCV (foundational library)
- Providing pre-trained models with permissive licenses

### ByteTrack Authors
We thank the ByteTrack authors for their innovative tracking algorithm.

### One-Euro Filter Authors
We thank the authors of the One-Euro filter for their elegant smoothing algorithm.

## License Compatibility Matrix

| This Project | Can Use | License Compatibility |
|--------------|---------|----------------------|
| Apache 2.0 | Apache 2.0 | ✅ Full compatibility |
| Apache 2.0 | MIT | ✅ Full compatibility |
| Apache 2.0 | BSD (2/3-clause) | ✅ Full compatibility |
| Apache 2.0 | LGPL | ✅ Yes (dynamic linking) |
| Apache 2.0 | GPL | ❌ Incompatible (not used) |
| Apache 2.0 | AGPL | ❌ Incompatible (not used) |

## Questions?

If you have questions about licensing:

1. **For this project**: Open an issue in the repository
2. **For OpenMMLab models**: See https://github.com/open-mmlab
3. **For commercial licensing**: All components are already commercial-friendly!

## Legal Disclaimer

This licensing document is provided for informational purposes. For legal advice specific to your use case, consult with a legal professional.
