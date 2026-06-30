# Pipeline Status

| Feature | Dev Status | QA Status | Notes |
|---------|-----------|-----------|-------|
| RVC v2 Migration (core) | ✅ Complete | ✅ Passed | Full architecture rewrite — models, engine, training |
| API: Train endpoints | ✅ Complete | ✅ Passed | POST /v1/audio/train + GET /v1/audio/training/status |
| CLI: train mode | ✅ Complete | ✅ Passed | --mode train with --audio-dir, --epochs, --pitch-adjustment (18 tests) |
| Voice Watcher rewrite | ✅ Complete | ✅ Passed | Watches voices/ dirs for config.json+v1.pth instead of raw audio (20 tests) |
| README update | ✅ Complete | ✅ Passed | Full rewrite covering RVC v2, training workflow, CLI, API |
| Device detection (XPU/MPS/CUDA) | ✅ Complete | ✅ Passed | Unified detect_device() + empty_cache() in rvc_assets.py; all 38 tests pass |
