# English Release

Date: 2026-10-27

This is the English edition of the fixed `AR-MultiModel-97k.zip` release, including its SSH transport and node diagnostics fixes. The source archive is preserved separately.

- Source archive SHA256: `535293b159d6b24a1acfe65bd6f44d921c50b53611f671b6bf57dc4305c1be3c`
- English archive: `AR-MultiModel-97k-English.zip`
- Internal project directory: `AR-MultiModel-97k/`
- Entry points, model keys, configuration paths, and training commands remain unchanged.

Documentation, comments, CLI help, runtime messages, and message assertions are translated into English. Multilingual regression inputs are represented with Unicode escapes to preserve their original values and test coverage. Historical filenames, including `docs/legacy/README_zh.md`, are retained; their documentation content is now English.

Training data, configuration values, model profiles, reward rules, and the prompts actually supplied to models are unchanged. Jinja template changes affect comments only. This is a language edition of the fixed release; no performance optimizations are included.

## Local validation

The full existing CPU suite passed: **540 passed, 94 subtests passed in 42.37 seconds**, with no failures or skipped tests. The log is in [verification/pytest_english_cpu.txt](verification/pytest_english_cpu.txt).

```bash
AR_ANNOTATION_SOURCE=/Users/zwj/Desktop/CVPR/AR/AR-action-training-32gpu-v3/tools/annotation \
QWEN3VL_PROCESSOR_PATH=/private/tmp/trsi-qwen3-official-processor \
/private/tmp/trsi-qwen3-venv/bin/python -B -m pytest -q -p no:cacheprovider tests
```

The check also verifies Python syntax and structure, Bash syntax, unchanged rendered prompt content, unchanged data/configuration files and key training contracts, and absence of untranslated Chinese text. Package file hashes and the ZIP integrity are checked after packaging.

These are local CPU checks. This English release has not been used to connect to the training cluster or start a GPU job during its preparation. Historical validation records remain in [VERIFICATION.md](VERIFICATION.md).
