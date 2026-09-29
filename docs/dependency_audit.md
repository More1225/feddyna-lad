# Dependencies and environment

requirements.txt lists dependencies needed by the selected public implementation. Versions are unpinned because no complete, validated historical environment is available. This file is an import-based dependency declaration, not a lockfile or a tested compatibility matrix.

| Import/use | Distribution | Public consumer |
| --- | --- | --- |
| torch | torch | Models, trainers and federated computation |
| numpy | numpy | Arrays, preprocessing and statistics |
| pandas | pandas | Client CSVs and result tables |
| sklearn | scikit-learn | Precision/recall/F1, average precision and AUROC |
| matplotlib | matplotlib | Training/diagnostic plotting |
| Excel writer engine | openpyxl | utils/save_results.py |
| tqdm | tqdm | Preparation progress |
| gensim | gensim | Word2Vec in selected preparation scripts |
| regex | regex | Bundled BGL/TB Drain parsers |

Drain is bundled locally; it is not an additional pip dependency. Standard-library and project imports are not listed as packages. PyYAML is not required by the final scope because the alternate YAML merge tools are excluded. No historical standalone-baseline or manuscript tooling dependencies are included.

## Environment limitations

- Source annotations require Python 3.10 or newer. Actual compatibility also depends on available package wheels and library APIs; no supported-version matrix was established by static checks.
- Exact versions of all nine listed packages remain unverified. No version bounds or historical pins were invented.
- PyTorch CPU/CUDA selection and the associated driver/toolkit environment must be chosen for the target machine.
- NumPy, pandas, gensim and transitive dependencies must form a compatible environment; dependency resolution alone is not a training/evaluation validation.
- Static compilation and import inspection do not execute preprocessing, training or evaluation. Release preparation did not install packages or certify numerical behavior on a new platform.

After validating an environment for your current experiment, record Python/package versions, CUDA details, source revision and input hashes with the run. This improves future traceability without claiming equivalence to a missing historical environment.
