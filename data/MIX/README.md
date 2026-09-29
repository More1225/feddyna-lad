# Mixed-system federated inputs

Use mix_federated_generator.py to combine selected client CSV triplets into a dense unified NPZ. Follow the [current dataset workflow](../../docs/canonical_reproduction_path.md#assemble-a-mixed-dataset).

| Source | Filename pattern | Sequence column |
| --- | --- | --- |
| HDFS | client_<id>_<split>.csv | event_ids |
| BGL | client<id>_<split>.csv | sequence |
| Thunderbird shuffle | client<id>_<split>_shuffle.csv | sequence |
| Existing sequential Thunderbird inputs | client<id>_<split>_sequential.csv | sequence |

Each client requires train, val and test files with a label column. The public Thunderbird producer uses shuffle; sequential filename support remains for compatibility. Put only the desired client triplets in the input directory. Source-specific IDs are retained in metadata while the output uses unified client indices.

```console
python data/MIX/mix_federated_generator.py --input_dir data/MIX/current --output_dir data/MIX/current --output_name mixed_fed_unified
```

Run this command from the repository root. The script fits shared Word2Vec embeddings using training sequences, preserves input splits and writes NPZ arrays, vocabulary/embeddings and usage information. It does not select the historical paper cohort automatically. No generated data or historical client files are included in the release.
