# MVTec AD 2 Setup

## Extract and place data like this
```bash
data/can/can/test_public...
```

## Activate environment
```bash
source .venv/bin/activate
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Train baseline (AE/VAE, 512 patches)
```bash
python workspace/local_eval/train_ae_vae.py \
  --dataset_base_dir data/can \
  --object_name can \
  --model_type ae \
  --patch_size 512 \
  --epochs 5 \
  --batch_size 4 \
  --validation_split validation \
  --output_dir workspace/local_eval/experiments/exp01/checkpoints
```

Notes:
- `patch_size` must be divisible by 16.
- `best.pt` is selected by loss on `<dataset>/<object>/<validation_split>/good`.
- `run_config.json` and `train_history.json` are saved with checkpoints.

## Infer anomaly maps
```bash
python workspace/local_eval/infer_ae_vae.py \
  --dataset_base_dir data/can \
  --object_name can \
  --checkpoint workspace/local_eval/experiments/exp01/checkpoints/best.pt \
  --anomaly_maps_dir workspace/local_eval/experiments/exp01/anomaly_maps
```

## Run local evaluation
```bash
cd mvtec_ad_evaluation
python evaluate_experiment.py \
  --dataset_base_dir ../data/can \
  --anomaly_maps_dir ../workspace/local_eval/experiments/exp01/anomaly_maps \
  --output_dir ../workspace/local_eval/metrics \
  --evaluated_objects can \
  --pro_integration_limit 0.3
```

## Print metrics
```bash
python print_metrics.py --metrics_folder ../workspace/local_eval/metrics
```

## Compare multiple methods (AE, VAE, ...)
Run a standardized benchmark across methods and seeds:
```bash
python workspace/local_eval/benchmark_methods.py \
  --dataset_base_dir data/can \
  --object_name can \
  --methods ae vae \
  --seeds 42 43 44 \
  --epochs 5 \
  --batch_size 4 \
  --patch_size 512
```

Results are written to:
- `workspace/local_eval/benchmark_runs/<object>/summary/per_run.csv`
- `workspace/local_eval/benchmark_runs/<object>/summary/by_method.csv`

Add custom methods via JSON config:
```bash
python workspace/local_eval/benchmark_methods.py \
  --dataset_base_dir data/can \
  --object_name can \
  --methods ae vae my_sota_method \
  --custom_methods_config workspace/local_eval/custom_methods.example.json
```

Supported placeholders in custom command templates:
- `{dataset_base_dir}`, `{object_name}`, `{patch_size}`, `{batch_size}`
- `{epochs}`, `{lr}`, `{latent_channels}`, `{beta}`
- `{validation_split}`, `{checkpoint_dir}`, `{checkpoint}`, `{anomaly_maps_dir}`
- `{metrics_dir}`, `{seed}`, `{num_workers}`, `{cache_size}`

## Prepare submission archive
Put predictions into:

- `workspace/submission_template/anomaly_images/...`
- `workspace/submission_template/anomaly_images_thresholded/...` (optional)

Then validate and create archive:
```bash
cd ../MVTecAD2_public_code_utils
python check_and_prepare_data_for_upload.py ../workspace/submission_template
```

Output: `submission_template.tar.gz`
