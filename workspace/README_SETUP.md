# MVTec AD 2 Setup (Minimal)

## 1) Activate environment
```bash
source .venv/bin/activate
```

## 2) Run local evaluation (example: `can`)
```bash
cd mvtec_ad_evaluation
python evaluate_experiment.py \
  --dataset_base_dir ../data/can \
  --anomaly_maps_dir ../workspace/local_eval/experiments/exp01/anomaly_maps \
  --output_dir ../workspace/local_eval/metrics \
  --evaluated_objects can \
  --pro_integration_limit 0.3
```

## 3) Print metrics
```bash
python print_metrics.py --metrics_folder ../workspace/local_eval/metrics
```

## 4) Prepare submission archive
Put predictions into:

- `workspace/submission_template/anomaly_images/...`
- `workspace/submission_template/anomaly_images_thresholded/...` (optional)

Then validate and create archive:
```bash
cd ../MVTecAD2_public_code_utils
python check_and_prepare_data_for_upload.py ../workspace/submission_template
```

Output: `submission_template.tar.gz`
