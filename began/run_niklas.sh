set -exo pipefail

for latent_dim in 256 512 1024; do
  CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --epochs 50 --run_name ELU_latent_dim_${latent_dim} --n_train -1 --dataset_dir ./data/celeba-preprocessed-v2/ --latent_dim $latent_dim
done
