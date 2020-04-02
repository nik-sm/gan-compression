set -exo pipefail

for latent_dim in 128 64 32; do
  CUDA_VISIBLE_DEVICES=0,2 python train.py --dataset celeba --epochs 50 --run_name ELU_latent_dim_${latent_dim} --n_train -1 --dataset_dir /home/john/projects/cs7180-project/dataset/celeba_preprocessed --latent_dim $latent_dim
done
