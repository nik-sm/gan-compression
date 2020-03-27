set -exo pipefail

[[ -n $1 ]] || { echo MISSING IMG >&2; exit 1; }

python search.py --generator_checkpoint ./checkpoints/celeba_cropped/gen_ckpt.49.pt \
  --run_name SCRAP234_test \
  --n_restarts 1 \
  --n_steps 10000 \
  --method simple \
  --image $1 \
  --initialization normal \
  --std 1.0
