set -exo pipefail

[[ -n $1 ]] || { echo MISSING IMG >&2; exit 1; }

for b in normal uniform; do
  for a in alternating simple; do
    for c in 0.01 0.1 1.0; do
      python search.py --generator_checkpoint ./checkpoints/celeba_cropped/gen_ckpt.49.pt \
        --run_name ${a}_${b}_${c} \
        --n_restarts 3 \
        --n_steps 10000 \
        --method $a \
        --image $1 \
        --initialization $b \
        --std $c
    done
  done
done
