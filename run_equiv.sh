#!/bin/bash
echo "Running equivariance section for all models..."

/vol/workspace/P4T1/miniforge3/envs/deeplife/bin/python scripts/run_eval_all.py \
    --out_dir plots/eval_overnight \
    --sections equivariance \
    --groups ca_only backbone all_atom

echo "Regenerating comprehensive heatmap..."
/vol/workspace/P4T1/miniforge3/envs/deeplife/bin/python scripts/plot_heatmap.py

echo "Copying updated heatmap to IDE artifacts..."
cp plots/eval_overnight/comprehensive_heatmap.png /vol/workspace/home/mmueller/.gemini/antigravity-ide/brain/734ffde1-dfea-467a-b3e6-6ea40501465e/comprehensive_heatmap.png

echo "All done! You can close this window."
# Keep tmux session alive for 60 seconds after finishing so you can read the output
sleep 60
