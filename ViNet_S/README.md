# ViNet-S outplay finetuning

Finetune ViNet-S on the sports outplay split. The local checkout is the source of truth. Commit here, then push to the `vm` remote. Do not edit tracked files on the VM.

- VM: `173.255.234.238`
- Repo on the VM: `/home/nitesh_sawadia/vinet_s/vinet-video-saliency`
- Virtualenv: `/home/nitesh_sawadia/vinet_s/venv`
- Checkpoint: `/home/nitesh_sawadia/vinet_s/weights/vinet_s_dhf1k.pt`
- Dataset mount: `/home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split`
- MLflow: `http://104.105.26.182:5000`, experiment `sports-saliency-ViNet-S`
- TensorBoard logdir: `/home/nitesh_sawadia/vinet_s/tensorboard`

The launch script trains for 40 epochs with 32-frame clips, batch size 16, 8 data workers, learning rate `1e-4`, and a checkpoint every epoch. Sample clips are written with `add_video` at 8 fps, one video for the frames and one for the saliency overlay.

## 1. Push the current commit

Run locally:

```bash
cd /Users/niteshs/amagi-FanPulse/vinet-video-saliency

git status
git log -1 --oneline
git push vm master
```

Leave the modified `fold_lists/*` files and `checkpoints.tar.gz` unstaged. They are not part of this training setup.

To publish the same commit on GitHub:

```bash
git push origin master
```

## 2. Check the VM checkout

```bash
ssh 173.255.234.238
```

```bash
cd /home/nitesh_sawadia/vinet_s/vinet-video-saliency

git status
git log -1 --oneline
git rev-parse HEAD
```

`git rev-parse HEAD` on the VM must match the local `HEAD`.

## 3. Mount the dataset

The split stays on Linode Object Storage and is mounted with goofys. Restore the mount if it is missing:

```bash
mountpoint -q /home/nitesh_sawadia/mnt/ml-fanpulse || \
  /usr/local/bin/goofys \
    --profile default \
    --endpoint https://de-fra-1.linodeobjects.com \
    --region de-fra-1 \
    ml-fanpulse \
    /home/nitesh_sawadia/mnt/ml-fanpulse
```

```bash
DATA_ROOT=/home/nitesh_sawadia/mnt/ml-fanpulse/sports_saliency/vinet_outplay/split

test -f "$DATA_ROOT/split_manifest.json"
ls -lh "$DATA_ROOT/split_manifest.json"
```

## 4. Check the checkpoint and GPU

```bash
test -f /home/nitesh_sawadia/vinet_s/weights/vinet_s_dhf1k.pt
ls -lh /home/nitesh_sawadia/vinet_s/weights/vinet_s_dhf1k.pt
```

```bash
/home/nitesh_sawadia/vinet_s/venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.ones(1, device="cuda"))
PY
```

The RTX PRO 6000 should report capability `(12, 0)`.

## 5. Check MLflow

```bash
curl --fail --silent --show-error \
  http://104.105.26.182:5000/api/2.0/mlflow/experiments/search \
  -H 'Content-Type: application/json' \
  -d '{"max_results":1}' >/dev/null
echo "MLflow reachable"
```

## 6. Start training

```bash
cd /home/nitesh_sawadia/vinet_s/vinet-video-saliency
mkdir -p /home/nitesh_sawadia/vinet_s/logs

LOG=/home/nitesh_sawadia/vinet_s/logs/train_$(date -u +%Y%m%dT%H%M%SZ).log

nohup bash ViNet_S/bash_scripts/run_train_ViNet_S_outplay.sh \
  >"$LOG" 2>&1 &

echo $! | tee /home/nitesh_sawadia/vinet_s/train.pid
echo "LOG=$LOG"
```

Confirm it is running:

```bash
ps -fp "$(cat /home/nitesh_sawadia/vinet_s/train.pid)"
tail -f /home/nitesh_sawadia/vinet_s/logs/train_*.log
```

Watch the GPU from another shell:

```bash
watch -n 2 nvidia-smi
```

Startup can take a while because the loader checks the mounted split.

## 7. Open TensorBoard

Ports 6006 and 8051 are already in use. Start TensorBoard on 6007:

```bash
nohup /home/nitesh_sawadia/vinet_s/venv/bin/tensorboard \
  --logdir /home/nitesh_sawadia/vinet_s/tensorboard \
  --host 127.0.0.1 \
  --port 6007 \
  >/home/nitesh_sawadia/vinet_s/logs/tensorboard.log 2>&1 &

echo $! >/home/nitesh_sawadia/vinet_s/tensorboard.pid
```

From the local machine:

```bash
ssh -N -L 6007:127.0.0.1:6007 173.255.234.238
```

Open `http://127.0.0.1:6007`. One log directory lists every run. Each run has `train` and `val` writers, so those curves share a chart.

## 8. Inspect a run

```bash
ls -lt /home/nitesh_sawadia/vinet_s/runs
ls -lt /home/nitesh_sawadia/vinet_s/tensorboard
```

A run directory contains `config/args.json`, `checkpoints/best.pt`, `checkpoints/last.pt`, and `checkpoints/epoch_<N>.pt`. TensorBoard events are not stored inside that directory. They are under `/home/nitesh_sawadia/vinet_s/tensorboard/<same run name>/{train,val}`.

## Sync later changes

Edit and commit locally, then push to the VM:

```bash
cd /Users/niteshs/amagi-FanPulse/vinet-video-saliency

git add <files>
git commit -m "Describe the training change"
git push vm master
```

Check the VM:

```bash
ssh 173.255.234.238 \
  'cd /home/nitesh_sawadia/vinet_s/vinet-video-saliency && git status && git log -1 --oneline'
```

If a commit was made on the VM, bring it back locally:

```bash
git pull vm master
```

Do not push while the VM worktree has uncommitted edits, and do not edit tracked files on the VM during a run.
