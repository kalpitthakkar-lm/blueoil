### Statistics script usage

To use the script `get_stats.sh`, you need to first get the experiment ID, path to save directory and the checkpoint step that needs to be restored.
Once you have that info, you can run the following from the repository directory:

```
$ CUDA_VISIBLE_DEVICES=0 ./get_stats.sh <exp_id> <save_dir> <ckpt_step>
```

This will run the scripts to get three types of information: (1) Model latency, (2) Model size profile, (3) Protobuffer for frozen network graph.
