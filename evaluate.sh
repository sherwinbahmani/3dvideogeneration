resolution=256 # Change to 128 for TaiChi
dataset=/path/to/dataset
pkl=/path/to/network.pkl
outdir=/path/to/tmpdir
python create_fake_dataset.py --network_pkl $pkl --output_path $outdir --as_frames true --video_len 16 --batch_size 1
python calc_metrics_for_dataset.py --real_data_path $dataset --fake_data_path $outdir --resolution $resolution --metrics fvd2048_16f --num_runs 1
# rm -rf $outdir