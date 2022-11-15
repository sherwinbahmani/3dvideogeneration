# For FaceForensics we used seeds: 431, 28, 83, 379, 288, 160, 612, 326
# For FaceForensics pre-trained on FFHQ we used seeds: 6, 45, 158, 418, 861, 154, 1496, 1574
# For MEAD we used seeds: 166, 196, 24, 70, 73, 87, 97, 158

pkl=/path/to/network.pkl
out_dir=/path/to/output
render_programs=(rotation_camera_yaw rotation_camera_yaw_pitch front front_zoom)

for render_program in ${render_programs[@]}
do
for seed in {0..9}
do
out_dir_program=${out_dir}/${render_program}
python generate.py --outdir ${out_dir_program} --truncation_psi 1.0 --seeds $seed --network_pkl ${pkl} --render_program $render_program --time_steps 16 --n_steps 16
done
done
