import os
from tqdm import tqdm
import shutil

def creat_video(in_path, out_path, stride):
    video_names = os.listdir(in_path)
    for video_name in tqdm(video_names):
        out_video_path = os.path.join(out_path, video_name)
        os.makedirs(out_video_path, exist_ok=True)
        in_video_path = os.path.join(in_path, video_name)
        i=0
        for f, frame_name in enumerate(sorted(os.listdir(in_video_path))):
            if f % stride == 0:
                in_frame_path = os.path.join(in_video_path, frame_name)
                out_frame_path = os.path.join(out_video_path, f"{i:07d}.png")
                shutil.copy(in_frame_path, out_frame_path)
                i += 1


if __name__ == "__main__":
    stride = 4
    in_path = "/path/to/taichi"
    out_path = "/path/to/taichi_stride"
    creat_video(in_path, out_path, stride)