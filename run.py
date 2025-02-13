# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import argparse
import numpy as np
import os
import torch
import time
import cv2  # Add this import for OpenCV

from video_depth_anything.video_depth import VideoDepthAnything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (default: 0)')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU ID {args.gpu_id} is not available. Available GPUs: {torch.cuda.device_count()}")
        torch.cuda.set_device(args.gpu_id)
        DEVICE = f'cuda:{args.gpu_id}'
    else:
        DEVICE = 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    start_time = time.time()
    
    # Model loading
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    model_load_time = time.time() - start_time

    # Video reading and processing in chunks
    read_start = time.time()
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')

    # Get video properties
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {args.input_video}")
        
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate output dimensions maintaining aspect ratio
    if orig_width > orig_height:
        output_width = min(args.max_res, orig_width)
        output_height = int(orig_height * (output_width / orig_width))
    else:
        output_height = min(args.max_res, orig_height)
        output_width = int(orig_width * (output_height / orig_height))
        
    # Use original fps if target_fps is -1
    output_fps = orig_fps if args.target_fps == -1 else args.target_fps

    # Initialize video writers using OpenCV
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_writer = cv2.VideoWriter(processed_video_path, fourcc, output_fps, (output_width, output_height))
        depth_writer = cv2.VideoWriter(depth_vis_path, fourcc, output_fps, (output_width, output_height), isColor=not args.grayscale)
        
        if not frame_writer.isOpened() or not depth_writer.isOpened():
            raise RuntimeError("Failed to initialize video writers")
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Error setting up video writers: {str(e)}")

    # Process video in chunks
    chunk_size = 300  # Smaller chunks to manage memory better
    frames = []
    inference_time = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) == chunk_size:
                # Convert frames list to a NumPy array
                frames_array = np.array(frames)
                
                # Process the chunk
                chunk_start = time.time()
                depths, fps = video_depth_anything.infer_video_depth(frames_array, args.target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
                inference_time += time.time() - chunk_start
                
                # Write frames and depths to video immediately
                for frame in frames:
                    frame_writer.write(frame)
                for depth in depths:
                    depth_uint8 = (depth * 255).astype(np.uint8)
                    depth_writer.write(depth_uint8)
                
                frames = []  # Clear the frames list
                del frames_array, depths  # Explicitly free memory
                torch.cuda.empty_cache()  # Clear CUDA cache if using GPU

        # Process any remaining frames
        if frames:
            frames_array = np.array(frames)
            chunk_start = time.time()
            depths, fps = video_depth_anything.infer_video_depth(frames_array, args.target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
            inference_time += time.time() - chunk_start
            
            for frame in frames:
                frame_writer.write(frame)
            for depth in depths:
                depth_uint8 = (depth * 255).astype(np.uint8)
                depth_writer.write(depth_uint8)
            del frames_array, depths
            torch.cuda.empty_cache()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release resources
        cap.release()
        frame_writer.release()
        depth_writer.release()

        read_time = time.time() - read_start
        save_time = 0  # Initialize save time
        
        # Optional saving of additional formats
        extra_save_start = time.time()
        if args.save_npz:
            depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
            np.savez_compressed(depth_npz_path, depths=depths)
        if args.save_exr:
            depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
            os.makedirs(depth_exr_dir, exist_ok=True)
            import OpenEXR
            import Imath
            for i, depth in enumerate(depths):
                output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": depth.tobytes()})
                exr_file.close()
        extra_save_time = time.time() - extra_save_start

        total_time = time.time() - start_time
        
        # Print timing information
        print("\nProcessing Time Breakdown:")
        print(f"Model Loading: {model_load_time:.2f}s")
        print(f"Video Reading: {read_time:.2f}s")
        print(f"Depth Inference: {inference_time:.2f}s")
        print(f"Video Saving: {save_time:.2f}s")
        if args.save_npz or args.save_exr:
            print(f"Additional Format Saving: {extra_save_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        
        # Print per-frame statistics
        print(f"\nPer-frame Statistics:")
        print(f"Number of Frames: {total_frames}")
        print(f"Average Processing Time per Frame: {inference_time/total_frames:.3f}s ({(total_frames/inference_time):.1f} FPS)")
    except Exception as e:
        # Clean up resources on error
        cap.release()
        frame_writer.release()
        depth_writer.release()
        raise RuntimeError(f"Error during video processing: {str(e)}")