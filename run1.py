# Copyright (2025) Bytedance Ltd. and/or its affiliates 
import argparse
import numpy as np
import os
import torch
import time
import cv2

from video_depth_anything.video_depth import VideoDepthAnything

class TimingStats:
    def __init__(self):
        self.model_load_time = 0
        self.total_read_time = 0
        self.total_write_time = 0
        self.inference_time = 0
        self.extra_save_time = 0
        self.start_time = time.time()
    
    def get_total_time(self):
        return time.time() - self.start_time
    
    def print_stats(self, total_frames):
        print("\nProcessing Time Breakdown:")
        print(f"Model Loading: {self.model_load_time:.2f}s")
        print(f"Video Reading: {self.total_read_time:.2f}s")
        print(f"Video Writing: {self.total_write_time:.2f}s")
        print(f"Depth Inference: {self.inference_time:.2f}s")
        if self.extra_save_time > 0:
            print(f"Additional Format Saving: {self.extra_save_time:.2f}s")
        print(f"Total Time: {self.get_total_time():.2f}s")
        
        print(f"\nPer-frame Statistics:")
        print(f"Number of Frames: {total_frames}")
        print(f"Average Processing Time per Frame: {self.inference_time/total_frames:.3f}s ({(total_frames/self.inference_time):.1f} FPS)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    # ... [parser arguments remain the same] ...
    args = parser.parse_args()

    timing = TimingStats()

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

    # Model loading with timing
    model_start = time.time()
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    timing.model_load_time = time.time() - model_start

    # Video setup
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')

    # Video properties setup with timing
    read_start = time.time()
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {args.input_video}")
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    timing.total_read_time += time.time() - read_start

    # Calculate output dimensions
    if orig_width > orig_height:
        output_width = min(args.max_res, orig_width)
        output_height = int(orig_height * (output_width / orig_width))
    else:
        output_height = min(args.max_res, orig_height)
        output_width = int(orig_width * (output_height / orig_height))
    
    output_fps = orig_fps if args.target_fps == -1 else args.target_fps

    # Initialize video writers
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_writer = cv2.VideoWriter(processed_video_path, fourcc, output_fps, (output_width, output_height))
        depth_writer = cv2.VideoWriter(depth_vis_path, fourcc, output_fps, (output_width, output_height), isColor=not args.grayscale)
        
        if not frame_writer.isOpened() or not depth_writer.isOpened():
            raise RuntimeError("Failed to initialize video writers")
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Error setting up video writers: {str(e)}")

    # Process video in chunks with separate timing for each operation
    chunk_size = 300
    frames = []
    all_depths = []  # Store depths if needed for additional formats

    try:
        while cap.isOpened():
            # Read frame with timing
            read_start = time.time()
            ret, frame = cap.read()
            timing.total_read_time += time.time() - read_start
            
            if not ret:
                break
                
            frames.append(frame)
            
            if len(frames) == chunk_size:
                # Convert frames and process chunk
                frames_array = np.array(frames)
                
                # Inference timing
                infer_start = time.time()
                depths, fps = video_depth_anything.infer_video_depth(frames_array, args.target_fps, 
                                                                   input_size=args.input_size, 
                                                                   device=DEVICE, fp32=args.fp32)
                timing.inference_time += time.time() - infer_start
                
                if args.save_npz or args.save_exr:
                    all_depths.extend(depths)
                
                # Write frames and depths with timing
                write_start = time.time()
                for frame in frames:
                    frame_writer.write(frame)
                for depth in depths:
                    depth_uint8 = (depth * 255).astype(np.uint8)
                    depth_writer.write(depth_uint8)
                timing.total_write_time += time.time() - write_start
                
                frames = []
                del frames_array, depths
                torch.cuda.empty_cache()

        # Process remaining frames
        if frames:
            frames_array = np.array(frames)
            
            infer_start = time.time()
            depths, fps = video_depth_anything.infer_video_depth(frames_array, args.target_fps,
                                                               input_size=args.input_size,
                                                               device=DEVICE, fp32=args.fp32)
            timing.inference_time += time.time() - infer_start
            
            if args.save_npz or args.save_exr:
                all_depths.extend(depths)
            
            write_start = time.time()
            for frame in frames:
                frame_writer.write(frame)
            for depth in depths:
                depth_uint8 = (depth * 255).astype(np.uint8)
                depth_writer.write(depth_uint8)
            timing.total_write_time += time.time() - write_start
            
            del frames_array, depths
            torch.cuda.empty_cache()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Release resources
        cap.release()
        frame_writer.release()
        depth_writer.release()
        
        # Handle additional format saving with timing
        if args.save_npz or args.save_exr:
            extra_save_start = time.time()
            
            if args.save_npz:
                depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
                np.savez_compressed(depth_npz_path, depths=np.array(all_depths))
            
            if args.save_exr:
                depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
                os.makedirs(depth_exr_dir, exist_ok=True)
                import OpenEXR
                import Imath
                for i, depth in enumerate(all_depths):
                    output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                    header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                    header["channels"] = {
                        "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    }
                    exr_file = OpenEXR.OutputFile(output_exr, header)
                    exr_file.writePixels({"Z": depth.tobytes()})
                    exr_file.close()
                    
            timing.extra_save_time = time.time() - extra_save_start

        # Print final timing statistics
        timing.print_stats(total_frames)

    except Exception as e:
        # Clean up resources on error
        cap.release()
        frame_writer.release()
        depth_writer.release()
        raise RuntimeError(f"Error during video processing: {str(e)}")