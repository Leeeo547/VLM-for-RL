import cv2
import os

def extract_frames_with_lookahead(video_path, output_dir, num_samples=5, new_size=(256, 256)):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 检查视频是否有足够的帧
    min_required_frames = num_samples + (num_samples - 1) * 2  # 至少需要的帧数
    if total_frames < min_required_frames:
        print(f"Warning: Video has {total_frames} frames, but need at least {min_required_frames} frames")
        print("Will extract as many frames as possible")
    
    # 计算除最后一组外的抽样间隔（抽取 num_samples-1 组）
    # 确保最后一组的 C 帧是视频的最后一帧
    last_c_frame = total_frames - 1
    interval = max(1, (last_c_frame) // (num_samples - 1))
    
    extracted_count = 0
    
    # 抽取前 num_samples-1 组
    for i in range(num_samples - 1):
        sample_frame_idx = i * interval  # C 帧的位置
        
        # 提取 A, B, C 三帧
        for offset, suffix in enumerate(['A', 'B', 'C']):
            frame_idx = sample_frame_idx + offset
            
            # 确保不超出视频范围
            if frame_idx >= total_frames:
                print(f"Warning: Frame {frame_idx} exceeds total frames, skipping")
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # --- Start of Change ---
            # Resize the frame
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            # --- End of Change ---

            output_path = os.path.join(output_dir, f"{i+1:02d}{suffix}.jpg")
            # --- Change: Save the resized frame ---
            cv2.imwrite(output_path, resized_frame)
            extracted_count += 1
    
    # 抽取最后一组，确保 C 帧是视频的最后一帧
    last_group_num = num_samples
    last_c_frame_idx = total_frames - 1
    
    # 提取最后一组的 A, B, C
    for offset, suffix in [(2, 'A'), (1, 'B'), (0, 'C')]:
        frame_idx = last_c_frame_idx - offset
        
        # 确保不小于 0
        if frame_idx < 0:
            print(f"Warning: Frame index {frame_idx} is negative, skipping")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue
        
        # --- Start of Change ---
        # Resize the frame
        resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        # --- End of Change ---

        output_path = os.path.join(output_dir, f"{last_group_num:02d}{suffix}.jpg")
        # --- Change: Save the resized frame ---
        cv2.imwrite(output_path, resized_frame)
        extracted_count += 1
    
    # 释放视频对象
    cap.release()
    print(f"Extracted and resized {extracted_count} frames (in {num_samples} groups) from {video_path} to {output_dir}")
    print(f"Total frames in video: {total_frames}")

# 示例用法
if __name__ == "__main__":
    video_path = "episode_000001.mp4"  # 替换为你的视频文件路径
    output_dir = "pickup"    # 替换为你想保存帧的目录
    # You can now also specify the new size
    extract_frames_with_lookahead(video_path, output_dir, num_samples=5, new_size=(256, 256))