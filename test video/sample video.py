import cv2
import os

def extract_frames(video_path, output_dir, num_frames=10):
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
    
    # 如果总帧数少于所需帧数，直接提取所有帧
    if total_frames < num_frames:
        frame_count = 0
        extracted_frames = 0
        while cap.isOpened() and extracted_frames < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            output_path = os.path.join(output_dir, f"frame_{extracted_frames:03d}.jpg")
            cv2.imwrite(output_path, frame)
            extracted_frames += 1
            frame_count += 1
        cap.release()
        print(f"Extracted {extracted_frames} frames from {video_path} to {output_dir}")
        return
    
    # 计算除最后一帧外的抽样间隔（抽取 num_frames-1 帧）
    interval = max(1, (total_frames - 1) // (num_frames - 1))
    
    # 抽取帧
    extracted_frames = 0
    for i in range(num_frames - 1):
        frame_idx = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        output_path = os.path.join(output_dir, f"potato_{extracted_frames:03d}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_frames += 1
    
    # 抽取最后一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    if ret:
        output_path = os.path.join(output_dir, f"potato_{extracted_frames:03d}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_frames += 1
    
    # 释放视频对象
    cap.release()
    print(f"Extracted {extracted_frames} frames from {video_path} to {output_dir}")

# 示例用法
if __name__ == "__main__":
    video_path = "episode_000001.mp4"  # 替换为你的视频文件路径
    output_dir = "pickup"    # 替换为你想保存帧的目录
    extract_frames(video_path, output_dir, num_frames=10)