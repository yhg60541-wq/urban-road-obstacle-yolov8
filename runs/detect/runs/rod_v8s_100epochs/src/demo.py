import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# 加载模型
model = YOLO('best.pt')

# 25个类别名称
CLASS_NAMES = ['Bike', 'Building', 'Car', 'Person', 'Stairs',
               'Traffic sign', 'Electrical Pole', 'Road', 'Motorcycle', 'Dustbin',
               'Dog', 'Manhole', 'Tree', 'Guard rail', 'Pedestrian crosswalk',
               'Truck', 'Bus', 'Bench', 'Traffic Cone', 'Fire hydrant',
               'Teraffic Barrel', 'Plant Pot', 'Electrical Box', 'Chair', 'Bicycle Rack']

# 颜色映射
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]


def detect_image(img, conf_threshold=0.25):
    """检测图片"""
    results = model.predict(img, conf=conf_threshold)
    result = results[0]

    annotated_img = result.plot()

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            'class': CLASS_NAMES[class_id],
            'confidence': f"{conf:.2%}",
            'bbox': box.xyxy[0].tolist()
        })

    return annotated_img, detections


def detect_video(video_path, conf_threshold=0.25):
    """检测视频"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 保存临时文件
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1
        total_detections += len(results[0].boxes)

    cap.release()
    out.release()

    return temp_output.name, f"处理完成: {frame_count}帧, 共检测到{total_detections}个物体"


def process_camera(conf_threshold=0.25):
    """实时摄像头检测（生成器）"""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()

        # 添加检测统计
        det_count = len(results[0].boxes)
        cv2.putText(annotated, f"Detected: {det_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        yield annotated

    cap.release()


# 创建界面
with gr.Blocks(title="城市道路障碍物检测系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 城市道路障碍物检测系统
    ### 基于YOLOv8s | 支持25类障碍物 | mAP50: 91.7%
    """)

    # 置信度滑块
    conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="置信度阈值", info="越高越准确，越低越敏感")

    with gr.Tabs():
        # 标签页1：图片检测
        with gr.TabItem("📷 图片检测"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(label="上传图片", type="numpy")
                    img_btn = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="检测结果")

            with gr.Row():
                img_results = gr.JSON(label="检测详情")

            img_btn.click(
                fn=detect_image,
                inputs=[input_img, conf_slider],
                outputs=[output_img, img_results]
            )

        # 标签页2：视频检测
        with gr.TabItem("🎬 视频检测"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="上传视频")
                    video_btn = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    output_video = gr.Video(label="检测结果")

            video_status = gr.Textbox(label="处理状态")

            video_btn.click(
                fn=detect_video,
                inputs=[input_video, conf_slider],
                outputs=[output_video, video_status]
            )

        # 标签页3：实时摄像头
        with gr.TabItem("📹 实时摄像头"):
            gr.Markdown("### 实时摄像头检测（需要允许摄像头权限）")
            camera_btn = gr.Button("开启摄像头检测", variant="primary")
            camera_output = gr.Image(label="实时检测画面", streaming=True)

            camera_btn.click(
                fn=process_camera,
                inputs=[conf_slider],
                outputs=[camera_output]
            )

        # 标签页4：统计信息
        with gr.TabItem("📊 统计信息"):
            gr.Markdown("""
            ### 模型性能统计

            | 指标 | 数值 |
            |------|------|
            | mAP50 | 91.7% |
            | mAP50-95 | 75.1% |
            | Precision | 88.2% |
            | Recall | 89.7% |
            | 模型大小 | 22.5 MB |
            | 推理速度(GPU) | 0.8ms/张 |
            | 推理速度(CPU) | ~300ms/张 |

            ### 支持的25类障碍物
            Bike, Building, Car, Person, Stairs, Traffic sign, Electrical Pole, Road, 
            Motorcycle, Dustbin, Dog, Manhole, Tree, Guard rail, Pedestrian crosswalk,
            Truck, Bus, Bench, Traffic Cone, Fire hydrant, Teraffic Barrel, 
            Plant Pot, Electrical Box, Chair, Bicycle Rack
            """)

# 运行
demo.launch(share=False, server_name="127.0.0.1", server_port=7860)