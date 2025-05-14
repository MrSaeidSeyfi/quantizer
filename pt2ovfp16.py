import os
from ultralytics import YOLO

# Define paths
model_path = r"D:\projects\quantizer\model_vhf_pm11m.pt_dsc23000_th6.2_bc0.368.pt"
save_dir = r"D:\projects\quantizer\exported_model"

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Initialize YOLO model
model = YOLO(model_path, task="classify")

# Export model to OpenVINO with FP16 optimization
model.export(
    format="openvino",           # Export format
    int8=False,                 # Disable INT8 quantization
    half=True,                  # Enable FP16 quantization
    imgsz=320,                  # Image size
    save_dir=save_dir,          # Output directory
    optimize=True,              # Enable optimizations
    dynamic=False,              # Disable dynamic shapes for better compatibility
    simplify=True,              # Simplify ONNX graph for better performance
    workspace=4,                # Workspace size in GB for ONNX simplifier
    batch=1,                    # Static batch size for inference
    verbose=True                # Enable verbose logging
)

print(f"Model successfully exported to {save_dir} in FP16 OpenVINO format.")