from ultralytics import YOLO
import os
import yaml
import glob

# Define paths
model_path = r"D:\projects\quantizer\model_vhf_pm11m.pt_dsc23000_th6.2_bc0.368.pt"
dataset_path = r"D:\vhf_dataset_quant"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "train")  # Use train as val to match class count
save_dir = r"D:\projects\quantizer"
data_yaml = r"D:\projects\quantizer\data.yaml"
image_path = r"D:\vhf_dataset_quant\train\color-purple\day_00509_0.jpg"

# Validate dataset paths
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train directory not found: {train_path}")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Inference image not found: {image_path}")

# Log dataset info
class_folders = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
print(f"Found {len(class_folders)} classes in {train_path}")
print(f"Sample classes: {class_folders[:5]}")
sample_class = class_folders[0]
sample_images = os.listdir(os.path.join(train_path, sample_class))[:5]
print(f"Sample images in {sample_class}: {sample_images}")

# Generate data.yaml
class_names = sorted(class_folders)
num_classes = len(class_names)
data_yaml_content = {
    "train": train_path,
    "val": val_path,
    "nc": num_classes,
    "names": class_names
}
with open(data_yaml, "w") as f:
    yaml.safe_dump(data_yaml_content, f, sort_keys=False)
print(f"Generated data.yaml at {data_yaml}")
 
model = YOLO(model_path, task="classify")
 
model.export(
    format="openvino",
    int8=True,
    data=dataset_path,
    imgsz=320,
    save_dir=save_dir
)

 