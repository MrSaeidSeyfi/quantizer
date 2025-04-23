import os
import cv2
import numpy as np
import gradio as gr
import onnxruntime as ort
import time
from pathlib import Path
from typing import Tuple, Optional

from utils.image_processing import preprocess_image, load_calibration_images
from utils.model_conversion import convert_to_onnx
from utils.quantization import ImageCalibrationReader, quantize_model
from onnxruntime.quantization import QuantFormat, QuantType

def process_model(
    model_path: str,
    calibration_image_folder: str,
    output_dir: str,
    input_size: int,
    input_name: str = 'input',
    quant_format: str = 'QDQ',
    activation_type: str = 'QUInt8',
    weight_type: str = 'QInt8',
    per_channel: bool = False,
    reduce_range: bool = False,
    max_calibration_images: int = 100,
    test_image: Optional[str] = None,
    progress: gr.Progress = gr.Progress()
) -> Tuple[str, Optional[str]]:
    """
    Process a model through the quantization pipeline.
    
    Args:
        model_path: Path to the input model
        calibration_image_folder: Directory containing calibration images
        output_dir: Directory to save output files
        input_size: Size of input images (assumed square)
        input_name: Name of the input tensor
        quant_format: Quantization format ('QDQ' or 'QOperator')
        activation_type: Type for activations ('QUInt8' or 'QInt8')
        weight_type: Type for weights ('QUInt8' or 'QInt8')
        per_channel: Whether to use per-channel quantization
        reduce_range: Whether to reduce the range of quantized values
        max_calibration_images: Maximum number of calibration images to use
        test_image: Optional path to a test image
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (result message, path to quantized model)
    """
    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(model_path).stem
    input_size_tuple = (input_size, input_size)
    
    # Convert model to ONNX
    progress(0.1, desc="Converting model to ONNX...")
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    if not convert_to_onnx(model_path, onnx_path, input_size_tuple, model_class, input_names, output_names):
        return "Error: Failed to convert model to ONNX", None
    
    # Load and preprocess calibration images
    progress(0.3, desc="Loading calibration images...")
    try:
        calibration_images = load_calibration_images(
            calibration_image_folder,
            input_size_tuple,
            max_calibration_images
        )
    except Exception as e:
        return f"Error loading calibration images: {str(e)}", None
    
    # Create calibration reader
    calib_reader = ImageCalibrationReader(calibration_images, input_name)
    
    # Map string parameters to enum values
    quant_format_map = {
        'QDQ': QuantFormat.QDQ,
        'QOperator': QuantFormat.QOperator
    }
    activation_type_map = {
        'QUInt8': QuantType.QUInt8,
        'QInt8': QuantType.QInt8
    }
    weight_type_map = {
        'QUInt8': QuantType.QUInt8,
        'QInt8': QuantType.QInt8
    }
    
    # Quantize model
    progress(0.5, desc="Quantizing model...")
    quantized_model = os.path.join(output_dir, f"{model_name}_quantized.onnx")
    if not quantize_model(
        onnx_path,
        quantized_model,
        calib_reader,
        quant_format=quant_format_map[quant_format],
        activation_type=activation_type_map[activation_type],
        weight_type=weight_type_map[weight_type],
        per_channel=per_channel,
        reduce_range=reduce_range
    ):
        return "Error: Failed to quantize model", None
    
    # Test model if test image provided
    results = {
        "Model Name": model_name,
        "Original Model Path": model_path,
        "ONNX Model Path": onnx_path,
        "Quantized Model Path": quantized_model,
        "Calibration Images Used": len(calibration_images),
        "Quantization Format": quant_format,
        "Activation Type": activation_type,
        "Weight Type": weight_type,
        "Per Channel": "Yes" if per_channel else "No",
        "Reduce Range": "Yes" if reduce_range else "No"
    }
    
    if test_image:
        progress(0.8, desc="Evaluating models...")
        test_img = preprocess_image(test_image, input_size_tuple)
        
        # Test FP32 model
        fp32_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        start_time = time.time()
        fp32_outputs = fp32_session.run(None, {input_name: test_img})
        fp32_time = time.time() - start_time
        
        # Test quantized model
        quant_session = ort.InferenceSession(quantized_model, providers=['CPUExecutionProvider'])
        start_time = time.time()
        quant_outputs = quant_session.run(None, {input_name: test_img})
        quant_time = time.time() - start_time
        
        results.update({
            "FP32 Inference Time": f"{fp32_time:.4f} seconds",
            "Quantized Inference Time": f"{quant_time:.4f} seconds",
            "Speed Improvement": f"{(fp32_time/quant_time):.2f}x faster"
        })
        
        # Try hardware acceleration
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            hw_session = ort.InferenceSession(quantized_model, providers=providers)
            start_time = time.time()
            hw_outputs = hw_session.run(None, {input_name: test_img})
            hw_time = time.time() - start_time
            
            results.update({
                "Hardware Accelerated Time": f"{hw_time:.4f} seconds",
                "Hardware Provider Used": hw_session.get_providers()[0],
                "HW Speed Improvement": f"{(fp32_time/hw_time):.2f}x faster than FP32"
            })
        except:
            results["Hardware Acceleration"] = "Not available or failed"
    
    progress(1.0, desc="Complete!")
    
    # Format results
    result_text = "## Model Quantization Results\n\n"
    for key, value in results.items():
        result_text += f"**{key}:** {value}\n"
    
    return result_text, quantized_model

def create_gradio_app():
    """Create and return the Gradio web interface."""
    with gr.Blocks(title="Neural Network Model Quantizer") as app:
        gr.Markdown("# Neural Network Model Quantizer")
        gr.Markdown("Convert and quantize neural network models to ONNX INT8 for faster inference on edge devices")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_path = gr.File(
                    label="Select Model File",
                    file_types=[".pt", ".pth", ".h5", ".onnx", ".weights"],
                    file_count="single"
                )
                
                calibration_dir = gr.Textbox(
                    label="Calibration Images Directory",
                    placeholder="Path to folder with calibration images"
                )
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./quantized_models",
                    placeholder="Where to save quantized models"
                )
                
                test_image = gr.File(
                    label="Test Image (optional)",
                    file_types=["image"],
                    file_count="single"
                )
                
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Model Configuration")
                    input_size = gr.Slider(
                        label="Input Size",
                        minimum=32,
                        maximum=1280,
                        value=640,
                        step=32,
                        info="Model input resolution (width=height)"
                    )
                    
                    input_name = gr.Textbox(
                        label="Input Tensor Name",
                        value="input",
                        info="ONNX model input tensor name"
                    )
                    
                    max_calibration_images = gr.Slider(
                        label="Calibration Images",
                        minimum=10,
                        maximum=1000,
                        value=100,
                        step=10,
                        info="Number of images to use for calibration"
                    )
                
                with gr.Group():
                    gr.Markdown("### Quantization Options")
                    quant_format = gr.Radio(
                        label="Quantization Format",
                        choices=["QDQ", "QOperator"],
                        value="QDQ",
                        info="QDQ is more compatible with more runtimes"
                    )
                    
                    activation_type = gr.Radio(
                        label="Activation Type",
                        choices=["QUInt8", "QInt8"],
                        value="QUInt8",
                        info="Type used for activations"
                    )
                    
                    weight_type = gr.Radio(
                        label="Weight Type",
                        choices=["QInt8", "QUInt8"],
                        value="QInt8",
                        info="Type used for weights"
                    )
                    
                    with gr.Row():
                        per_channel = gr.Checkbox(
                            label="Per Channel",
                            value=False,
                            info="Better accuracy but may not work on all hardware"
                        )
                        
                        reduce_range = gr.Checkbox(
                            label="Reduce Range",
                            value=False,
                            info="Required for some hardware (e.g. older Intel CPUs)"
                        )
        
        with gr.Row():
            quantize_btn = gr.Button("Start Quantization", variant="primary")
            download_btn = gr.Button(
                "Download Quantized Model", 
                variant="secondary", 
                interactive=False  # Set initial state here
            )
        
        with gr.Row():
            result_text = gr.Markdown()
            quantized_model_path = gr.State()
        
        def on_quantize(model_file, calib_dir, out_dir, in_size, in_name, q_format, act_type,
                       w_type, per_chan, red_range, max_calib_imgs, test_img):
            if not model_file or not calib_dir:
                return "Please provide both a model file and calibration images directory.", None
            
            model_path = model_file.name
            test_image_path = test_img.name if test_img else None
            
            results, quant_model = process_model(
                model_path=model_path,
                calibration_image_folder=calib_dir,
                output_dir=out_dir,
                input_size=in_size,
                input_name=in_name,
                quant_format=q_format,
                activation_type=act_type,
                weight_type=w_type,
                per_channel=per_chan,
                reduce_range=red_range,
                max_calibration_images=max_calib_imgs,
                test_image=test_image_path
            )
            
            return results, quant_model
        
        quantize_btn.click(
            fn=on_quantize,
            inputs=[model_path, calibration_dir, output_dir, input_size, input_name,
                    quant_format, activation_type, weight_type, per_channel, reduce_range,
                    max_calibration_images, test_image],
            outputs=[result_text, quantized_model_path]
        )
        
        download_btn.click(
            fn=lambda x: x,
            inputs=[quantized_model_path],
            outputs=gr.File()
        )
        
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()