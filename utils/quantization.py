import onnx
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationDataReader
)
from typing import List, Union, Tuple
import numpy as np
from pathlib import Path
class ImageCalibrationReader(CalibrationDataReader):
    """Calibration data reader for image-based model quantization."""
    
    def __init__(
        self,
        calibration_images: List[np.ndarray],
        input_name: str = 'input'
    ):
        """
        Initialize the calibration reader.
        
        Args:
            calibration_images: List of preprocessed calibration images
            input_name: Name of the input tensor
        """
        self.calibration_images = calibration_images
        self.input_name = input_name
        self.index = 0
        
    def get_next(self):
        """Get the next calibration data."""
        if self.index >= len(self.calibration_images):
            return None
            
        data = {self.input_name: self.calibration_images[self.index]}
        self.index += 1
        return data
        
    def rewind(self):
        """Reset the reader to the beginning."""
        self.index = 0

def quantize_model(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    calibration_reader: CalibrationDataReader,
    quant_format: QuantFormat = QuantFormat.QDQ,
    activation_type: QuantType = QuantType.QUInt8,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = False,
    reduce_range: bool = False
) -> bool:
    """
    Quantize an ONNX model using static quantization.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the quantized model
        calibration_reader: Calibration data reader
        quant_format: Quantization format (QDQ or QOperator)
        activation_type: Type for activations
        weight_type: Type for weights
        per_channel: Whether to use per-channel quantization
        reduce_range: Whether to reduce the range of quantized values
        
    Returns:
        bool: True if quantization successful, False otherwise
    """
    try:
        quantize_static(
            model_path,
            output_path,
            calibration_reader,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            per_channel=per_channel,
            reduce_range=reduce_range
        )
        return True
    except Exception as e:
        print(f"Error quantizing model: {str(e)}")
        return False 