import torch
import onnx
from pathlib import Path
from typing import Tuple, Union, Optional

def convert_to_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    input_size: Tuple[int, int],
    input_names: Optional[list] = None,
    output_names: Optional[list] = None
) -> bool:
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the PyTorch model
        output_path: Path to save the ONNX model
        input_size: Tuple of (height, width) for input tensor
        input_names: List of input tensor names
        output_names: List of output tensor names
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Load PyTorch model
        model = torch.load(model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *input_size)
        
        # Set default names if not provided
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
            
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=12,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return True
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False 