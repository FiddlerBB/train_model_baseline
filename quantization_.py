def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic,quantize, QuantType
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QUInt8)
    print(f"quantized model saved to:{quantized_model_path}")


quantize_onnx_model(r"D:\study\Projects\train_model_baseline\models\liveness_detection.onnx", "quantizated_liveness.onnx")
import onnxruntime as rt
import onnx
model_path = 'quantizated_liveness.onnx'

model = onnx.load(model_path)
session = rt.InferenceSession(model_path)
#
# # Print the input and output details
input_details = session.get_inputs()
output_details = session.get_outputs()

print("Input Details:")
for i in input_details:
    print(i)

print("\nOutput Details:")
for o in output_details:
    print(o)