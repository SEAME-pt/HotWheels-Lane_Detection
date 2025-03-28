#!/usr/bin/python3.6
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        """
        Initialize TensorRT inference engine

        :param engine_path: Path to the TensorRT engine file
        """
        # TensorRT logger
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load the TensorRT engine
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        # Determine input and output shapes and allocate memory
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

    def preprocess_image(self, image):
        """
        Preprocess the input image to match model requirements

        :param image: Input image from camera
        :return: Preprocessed image ready for inference
        """
        # Resize image to match model input shape
        # Modify these values based on your specific model's requirements
        resized = cv2.resize(image, (224, 224))

        # Normalize (adjust based on your model's requirements)
        normalized = resized.astype(np.float32) / 255.0

        # Transpose to match TensorRT model input format (if needed)
        # From HWC to CHW
        transposed = normalized.transpose(2, 0, 1)

        return transposed

    def inference(self, input_data):
        """
        Run inference on the input data

        :param input_data: Preprocessed input image
        :return: Model output
        """
        # Copy input data to the device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0].device,
            self.inputs[0].host,
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output data back to host
        cuda.memcpy_dtoh_async(
            self.outputs[0].host,
            self.outputs[0].device,
            self.stream
        )

        # Synchronize the stream
        self.stream.synchronize()

        return self.outputs[0].host

    def visualize_output(self, original_image, model_output):
        """
        Visualize model output (customize based on your model)

        :param original_image: Original camera frame
        :param model_output: Output from the model
        :return: Visualization of the results
        """
        output_image = original_image.copy()

        # Add text to show inference occurred
        cv2.putText(output_image,
                    f"Model Output: {model_output[:5]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        return output_image

class HostDeviceMem:
    """
    Helper class to manage host and device memory
    """
    def __init__(self, host_mem, device_mem):
        """
        :param host_mem: Pinned host memory
        :param device_mem: Device memory allocation
        """
        self.host = host_mem
        self.device = device_mem

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=720,
    display_height=480,
    display_width=640,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():
    # Initialize camera
    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize TensorRT model
    try:
        # Replace with your actual TensorRT engine path
        model = TensorRTInference('model.engine')
    except Exception as e:
        print(f"Failed to load model: {e}")
        cam.release()
        return

    while True:
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Preprocess image
        input_tensor = model.preprocess_image(frame)

        # Run inference
        model_output = model.inference(input_tensor)

        # Visualize results
        output_frame = model.visualize_output(frame, model_output)

        # Display the frame
        cv2.imshow('TensorRT Inference', output_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
