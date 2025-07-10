# TensorRTInferencer Unit Tests

This test suite verifies the correctness, robustness, and stability of the `TensorRTInferencer` class. The class performs GPU-based preprocessing and inference using TensorRT and OpenCV CUDA. The tests are written using Google Test.

## ✅ What’s Covered

| Test Category                  | What It Validates                                                                 |
|-------------------------------|------------------------------------------------------------------------------------|
| Engine file loading           | Whether valid/invalid engine files are handled properly                           |
| Image preprocessing           | Format conversion, resizing, normalization, and type validation                   |
| Inference execution           | Proper memory transfer, input validation, and exception handling                  |
| Output correctness            | Whether inference outputs have correct size, type, and content                    |
| Edge cases                    | Behavior on empty inputs, invalid formats, incorrect shapes                       |
| Performance/Robustness        | Threaded execution, multiple reuse, determinism                                   |

---

## Test Descriptions

### Engine Initialization
- `CanReadEngineFile`
  Verifies that a valid engine file can be loaded without throwing.

- `WrongEngineFile`
  Confirms a `std::runtime_error` is thrown when an invalid path is passed.

---

### Image Preprocessing (`preprocessImage`)
- `PreprocessImageGrayscale`
  BGR input is converted to grayscale and normalized.

- `ThrowsOnEmptyImage`
  Ensures an empty `GpuMat` raises a `std::runtime_error`.

- `PreprocessGrayscaleNoConvert`
  Grayscale input bypasses conversion.

- `PreprocessGrayscaleWithConvert`
  Grayscale input is handled correctly and normalized.

- `PreprocessColorImage`
  Full color input is correctly converted and resized.

- `PreprocessImageWithInvalidType`
  `CV_8UC4` input causes an exception (not supported).

- `PreprocessImageWithEmptyGpuMat`
  Confirms an empty input is rejected.

- `PreprocessImageWithWrongSize`
  Smaller image is accepted (no shape validation happens here).

- `PreprocessImageWithValidSize`, `WithValidSizeAndType`, etc.
  Valid images are accepted without exceptions.

---

### Inference Execution (`runInference`)
- `RunInferenceThrowsOnWrongSize`
  Image of wrong dimensions triggers an exception.

- `RunInferenceSucceedsOnValidInput`
  A correctly sized float image runs inference without error.

- `RunInferenceFailsOnNullInput`
  Empty `GpuMat` leads to exception during memory copy.

---

### Output Validation
- `OutputHasNonZeroValuesAfterInference`
  Ensures inference produces non-zero values.

- `MakePredictionDoesNotThrowOnValidInput`
  Sanity check that `makePrediction()` works with valid input.

- `MakePredictionReturnsCorrectSize`
  Output is the expected shape and type (`CV_32F`).

- `MakePredictionOutputNotAllZero`
  Ensures the output isn't all zeros, indicating a valid prediction.

- `MakePredictionReturnsGpuMat`
  Checks the return type is non-empty and GPU-based.

- `PredictionOutputHasExpectedRange`
  Confirms the output values are normalized within `[0, 1]`.

---

### Robustness & Edge Cases
- `ThreadedInferenceDoesNotCrash`
  Inference can run concurrently in multiple threads.

- `ReuseInferenceMultipleTimes`
  A single `TensorRTInferencer` instance can run multiple predictions safely.

- `MakePredictionIsDeterministic`
  The same input consistently yields the same output (within a small delta).

---

## Requirements

- CUDA-compatible GPU
- OpenCV with CUDA support
- TensorRT properly installed
- A valid `.engine` file for inference
- Google Test
