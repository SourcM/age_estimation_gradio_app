# Age Estimation Application

## Overview

This application provides a privacy-focused, non-invasive age estimation tool based on facial analysis. It uses computer vision and machine learning to estimate a person's age from a facial image without storing any personal data.

## Purpose and Ethical Use

This tool was designed with several legitimate use cases in mind:

### Child Online Protection
- **Age Verification**: Provides a non-invasive method to verify age claims without requiring personal identification documents
- **Content Filtering**: Helps platforms ensure age-appropriate content delivery
- **Guardian Oversight**: Allows parents/guardians to verify appropriate platform usage

### Privacy-Focused Approach
- **No Data Storage**: Images are processed locally and immediately deleted
- **No Biometric Retention**: The system doesn't store facial templates or other biometric data
- **Anonymous Assessment**: Only returns an estimated age without identifying individuals

## How It Works

1. **Face Detection**: The app uses the CenterFace model to detect faces and facial landmarks
2. **Face Alignment**: Detected faces are aligned to a standard position using facial landmarks
3. **Age Estimation**: A deep learning model analyzes the aligned face to estimate age
4. **Result Display**: The estimated age is shown alongside a visualization of the detection
5. **Image Deletion**: All uploaded images are immediately deleted after processing

## Technical Details

The application utilizes:
- **CenterFace** for accurate face detection and landmark identification
- **Procrustes transformation** for face alignment
- **ONNX Runtime** for efficient neural network inference
- **Gradio** for a simple, accessible web interface

## Installation

### Prerequisites
- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`):
  - numpy
  - onnx
  - onnxruntime
  - opencv-python
  - gradio

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/age-estimation-app.git
   cd age-estimation-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the required model files are in the root directory:
   - `centerface_640_640.onnx`: Face detection model
   - `age1.onnx`: Age estimation model

## Usage

Run the application:
```bash
python app.py
```

The interface will be accessible at `http://127.0.0.1:7860` in your web browser.

### Best Practices for Accurate Results
- Upload images with clear, well-lit frontal faces
- For best results, ensure the face is not obscured by glasses, masks, or other objects
- The system works best with unobstructed, neutral facial expressions

## Privacy Statement

This application is designed with privacy as a core principle:

1. **No Data Storage**: All uploaded images are processed locally and immediately deleted after processing
2. **No Server Uploads**: When run locally, images never leave your device
3. **No User Tracking**: No cookies, user IDs, or tracking mechanisms are used
4. **Transparent Processing**: The code is open-source and available for inspection

## Limitations

- The age estimation is an approximation and may have varying accuracy across different demographics
- Performance is dependent on image quality, lighting, and face positioning
- The system is designed for frontal faces and may be less accurate with profile views

## License

MIT License

Copyright (c) 2025 SourcM

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- The face detection model is based on [CenterFace](https://github.com/Star-Clouds/CenterFace)
- Face processing utilities are derived from methods commonly used in computer vision research

## Responsible Use Guidelines

This tool is intended for legitimate age verification purposes with appropriate consent. It should not be used for:
- Surveillance without consent
- Discriminatory practices
- Collecting or profiling personal data
- Any purpose that violates privacy laws or regulations

## Contributing

Contributions to improve the system's accuracy, privacy measures, or usability are welcome. Please feel free to submit pull requests or open issues for discussion.
