import gradio as gr
import cv2
import numpy as np
import onnxruntime as ort
import os
import shutil
from pathlib import Path
import face_processing_helpers as fph

# Paths and constants
path = Path(__file__).parent
path2 = os.getcwd()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Face detection class (CenterFace)
class CenterFace(object):
    def __init__(self, landmarks=True):
        self.landmarks = landmarks
        if self.landmarks:
            self.net = cv2.dnn.readNetFromONNX(os.path.join(path2, 'centerface_640_640.onnx'))
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img, height, width, threshold=0.5):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(height, width)
        return self.inference_opencv(img, threshold)

    def inference_opencv(self, img, threshold):
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(self.img_w_new, self.img_h_new), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        if self.landmarks:
            heatmap, scale, offset, lms = self.net.forward(["537", "538", "539", '540'])
        else:
            heatmap, scale, offset = self.net.forward(["535", "536", "537"])
        return self.postprocess(heatmap, lms, offset, scale, threshold)

    def transform(self, h, w):
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode_fast(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode_fast(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets
            
    def decode_fast(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        sz = size
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)

        #numpy computation
        s0, s1 = np.exp(scale0[c0, c1]) * 4, np.exp(scale1[c0, c1]) * 4
        o0, o1 = offset0[c0, c1], offset1[c0, c1]
        s = heatmap[c0, c1]
        x1, y1 = np.maximum(0, (c1 + o1 + 0.5) * 4 - s1 / 2), np.maximum(0, (c0 + o0 + 0.5) * 4 - s0 / 2)
        x1, y1 = np.minimum(x1, sz[1]), np.minimum(y1, sz[0])
        boxes = np.vstack((x1, y1, np.minimum(x1 + s1, sz[1]), np.minimum(y1 + s0, sz[0]), s)).T
        boxes = np.asarray(boxes, dtype=np.float32)
        keep = self.faster_nms(boxes[:, :4], boxes[:, 4], 0.3)
        boxes = boxes[keep, :].copy()

        #landmarks
        lx = np.array([1, 3, 5, 7, 9]) 
        ly = np.array([0, 2, 4, 6, 8])
        landmark_o = landmark[0, :, :, :].copy()

        landmark_o_x = landmark_o[lx,:,:].copy()
        landmark_o_x = landmark_o_x[:, c0, c1] * s1 + x1
        landmark_o_x = landmark_o_x.T

        landmark_o_y = landmark_o[ly,:,:].copy()
        landmark_o_y = landmark_o_y[:, c0, c1] * s0 + y1
        landmark_o_y = landmark_o_y.T

        row_a, col_a = np.shape(landmark_o_x)
        row_b, col_b = np.shape(landmark_o_y)
        all_landmarks = np.ravel([landmark_o_x.T, landmark_o_y.T], 'F').reshape(row_a, col_a+col_b)

        lms = all_landmarks[keep, :].copy()

        return boxes, lms

    def faster_nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]

        return keep

# Helper functions
def remove_fourth_channel(img):
    if len(img.shape) > 2 and img.shape[2] == 4:
        # Convert the image from RGBA to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) < 2:
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        image = img
    return image

def make_image_square(img):
    # Check image channels
    if len(img.shape) > 2 and img.shape[2] == 3:
        pass
    else:
        img = remove_fourth_channel(img)
    # Get size
    height, width, channels = img.shape
   
    # Create a black image
    x = height if height > width else width
    y = height if height > width else width
    square = np.zeros((x, y, 3), np.uint8)
    
    # Place the original image in the center
    square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
    
    return cv2.resize(square, (640, 640))

def get_age(sess, img):
    img = cv2.resize(img, (224, 224))
    img = img/255.0
    img[:,:,0] = (img[:,:,0] - mean[0])/std[0]
    img[:,:,1] = (img[:,:,1] - mean[1])/std[1]
    img[:,:,2] = (img[:,:,2] - mean[2])/std[2]

    img = img.transpose((2, 0, 1))
    im = img[np.newaxis, :, :, :]
    im = im.astype(np.float32)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run(None, {input_name: im})
    oup = result[0].item()
    return oup

def get_model(model_file):
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    EP_list = ['CPUExecutionProvider']
    sess = ort.InferenceSession(model_file, providers=EP_list, sess_options=so)
    return sess

# Initialize models
centerface = CenterFace()
sess = get_model(os.path.join(path2, 'age1.onnx'))

# Main prediction function
def predict_age(input_img):
    # Track temporary file to delete it after processing
    temp_file_path = None
    
    # If input is a file path, read the image and mark for deletion
    if isinstance(input_img, str):
        temp_file_path = input_img
        img = cv2.imread(input_img)
    else:
        # Convert from RGB (Gradio format) to BGR (OpenCV format)
        img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    
    # Process the image
    frame = make_image_square(img)
    h, w = frame.shape[:2]
    
    # Use centerface to get face bounding box and 5-keypoints
    dets, lms = centerface(frame, h, w, threshold=0.35)
    
    result_img = frame.copy()
    
    if len(dets) > 0:
        for i, (det, lm) in enumerate(zip(dets, lms)):
            # Extract face coordinates
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Create landmark array
            lm_arr = []
            for i in range(0, 5):
                lm_arr.append([int(lm[i * 2]), int(lm[i * 2 + 1])])
                # Draw landmarks on result image
                cv2.circle(result_img, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 255, 0), -1)
            
            # Draw face bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Align and crop face
            coords5 = np.asarray(lm_arr)
            aligned_face = fph.crop_face(frame, coords5)
            
            # Convert to RGB
            aligned_face_rgb = aligned_face[:, :, ::-1]
            
            # Get age prediction
            age = get_age(sess, aligned_face_rgb)
            age_text = f"Age: {round(age)}"
            
            # Add age text to result image
            cv2.putText(result_img, age_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Convert result image back to RGB for display
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Clean up temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Temporary file {temp_file_path} deleted.")
            except Exception as e:
                print(f"Error deleting temporary file: {e}")
                
        return result_img, f"Estimated age: {round(age)}"
    else:
        # No face detected
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Clean up temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Temporary file {temp_file_path} deleted.")
            except Exception as e:
                print(f"Error deleting temporary file: {e}")
                
        return result_img, "No face detected. Please upload a clearer image with a frontal face."

# Create Gradio interface
title = "Age Estimation App"
description = """
Upload an image containing a face, and the app will estimate the person's age.
For best results:
- Image should contain a clear, frontal face
- Good lighting conditions are recommended
- Consider removing glasses for better accuracy
"""

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_age,
    inputs=gr.Image(type="filepath"),  # Use filepath to get the temporary file path
    outputs=[
        gr.Image(label="Detection Result"),
        gr.Textbox(label="Age Estimation")
    ],
    title=title,
    description=description,
    examples=[
        # You can add example images here if you want
    ],
    cache_examples=False  # Don't cache example inputs
)

# Define cleanup function for when the app closes
def cleanup_temp_files():
    # Clean up Gradio's temporary directory
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_cached_examples")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up directory: {e}")

# Launch the app
if __name__ == "__main__":
    try:
        # Add privacy notice to description
        description += "\n\n**Privacy Notice**: All uploaded images are processed locally and immediately deleted after processing."
        iface.launch()
    finally:
        # Clean up when the app is closed
        cleanup_temp_files()
