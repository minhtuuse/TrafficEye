import gradio as gr
import cv2
import numpy as np
import json
import os
from core.traffic_system import TrafficSystem
from utils.zones import save_zones, load_zones
from utils.config import save_config
from utils.storage import MinioClient

# Initialize System
system = TrafficSystem()
minio_client = MinioClient()

def get_dashboard_stats():
    # In a real app, query MinIO or a DB
    # For now, return mock or in-memory stats if accessible
    # System doesn't expose stats easily in current design without running
    # So we will just show placeholders or query MinIO bucket count if possible
    try:
        proofs = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'])
        count = proofs.get('KeyCount', 0)
        return f"Total Violations Recorded: {count}"
    except Exception as e:
        return f"Error connecting to MinIO: {e}"

def get_proof_gallery():
    try:
        objects = minio_client.s3.list_objects_v2(Bucket=minio_client.buckets['proofs'], MaxKeys=10)
        images = []
        if 'Contents' in objects:
            for obj in objects['Contents']:
                # Generate presigned URL or download
                # For local display, we might need to download to temp
                # But for simplicity, let's just return a placeholder or skip if complex
                images.append((f"http://localhost:9000/{minio_client.buckets['proofs']}/{obj['Key']}", obj['Key']))
        return images
    except:
        return []

# --- Visualization Logic ---
def stream_video():
    system.start()
    for frame, stats in system._process_flow():
        if not system.running:
            break
        yield frame, str(stats)

def stop_system():
    system.stop()
    return None, "System Stopped"

# --- Drawing Logic ---
current_points = []
current_image = None
drawing_mode = "polygon" # "polygon" or line categories

def capture_frame_for_drawing():
    frame = system.capture_first_frame()
    if frame is None:
        return None, "Failed to capture frame"
    
    # Also reset current points for fresh start?
    # global current_points
    # current_points = []
    global current_image
    current_image = frame
    return frame, "Frame Captured"

def load_image_for_drawing(image_input):
    global current_image, current_points
    if image_input is None:
        return None
    current_image = image_input
    current_points = []
    # If user uploads, we use it
    return image_input

def on_select(evt: gr.SelectData, image):
    global current_points, current_image
    if image is None: 
        return image
    
    x, y = evt.index[0], evt.index[1]
    current_points.append([x, y])
    
    # Draw points on a copy
    img_copy = image.copy()
    for pt in current_points:
        cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)
    
    # Draw polygon preview
    if drawing_mode == "polygon" and len(current_points) >= 2:
         # Draw lines connecting points for polygon preview
         pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
         cv2.polylines(img_copy, [pts], False, (255, 0, 0), 2)
         if len(current_points) >= 3:
              cv2.polylines(img_copy, [pts], True, (255, 0, 0), 2)
              
    elif drawing_mode != "polygon" and len(current_points) >= 2:
         # Draw lines pairs
         for j in range(0, len(current_points)-1, 2):
            cv2.line(img_copy, tuple(current_points[j]), tuple(current_points[j+1]), (0, 0, 255), 2)

    return img_copy

def clear_points(image_input):
    global current_points
    current_points = []
    # Return original image (we need to track it)
    global current_image
    if current_image is not None:
        return current_image
    return image_input

def save_drawing(mode):
    global current_points
    if not current_points:
        return "No points to save"
    
    zones = load_zones()
    print(zones)
    
    if mode == "Polygon":
        if len(current_points) < 3:
            return "Polygon needs at least 3 points"
        zones["polygon"] = current_points
    else:
        # It's a line category
        category = mode_map.get(mode, "violation_lines")
        
        # Ensure 'lines_config' exists
        if "lines_config" not in zones:
            zones["lines_config"] = {}
            
        zones["lines_config"][category] = current_points
        # Also save to "lines" flat list for backward compat/default if basic "lines"
        if category == "violation_lines":
             zones["lines"] = current_points
        
    save_zones(zones)
    return f"Saved {len(current_points)} points for {mode}"

def revert_point(image_input):
    global current_points
    if current_points:
        current_points.pop()
    
    # Redraw
    global current_image
    if current_image is None: return None
    img_copy = current_image.copy()
    
    for pt in current_points:
        cv2.circle(img_copy, tuple(pt), 5, (0, 255, 0), -1)
        
    if drawing_mode == "polygon":
         if len(current_points) >= 2:
             pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
             cv2.polylines(img_copy, [pts], len(current_points)>=3, (255, 0, 0), 2)
    else:
         if len(current_points) >= 2:
             for j in range(0, len(current_points)-1, 2):
                cv2.line(img_copy, tuple(current_points[j]), tuple(current_points[j+1]), (0, 0, 255), 2)
    return img_copy

mode_map = {
    "Violation Lines": "violation_lines", 
    "Special Violation Lines": "special_violation_lines",
    "Left Exception Lines": "left_exception_lines",
    "Right Exception Lines": "right_exception_lines",
    "Other Exception Lines": "other_exception_lines"
}

def set_drawing_mode(mode):
    global drawing_mode, current_points
    if mode == "Polygon":
        drawing_mode = "polygon"
    else:
        # Map nice name to internal key if needed, or just use mode
        drawing_mode = mode # We'll handle looking up keys in save/draw
        
    current_points = []
    # Return status and cleared image
    global current_image
    return f"Mode switched to {mode}. Points cleared.", current_image

# --- Settings Logic ---
def update_settings(data_path, vehicle_model, license_model, tracker, conf, fps):
    new_config = system.config.copy()
    
    # Update system config
    if 'system' not in new_config: new_config['system'] = {}
    new_config['system']['data_path'] = data_path
    new_config['system']['vehicle_model'] = vehicle_model
    new_config['system']['license_model'] = license_model
    new_config['system']['tracker'] = tracker
    
    # Update detection/violation config
    new_config['detections']['conf_threshold'] = conf
    new_config['violation']['fps'] = fps
    
    system.update_config(new_config)
    
    # Reload models if changed? 
    # For now, we update config. 
    # Ideally TrafficSystem should detect change and reload, but that's complex. 
    # We'll just save and let user restart or handle it in system.update_config if we want dynamic reload.
    
    save_config(new_config, system.config_path)
    return "Settings Saved. Restart to apply model changes."


# --- UI Construction ---
with gr.Blocks(title="Traffic Violation Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Traffic Violation Detection System")
    
    with gr.Tabs():
        # --- Tab 1: Dashboard ---
        with gr.Tab("Dashboard"):
            gr.Markdown("### System Statistics")
            stats_output = gr.Textbox(label="Status", value=get_dashboard_stats)
            refresh_btn = gr.Button("Refresh Stats")
            refresh_btn.click(get_dashboard_stats, outputs=stats_output)
            
            # Proofs (Placeholder for now as linking to MinIO images needs signed URLs or proxy)
            # gr.Gallery(label="Latest Proofs", value=get_proof_gallery)
            
        # --- Tab 2: Visualization ---
        with gr.Tab("Visualization"):
            with gr.Row():
                start_btn = gr.Button("Start System", variant="primary")
                stop_btn = gr.Button("Stop System", variant="stop")
            
            with gr.Row():
                video_output = gr.Image(label="Live Feed", interactive=False)
                logs_output = gr.Textbox(label="Live Statistics")
            
            start_btn.click(stream_video, outputs=[video_output, logs_output])
            stop_btn.click(stop_system, outputs=[video_output, logs_output])
            
        # --- Tab 3: Zone Drawing ---
        with gr.Tab("Zone Drawing"):
            gr.Markdown("### Interactive Zone Editor")
            with gr.Row():
                with gr.Column(scale=1):
                    # Dropdown for advanced modes
                    drawing_modes = ["Polygon", "Violation Lines", "Special Violation Lines", "Left Exception Lines", "Right Exception Lines", "Other Exception Lines"]
                    mode_dropdown = gr.Dropdown(drawing_modes, label="Drawing Mode", value="Polygon")
                    
                    capture_btn = gr.Button("Capture Frame from Source")
                    save_poly_btn = gr.Button("Save Configuration")
                    revert_btn = gr.Button("Revert Last Point")
                    clear_btn = gr.Button("Clear All Points")
                    draw_status = gr.Textbox(label="Status")
                
                with gr.Column(scale=3):
                    # We need an image to draw on. Let's allow uploading or capturing a frame.
                    canvas = gr.Image(label="Drawing Canvas (Upload or Capture)", interactive=True, type="numpy")

            # Update state when mode changes
            mode_dropdown.change(set_drawing_mode, inputs=mode_dropdown, outputs=[draw_status, canvas])
            
            # Event listeners
            capture_btn.click(capture_frame_for_drawing, outputs=[canvas, draw_status])
            canvas.upload(load_image_for_drawing, inputs=canvas, outputs=canvas)
            canvas.select(on_select, inputs=canvas, outputs=canvas)
            
            revert_btn.click(revert_point, inputs=canvas, outputs=canvas)
            clear_btn.click(clear_points, inputs=canvas, outputs=canvas)
            save_poly_btn.click(save_drawing, inputs=mode_dropdown, outputs=draw_status)
            
        # --- Tab 4: Settings ---
        with gr.Tab("Settings"):
            gr.Markdown("### System Configuration")
            with gr.Row():
                data_path_input = gr.Textbox(label="Data Path / RSTP URL", value=system.config.get('system', {}).get('data_path', 'cam_ai'))
                tracker_input = gr.Dropdown(["sort", "bytetrack"], label="Tracker", value=system.config.get('system', {}).get('tracker', 'bytetrack'))
            
            with gr.Row():
                vehicle_model_input = gr.Textbox(label="Vehicle Model Path", value=system.config.get('system', {}).get('vehicle_model', 'detect_gtvn.pt'))
                license_model_input = gr.Textbox(label="License Plate Model Path", value=system.config.get('system', {}).get('license_model', 'lp_yolo11s.pt'))
            
            with gr.Row():
                conf_slider = gr.Slider(0.0, 1.0, value=system.config['detections']['conf_threshold'], label="Confidence Threshold")
                fps_slider = gr.Slider(1, 60, value=system.config['violation']['fps'], label="FPS")
            
            save_settings_btn = gr.Button("Save Settings")
            settings_status = gr.Textbox(label="Status")
            
            save_settings_btn.click(update_settings, 
                                    inputs=[data_path_input, vehicle_model_input, license_model_input, tracker_input, conf_slider, fps_slider], 
                                    outputs=settings_status)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
