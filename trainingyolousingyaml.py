from ultralytics import YOLO

def train_model():
    """
    Loads a pre-trained YOLOv8s model and fine-tunes it on the custom pothole dataset.
    """
    # Load the pre-trained YOLOv8s model
    model = YOLO('yolov8s.pt')

    # Train the model with adjusted memory settings
    print("Starting model training with optimized memory settings...")
    results = model.train(
        data='Pothole_Dataset/data.yaml',
        epochs=100,
        imgsz=640,
        # --- KEY CHANGES ARE HERE ---
        batch=4,       # Reduced from the default of 16 (or 8) to use less VRAM
        workers=1,     # Drastically reduced from 8 to prevent memory overload. Use 0 or 1.
        # --- END OF CHANGES ---
        name='pothole_detector_yolov8s'
    )
    print("Training complete!")
    print("Your trained model is saved in the 'runs/detect/' directory.")

if __name__ == '__main__':
    # This check is important for multiprocessing on Windows
    train_model()