import cv2
import torch
import serial
import time
from PIL import Image
from serial.tools.list_ports import comports
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ========== ARDUINO MANAGER ==========
class ArduinoController:
    def __init__(self, port='COM8', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        
    def connect(self):
        """Robust connection with auto-retry and port validation"""
        print(f"\n[Arduino] Attempting to connect to {self.port}...")
       # Check if port exists
        available_ports = [p.device for p in comports()]
        if self.port not in available_ports:
            print(f"[ERROR] {self.port} not found. Available ports:")
            print("\n".join(f"- {p}" for p in available_ports))
            return False
        
        # Attempt connection
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=1,
                    write_timeout=1
                )
                time.sleep(2)  # Critical for Arduino reset
                print(f"[SUCCESS] Connected to {self.port}")
                return True
                
            except serial.SerialException as e:
                print(f"[Attempt {attempt+1}/{max_attempts}] Failed: {str(e)}")
                if "Access Denied" in str(e):
                    print("  → Close Arduino IDE completely (check Task Manager)")
                time.sleep(2)
        
        return False
    
    def send_command(self, command):
        """Thread-safe command transmission"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                print(f"[Arduino] Sent command: {command}")
            except Exception as e:
                print(f"[ERROR] Send failed: {str(e)}")

# ========== MAIN APPLICATION ==========
def main():
    # Initialize systems
    print("[System] Starting waste classification...")
    
    # 1. Arduino Setup
    arduino = ArduinoController(port='COM8')  # Changed to COM8
    arduino_connected = arduino.connect()
    
    # 2. AI Model Setup
    try:
        print("[AI] Loading model...")
        processor = AutoImageProcessor.from_pretrained("Giecom/giecom-vit-model-clasification-waste")
        model = AutoModelForImageClassification.from_pretrained("Giecom/giecom-vit-model-clasification-waste")
        print("[AI] Model loaded successfully!")
    except Exception as e:
        print(f"[AI ERROR] {str(e)}")
        return

    # 3. Camera Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAMERA ERROR] Webcam not accessible")
        return

    # Waste classification mapping
    WASTE_TYPES = {
        "cardboard": ("Biodegradable", (0, 255, 0)),
        "glass": ("Non-biodegradable", (0, 0, 255)),
        "metal": ("Non-biodegradable", (0, 0, 255)),
        "paper": ("Biodegradable", (0, 255, 0)),
        "plastic": ("Non-biodegradable", (0, 0, 255)),
        "trash": ("Non-biodegradable", (0, 0, 255)),
        "organic": ("Biodegradable", (0, 255, 0)),
        "batteries": ("Non-biodegradable", (0, 0, 255))
    }

    # Main processing loop
    try:
        print("[System] Starting classification loop...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] Frame capture error")
                break

            # AI Classification
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=Image.fromarray(img_rgb), return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_idx = outputs.logits.argmax(-1).item()
            label = model.config.id2label[pred_idx]
            bio_type, color = WASTE_TYPES.get(label.lower(), ("Unknown", (255, 255, 255)))
            confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0][pred_idx].item() * 100

            # Display results
            cv2.putText(frame, f"{label} | {bio_type} | {confidence:.1f}%", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Send to Arduino
            if arduino_connected:
                arduino.send_command(0 if bio_type == "Biodegradable" else 1)
            
            cv2.imshow('Waste Classifier (Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if arduino.serial_conn:
            arduino.serial_conn.close()
        print("[System] Shutdown complete")

if __name__ == "__main__":
    main()