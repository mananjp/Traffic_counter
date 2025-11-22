# 🚗 YOLO Traffic Counter

A real-time people and vehicle detection system powered by YOLOv8, featuring persistent SQLite3 storage, live webcam detection, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ [Site Link](https://traffic-counter-cv.streamlit.app)

## ✨ Features

### 🎯 Detection Capabilities
- **Real-time Object Detection** using YOLOv8 (state-of-the-art YOLO architecture)
- **Multi-class Detection**: People and vehicles (cars, motorcycles, buses, trucks)
- **Object Tracking**: SORT algorithm for persistent tracking across frames
- **Duplicate Prevention**: Unique ID assignment prevents counting the same object multiple times

### 📹 Input Sources
- **Video Upload**: Process MP4, AVI, MOV, MKV video files
- **Live Webcam**: Real-time detection from webcam feed
- **Multi-camera Support**: Switch between different camera sources

### 💾 Data Management
- **SQLite3 Database**: Persistent storage of all detections
- **Timestamp Tracking**: Every detection recorded with precise timestamps
- **Session Management**: Organize detections by unique session IDs
- **Data Export**: Download data in CSV or JSON formats

### 🎨 User Interface
- **Modern Dashboard**: Clean, responsive Streamlit interface
- **Real-time Metrics**: Live count updates and FPS monitoring
- **Interactive Charts**: Plotly visualizations for analytics
- **Auto-save**: Configurable automatic data saving intervals

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Database Schema](#-database-schema)
- [Configuration](#️-configuration)
- [Features Guide](#-features-guide)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection feature)
- Internet connection (for first-time model download)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/yolo-traffic-counter.git
cd yolo-traffic-counter
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install ultralytics streamlit opencv-python-headless pillow pandas plotly
```

**Alternative**: Install from requirements.txt

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import cv2; from ultralytics import YOLO; print('✅ All dependencies installed successfully!')"
```

## 🎬 Quick Start

### Run the Application

```bash
streamlit run app.py
```

The application will automatically:
1. Download YOLOv8n model (~6MB) on first run
2. Initialize SQLite database (`traffic_data.db`)
3. Open in your default browser at `http://localhost:8501`

### First Detection

1. Navigate to **"📹 Video Detection"** tab
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Click **"🎯 Process Video"**
4. View real-time detections and counts
5. Data automatically saved to database

## 📖 Usage

### Video Detection

Upload and process pre-recorded video files:

1. **Upload Video**: Click "Choose a video file" and select your video
2. **Adjust Settings**: Use sidebar to set confidence threshold and detection classes
3. **Process**: Click "🎯 Process Video" to start detection
4. **Monitor**: Watch real-time annotations and count updates
5. **Review**: Check Analytics tab for insights

### Live Webcam Detection

Real-time detection from webcam:

1. **Navigate**: Go to "🎥 Live Detection" tab
2. **Configure**:
   - Select camera index (usually 0 for default webcam)
   - Adjust frame skip for performance
   - Set auto-save interval
3. **Start**: Check "🟢 Start Webcam"
4. **Monitor**: Watch live detections with FPS counter
5. **Save**: Data auto-saves at intervals, or click "💾 Save Current Counts"

### Analytics Dashboard

View detection history and statistics:

- **Summary Metrics**: Total sessions, people, cars, and averages
- **Time Series Chart**: Detection counts over time
- **Distribution Pie Chart**: People vs. cars ratio
- **Recent Detections Table**: Latest detection records

### Data Export

Download your detection data:

1. Navigate to "💾 Data Export" tab
2. Choose format: CSV or JSON
3. Click download button
4. Preview data in the table

## 🗄️ Database Schema

The SQLite database (`traffic_data.db`) uses the following schema:

| Column        | Type                | Description                    |
|---------------|---------------------|--------------------------------|
| id            | INTEGER PRIMARY KEY | Auto-incrementing record ID    |
| timestamp     | TEXT                | Date and time of detection     |
| people_count  | INTEGER             | Number of people detected      |
| car_count     | INTEGER             | Number of cars detected        |
| session_id    | TEXT                | Unique session identifier      |

### Sample Query

```sql
-- Get total detections by session
SELECT 
    session_id,
    SUM(people_count) as total_people,
    SUM(car_count) as total_cars,
    COUNT(*) as detections
FROM detections
GROUP BY session_id;
```

## ⚙️ Configuration

### Sidebar Settings

**Confidence Threshold** (0.0 - 1.0)
- Default: 0.5
- Lower values: More detections (may include false positives)
- Higher values: Fewer, more confident detections

**Detection Classes**
- ✅ Detect People (COCO class ID: 0)
- ✅ Detect Cars (COCO class IDs: 2, 3, 5, 7)
  - 2: Car
  - 3: Motorcycle
  - 5: Bus
  - 7: Truck

**Session Management**
- Click "🔄 New Session" to start fresh session
- Each session gets unique UUID
- Click "🗑️ Clear Database" to reset all data (⚠️ permanent)

### Live Detection Settings

**Camera Index**: Select camera source (0-5)
- 0: Default webcam
- 1+: External cameras

**Frame Skip**: Process every Nth frame (1-10)
- Lower: More accurate, slower
- Higher: Faster, may miss objects

**Auto-save Interval**: Seconds between saves (5-300)
- Recommended: 30 seconds

## 📚 Features Guide

### Object Tracking Algorithm

The system uses SORT (Simple Online and Realtime Tracking):

```python
# Persistent tracking across frames
results = model.track(frame, persist=True, conf=threshold)

# Each object gets unique ID
track_ids = boxes.id.cpu().numpy()

# Prevents duplicate counting
tracked_ids['people'].add(int(track_id))
```

### Performance Monitoring

Real-time FPS calculation:

- Uses rolling average over 30 frames
- Displayed on live video feed
- Helps optimize frame skip settings

### Auto-save Mechanism

```python
# Saves automatically at intervals
if current_time - last_save >= interval:
    save_detection(conn, people_count, car_count, session_id)
    show_toast("💾 Auto-saved to database")
```

## 🔧 Troubleshooting

### Common Issues

**Camera Won't Start**
```
❌ Cannot access camera 0
```
**Solution**: Try different camera index (1, 2) or check camera permissions

**Low FPS**
```
FPS: 5.2
```
**Solution**: Increase frame skip value or use lighter model (yolov8n)

**Module Not Found**
```
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**: Reinstall dependencies
```bash
pip install --upgrade ultralytics streamlit opencv-python-headless
```

**Video Upload Error (Windows)**
```
PermissionError: [WinError 32]
```
**Solution**: Already fixed in latest version - temporary file handling improved

### Model Download Issues

If YOLOv8 model download fails:

```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Or use Python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

## ⚡ Performance Optimization

### Model Selection

| Model    | Size  | Speed      | Accuracy |
|----------|-------|------------|----------|
| yolov8n  | 6MB   | Fastest    | Good     |
| yolov8s  | 22MB  | Fast       | Better   |
| yolov8m  | 52MB  | Medium     | Great    |
| yolov8l  | 87MB  | Slow       | Excellent|
| yolov8x  | 137MB | Slowest    | Best     |

**To change model**:
```python
# In app.py, modify:
model = YOLO('yolov8s.pt')  # Use yolov8s instead
```

### Frame Processing Tips

**For Video Files**:
- Process every 5th frame (current default)
- Adjust in code: `if frame_count % 5 == 0`

**For Live Detection**:
- Use frame skip slider (3-5 recommended)
- Lower resolution if needed
- Close other applications

### Database Optimization

For large datasets:

```sql
-- Add index for faster queries
CREATE INDEX idx_timestamp ON detections(timestamp);
CREATE INDEX idx_session ON detections(session_id);
```

## 🛠️ Tech Stack

| Technology    | Purpose                          |
|---------------|----------------------------------|
| **Python**    | Core programming language        |
| **YOLOv8**    | Object detection model           |
| **Streamlit** | Web dashboard framework          |
| **SQLite3**   | Database for persistent storage  |
| **OpenCV**    | Video processing and webcam      |
| **Pandas**    | Data manipulation and analysis   |
| **Plotly**    | Interactive visualizations       |
| **Pillow**    | Image processing utilities       |

## 📁 Project Structure

```
yolo-traffic-counter/
├── app.py                 # Main Streamlit application
├── traffic_data.db        # SQLite database (auto-generated)
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore file
└── yolov8n.pt            # YOLO model (auto-downloaded)
```

## 🎯 Use Cases

- **Traffic Monitoring**: Count vehicles at intersections
- **Parking Management**: Track available parking spaces
- **Security**: Monitor people in restricted areas
- **Retail Analytics**: Count customers entering stores
- **Event Management**: Track attendance and crowd density
- **Smart Cities**: Urban planning and traffic optimization

## 🔮 Future Enhancements

- [ ] Zone-based counting (entry/exit lines)
- [ ] Heat map visualization
- [ ] Multi-stream support (multiple cameras)
- [ ] Cloud database integration (PostgreSQL, MongoDB)
- [ ] Email/SMS alerts for threshold breaches
- [ ] Historical trend analysis
- [ ] Model fine-tuning on custom datasets
- [ ] Docker containerization

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [SORT Algorithm](https://github.com/abewley/sort) - Object tracking

## 📧 Contact

Your Name - Manan Panchal

Project Link: [https://github.com/yourusername/yolo-traffic-counter](https://github.com/yourusername/yolo-traffic-counter)

---

**⭐ If you find this project helpful, please give it a star!**

Made with ❤️ using YOLOv8 and Streamlit
