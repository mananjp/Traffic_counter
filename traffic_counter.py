import streamlit as st
import cv2
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import uuid
import time
from collections import deque

# Page configuration
st.set_page_config(
    page_title="YOLO Traffic Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Database initialization
def init_database():
    # Use a relative path for better deployment compatibility
    db_path = os.path.join(os.getcwd(), 'traffic_data.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            people_count INTEGER DEFAULT 0,
            car_count INTEGER DEFAULT 0,
            session_id TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn


# Save detection data
def save_detection(conn, people_count, car_count, session_id):
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO detections (timestamp, people_count, car_count, session_id)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, people_count, car_count, session_id))
    conn.commit()


# Load model with caching
@st.cache_resource
def load_model():
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Please ensure the model is downloaded.")
        st.stop()
    return YOLO(model_path)


# Initialize session state
if 'conn' not in st.session_state:
    st.session_state.conn = init_database()
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

model = load_model()
conn = st.session_state.conn

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    st.divider()

    st.subheader("📊 Detection Classes")
    detect_people = st.checkbox("Detect People", value=True)
    detect_cars = st.checkbox("Detect Cars", value=True)

    st.divider()

    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.success("New session started!")

    st.divider()

    if st.button("🗑️ Clear Database"):
        c = conn.cursor()
        c.execute("DELETE FROM detections")
        conn.commit()
        st.success("Database cleared!")

# Main app
st.title("🚗 YOLO Traffic Counter")
st.markdown("Real-time people and vehicle detection with persistent storage")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Video Detection", "🎥 Live Detection", "📊 Analytics", "💾 Data Export"])

with tab1:
    st.subheader("Upload Video for Detection")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv']
    )

    col1, col2 = st.columns(2)

    with col1:
        process_button = st.button("🎯 Process Video", type="primary", use_container_width=True)

    if uploaded_file and process_button:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())

        # Placeholders
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()
        progress_bar = st.progress(0)

        # Open video
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # COCO classes
        PERSON_CLASS_ID = 0
        CAR_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        tracked_ids = {'people': set(), 'cars': set()}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection
            results = model.track(frame, persist=True, conf=confidence_threshold)

            # Process results
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy()
                track_ids = boxes.id.cpu().numpy()

                # Count people
                if detect_people:
                    for i, cls_id in enumerate(class_ids):
                        if int(cls_id) == PERSON_CLASS_ID:
                            tracked_ids['people'].add(int(track_ids[i]))

                # Count cars
                if detect_cars:
                    for i, cls_id in enumerate(class_ids):
                        if int(cls_id) in CAR_CLASS_IDS:
                            tracked_ids['cars'].add(int(track_ids[i]))

            # Annotate frame
            annotated_frame = results[0].plot()

            # Display every 5th frame to reduce lag
            if frame_count % 5 == 0:
                video_placeholder.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("👥 People Detected", len(tracked_ids['people']))
                    col2.metric("🚗 Cars Detected", len(tracked_ids['cars']))
                    col3.metric("📊 Total Objects",
                                len(tracked_ids['people']) + len(tracked_ids['cars']))

                progress_bar.progress(frame_count / total_frames)

        cap.release()

        tfile.close()

        import time

        time.sleep(0.1)

        try:
            os.unlink(tfile.name)
        except PermissionError:
            pass

        # Save to database
        save_detection(
            conn,
            len(tracked_ids['people']),
            len(tracked_ids['cars']),
            st.session_state.session_id
        )

        st.success("✅ Video processing complete! Data saved to database.")

with tab2:
    st.subheader("🎥 Live Webcam Detection")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.info("💡 **Live Detection Controls**")

        run_webcam = st.checkbox("🟢 Start Webcam", key="webcam_running")

        st.divider()

        camera_index = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=5,
            value=0,
            help="Usually 0 for default webcam, try 1 or 2 if not working"
        )

        st.divider()

        frame_skip = st.slider(
            "Frame Skip (for performance)",
            min_value=1,
            max_value=10,
            value=3,
            help="Process every Nth frame to improve performance"
        )

        st.divider()

        auto_save_interval = st.number_input(
            "Auto-save interval (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            help="Automatically save counts to database"
        )

        if st.button("💾 Save Current Counts", use_container_width=True):
            if 'live_tracked_ids' in st.session_state:
                save_detection(
                    conn,
                    len(st.session_state.live_tracked_ids['people']),
                    len(st.session_state.live_tracked_ids['cars']),
                    st.session_state.session_id
                )
                st.success("✅ Counts saved!")
            else:
                st.warning("No active detection session")

        if st.button("🔄 Reset Counts", use_container_width=True):
            if 'live_tracked_ids' in st.session_state:
                st.session_state.live_tracked_ids = {'people': set(), 'cars': set()}
                st.success("✅ Counts reset!")

    with col1:
        video_frame_placeholder = st.empty()
        live_metrics_placeholder = st.empty()

        if 'live_tracked_ids' not in st.session_state:
            st.session_state.live_tracked_ids = {'people': set(), 'cars': set()}
        if 'last_save_time' not in st.session_state:
            st.session_state.last_save_time = time.time()

        PERSON_CLASS_ID = 0
        CAR_CLASS_IDS = [2, 3, 5, 7]

        if run_webcam:
            # Check if we're in a cloud environment
            is_cloud = (os.environ.get('NETLIFY') or os.environ.get('HEROKU') or 
                       os.environ.get('STREAMLIT_SHARING') or os.environ.get('STREAMLIT_CLOUD') or
                       os.environ.get('HOSTNAME', '').startswith('streamlit') or
                       'streamlit.app' in os.environ.get('SERVER_NAME', ''))
            
            if is_cloud:
                st.warning("⚠️ Webcam access is not available in cloud deployments. Please use video upload instead.")
                st.info("This limitation exists because cloud servers don't have access to your local camera.")
                run_webcam = False
            else:
                cap = cv2.VideoCapture(camera_index)
                
                if not cap.isOpened():
                    st.error(f"❌ Cannot access camera {camera_index}. Please check your camera or try a different index.")
                else:
                    st.success(f"✅ Camera {camera_index} connected successfully!")

                    frame_count = 0
                    fps_queue = deque(maxlen=30)

                    while run_webcam:
                        start_time = time.time()

                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to grab frame from camera")
                            break

                    frame_count += 1

                    if frame_count % frame_skip == 0:
                        results = model.track(frame, persist=True, conf=confidence_threshold, verbose=False)

                        if results[0].boxes is not None and results[0].boxes.id is not None:
                            boxes = results[0].boxes
                            class_ids = boxes.cls.cpu().numpy()
                            track_ids = boxes.id.cpu().numpy()

                            if detect_people:
                                for i, cls_id in enumerate(class_ids):
                                    if int(cls_id) == PERSON_CLASS_ID:
                                        st.session_state.live_tracked_ids['people'].add(int(track_ids[i]))

                            if detect_cars:
                                for i, cls_id in enumerate(class_ids):
                                    if int(cls_id) in CAR_CLASS_IDS:
                                        st.session_state.live_tracked_ids['cars'].add(int(track_ids[i]))

                        annotated_frame = results[0].plot()
                    else:
                        annotated_frame = frame

                    end_time = time.time()
                    fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                    fps_queue.append(fps)
                    avg_fps = sum(fps_queue) / len(fps_queue)

                    cv2.putText(
                        annotated_frame,
                        f"FPS: {avg_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                    video_frame_placeholder.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                    with live_metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("👥 People", len(st.session_state.live_tracked_ids['people']))
                        col2.metric("🚗 Cars", len(st.session_state.live_tracked_ids['cars']))
                        col3.metric("📊 Total",
                                    len(st.session_state.live_tracked_ids['people']) +
                                    len(st.session_state.live_tracked_ids['cars']))
                        col4.metric("⚡ FPS", f"{avg_fps:.1f}")

                    current_time = time.time()
                    if current_time - st.session_state.last_save_time >= auto_save_interval:
                        save_detection(
                            conn,
                            len(st.session_state.live_tracked_ids['people']),
                            len(st.session_state.live_tracked_ids['cars']),
                            st.session_state.session_id
                        )
                        st.session_state.last_save_time = current_time
                        st.toast("💾 Auto-saved to database", icon="✅")

                    time.sleep(0.01)

                cap.release()
                st.info("📷 Camera stopped")
        else:
            st.info("👆 Check **'Start Webcam'** above to begin live detection")

with tab3:
    st.subheader("📊 Detection Analytics")

    # Load data
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp DESC", conn)

    if not df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Sessions",
                df['session_id'].nunique(),
                delta=None
            )

        with col2:
            st.metric(
                "Total People",
                df['people_count'].sum(),
                delta=None
            )

        with col3:
            st.metric(
                "Total Cars",
                df['car_count'].sum(),
                delta=None
            )

        with col4:
            avg_total = (df['people_count'] + df['car_count']).mean()
            st.metric(
                "Avg Objects/Session",
                f"{avg_total:.1f}",
                delta=None
            )

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Time series
            fig_time = px.line(
                df,
                x='timestamp',
                y=['people_count', 'car_count'],
                title='Detections Over Time',
                labels={'value': 'Count', 'timestamp': 'Time'},
                template='plotly_white'
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with col2:
            # Pie chart
            total_people = df['people_count'].sum()
            total_cars = df['car_count'].sum()

            fig_pie = go.Figure(data=[go.Pie(
                labels=['People', 'Cars'],
                values=[total_people, total_cars],
                hole=.3
            )])
            fig_pie.update_layout(title='Total Detections Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Recent detections table
        st.subheader("Recent Detections")
        st.dataframe(
            df[['timestamp', 'people_count', 'car_count', 'session_id']].head(10),
            use_container_width=True
        )
    else:
        st.info("No detection data available yet. Process a video or use live detection to see analytics!")

with tab4:
    st.subheader("💾 Export Data")

    df = pd.read_sql_query("SELECT * FROM detections", conn)

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f'traffic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col2:
            st.download_button(
                label="📥 Download JSON",
                data=df.to_json(orient='records', indent=2),
                file_name=f'traffic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
                use_container_width=True
            )

        st.divider()
        st.subheader("Preview Data")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data to export. Process videos to generate detection data.")