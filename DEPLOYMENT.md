# 🚀 YOLO Traffic Counter - Netlify Deployment Guide

This guide will help you deploy the YOLO Traffic Counter application to Netlify.

## 📋 Prerequisites

- A GitHub account
- A Netlify account (free tier works fine)
- Git installed on your local machine

## 🔧 Deployment Files

The following files have been added for Netlify deployment:

- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `netlify.toml` - Netlify build configuration
- `setup.py` - Build script for model download
- `.streamlit/config.toml` - Streamlit configuration

## 🚀 Step-by-Step Deployment

### 1. Push to GitHub

First, make sure your code is pushed to a GitHub repository:

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Add Netlify deployment configuration"

# Add your GitHub repository as remote
git remote add origin https://github.com/yourusername/traffic-counter.git

# Push to GitHub
git push -u origin main
```

### 2. Connect to Netlify

1. Go to [netlify.com](https://netlify.com) and sign up/log in
2. Click "New site from Git"
3. Choose "GitHub" as your Git provider
4. Select your traffic counter repository
5. Configure build settings:
   - **Build command**: `pip install -r requirements.txt && python setup.py`
   - **Publish directory**: `.`
   - **Base directory**: (leave empty)

### 3. Environment Variables (Optional)

In Netlify dashboard, go to Site settings > Environment variables and add:

```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_PORT=8501
```

### 4. Deploy

Click "Deploy site" and wait for the build to complete.

## ⚠️ Important Notes

### Limitations in Cloud Deployment

1. **Webcam Access**: Live webcam detection won't work in cloud environments due to security restrictions. Only video upload functionality will be available.

2. **File Storage**: The SQLite database is ephemeral - data will be lost when the container restarts. Consider upgrading to a persistent database for production use.

3. **Performance**: YOLO model inference may be slower on Netlify's free tier due to limited computational resources.

### File Size Considerations

- The YOLOv8 model (`yolov8n.pt`) is approximately 6MB
- Ensure your Git LFS is configured if the repository becomes large
- The model will be downloaded during build time to avoid repository bloat

## 🔧 Troubleshooting

### Common Issues

1. **Build Timeout**: If the build times out, try using a smaller model or optimize the requirements.txt

2. **Memory Issues**: The app requires sufficient memory for YOLO model loading. Consider using Netlify Pro for better resources.

3. **OpenCV Issues**: We use `opencv-python-headless` which is optimized for server environments.

### Build Logs

Check Netlify build logs for specific error messages:
- Go to your site dashboard
- Click on "Deploys"
- Click on the failed deploy to see logs

## 🛠️ Alternative Deployment Options

If Netlify doesn't work well for this resource-intensive app, consider:

1. **Streamlit Cloud**: Native Streamlit hosting (recommended for Streamlit apps)
2. **Heroku**: Better for apps requiring more computational resources
3. **Railway**: Good alternative with generous free tier
4. **Google Cloud Run**: For production deployments

## 📞 Support

If you encounter issues:

1. Check the build logs in Netlify dashboard
2. Verify all files are committed and pushed to GitHub
3. Ensure the model file is being downloaded correctly
4. Check that all dependencies are listed in requirements.txt

## 🔄 Updating the Deployment

To update your deployment:

1. Make changes to your local code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update application"
   git push
   ```
3. Netlify will automatically rebuild and deploy

## 🌟 Features Available in Cloud Deployment

✅ Video upload and processing
✅ Real-time detection visualization  
✅ Analytics and data export
✅ Session management
✅ SQLite database (ephemeral)

❌ Live webcam detection (security limitation)
❌ Persistent data storage (use external database for production)

---

For more information about the application features, see the main [README.md](README.md).