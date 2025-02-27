# Floating Waste Detection System

A web application that detects and tracks floating waste in video footage using YOLOv8 and ByteTrack.

## Features

- Upload and process video files
- Real-time waste detection and tracking
- Interactive dashboard with statistics
- Detailed analysis of waste categories
- Beautiful visualizations

## Technologies Used

- Backend: Flask
- Frontend: Streamlit
- Computer Vision: YOLOv8, OpenCV, ByteTrack
- Data Visualization: Plotly
- Data Processing: Pandas, NumPy

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask backend:
```bash
python app.py
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and go to `http://localhost:8501`

## Project Structure

```
├── app.py              # Flask backend
├── streamlit_app.py    # Streamlit frontend
├── requirements.txt    # Project dependencies
├── static/            
│   └── uploads/       # Directory for uploaded videos
└── templates/         # HTML templates
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 