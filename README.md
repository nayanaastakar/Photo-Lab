# PhotoLab

PhotoLab is a small, local image editor built with Streamlit. It lets you upload an image, apply common adjustments and filters, preview changes side-by-side, analyze pixel differences, and download the result.

## Features
- Brightness, contrast, sharpness, and saturation adjustments
- Blur and vignette effects
- Grayscale, sepia and invert filters
- Rotation and flip transforms
- Side-by-side original vs processed preview
- Change analysis: average change, max change, and percent pixels affected

## Requirements
- Python 3.8+
- See `requirements.txt` for the minimal Python packages (`streamlit`, `numpy`, `pillow`).

## Quick start (local)

1. Clone the repo:

```powershell
git clone https://github.com/nayanaastakar/Photo-Lab.git
cd Photo-Lab
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py --server.port 8503
```

Then open http://localhost:8503 in your browser.

## Docker

Build and run the container (requires Docker installed):

```bash
docker build -t photolab:latest .
docker run -p 8501:8501 photolab:latest
```

The container starts Streamlit on port `8501` inside the container.

## Included files
- Main app: [Photo-Lab/app.py](app.py)
- Image utilities: [Photo-Lab/image_processing.py](image_processing.py)
- Dependencies: [Photo-Lab/requirements.txt](requirements.txt)
- Convenience script: `run_app.ps1` (Windows)
- Dockerfile and `LICENSE`

## Troubleshooting
- If Streamlit fails to start, confirm dependencies are installed and check any error messages printed in the terminal.
- On Windows, run `.
un_app.ps1` from PowerShell (you may need to change execution policy or run as Administrator).

## Contributing
- This is a small demo project — feel free to open issues or PRs to add features or fixes.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

