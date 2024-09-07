# Vehicles Real Time Detection

This repository contains a Streamlit app for real-time object detection using customized YOLO models. The project is organized as follows:

## Repository Structure

```
my_project/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
├── vlens/
├── notebooks/
├── data/
├── tests/
│   ├── input/
│   ├── results/
└── runs/
```

- `vlens/`: Contains the virtual environment for the project.
- `runs/`: Stores the customized YOLO models for bus, car, and truck classes.
- `data/`: Includes the customized dataset.
- `src/`: Contains source code files.
- `notebooks/`: Jupyter notebooks used for experimentation and analysis.
- `tests/`: Directory for testing, including input and result folders.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/amrabdalbadee/Vlens.git
   cd Vlens
   ```

2. **Create and Activate the Virtual Environment**

   If you're using the provided virtual environment (`vlens`), activate it:

   ```bash
   source vlens/bin/activate
   ```

   Otherwise, create and activate a new virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   Install the required libraries using `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Streamlit App

The project includes two Streamlit apps:

1. **app.py**: Uses the YOLOv8 model (saved as `.pt` file).
2. **optimized_app.py**: Uses the optimized ONNX model.

To run the Streamlit app:

1. **Run the YOLOv8 Model App**

   ```bash
   streamlit run streamlit/app.py
   ```

2. **Run the ONNX Model App**

   ```bash
   streamlit run  streamlit/optimized_app.py
   ```

   Make sure the `.onnx` model is available in the appropriate directory.

## Model Files

- YOLOv8 model file (`best.pt`) is located at: `Vlens/runs/detect/custom_yolov8/weights/best.pt`
- Ensure that the model file is correctly referenced in `app.py` and `optimized_app.py`.

## Data

The customized data used for training is located in the `data/` directory. Ensure your data is in the expected format.

## Contribution

Feel free to fork the repository and submit pull requests. For any issues or suggestions, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any details to better fit your project!