# Iris Prediction Application

This is a FastAPI application that predicts the species of an Iris flower based on its sepal length, sepal width, petal length, and petal width. It utilizes machine learning to provide predictions along with the probabilities of each Iris species.

## Demo Link for application

url - [https://iris-fastapi-meer.onrender.com/](https://iris-fastapi-meer.onrender.com/)

## Features
- User-friendly web interface to input Iris flower measurements.
- Predictions of Iris species (Setosa, Versicolor, or Virginica).
- Displays probabilities for each species.

## Requirements
To run this application, you need the following dependencies:

- Python 3.12.4 
- FastAPI
- Jinja2
- Uvicorn
- pydantic
- scikit-learn

You can install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. Set Up Python Virtual environment

   ```bash
   python -m venv env
   ```

3. Activaate virtual environment

   ```bash
   source env/bin/activate # env\Script\activate.bat for Windows machine
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   python main.py
   ```

6. Open your web browser and go to `http://localhost:8000`. You will see a form where you can input the sepal and petal measurements for the Iris flower.

7. After submitting the form, the application will display the predicted species along with the probabilities for each species.

## Endpoints
### `GET /`
Renders the form where users can input the measurements for the Iris flower.

### `POST /predict`
Handles form submission and returns the predicted Iris species along with the probabilities of each species.

## Example

Here's an example of how to use the web interface:

1. Input the following measurements into the form:
   - Sepal length: 5.1
   - Sepal width: 3.5
   - Petal length: 1.4
   - Petal width: 0.2

2. Click the submit button.

3. The application will display the predicted species along with the probabilities of each species.

## Dockerhub Link

Dockerhub link - [https://hub.docker.com/r/utkarshg1/fastapi-iris](https://hub.docker.com/r/utkarshg1/fastapi-iris)

## License
This project is licensed under the Apache License 2.0, See the LICENSE file for more details.

## Acknowledgments
- This project is built using [FastAPI](https://fastapi.tiangolo.com/).
- The machine learning model used for predictions is powered by [scikit-learn](https://scikit-learn.org/).
