from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.prediction import get_input_data, predict_results
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("iris_prediction.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...),
):
    # Predict results with probabilities
    iris_data = get_input_data(sepal_length, sepal_width, petal_length, petal_width)
    results = predict_results(sepal_length, sepal_width, petal_length, petal_width)
    label = results.prediction
    probs = results.probability

    # Return results
    return templates.TemplateResponse(
        "iris_prediction.html",
        {
            "request": request,
            "prediction": label,
            "probabilities": probs,
            "iris_data": iris_data,
        },
    )


if __name__ == "__main__":
    # Running uvicorn programmatically in production
    uvicorn.run(
        "main:app",               # The FastAPI app instance
        host="0.0.0.0",           # Bind to all interfaces
        port=8000,                # Port number for the application
        log_level="info",         # Log level
        workers=1,                # Number of workers 
        lifespan="on"
    )
