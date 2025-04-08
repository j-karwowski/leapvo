import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AIRequest(BaseModel):
    input_dir: str
    # calib_file: str
    output_dir: str
    # add adjustable config


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/run-model")
def run_model(request: AIRequest):
    """Triggers LeapVO model processing."""
    try:
        command = [
            "python", "main/eval.py",
            "--config-path=../configs",
            "--config-name=demo",
            f"data.imagedir={request.input_dir}",
            f"data.calib=calibs/demo_calib.txt",
            f"data.savedir={request.output_dir}",
            "save_trajectory=true",
            "save_video=true",
            "save_plot=true"
        ]
        subprocess.run(command, check=True)
        return {"status": "success", "message": "AI model executed successfully."}
    
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}

# To run: `uvicorn api:app --host 0.0.0.0 --port 8000`
