import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AIRequest(BaseModel):
    calib_file: str
    input_dir: str
    output_dir: str
    config_folder: str
    config_file: str
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
            "--config-path={request.config_folder}",
            "--config-name={request.config_file}",
            f"data.imagedir={request.input_dir}",
            # f"data.calib=calibs/demo_calib.txt",
            f"data.calib={request.calib_file}",
            f"data.savedir={request.output_dir}",
            "save_trajectory=true",
            "save_video=true",
            "save_plot=true"
        ]
        subprocess.run(command, check=True)
        return {"status": "success", "message": "AI model executed successfully."}
    
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}
