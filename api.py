import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AIRequest(BaseModel):
    calib_file: str
    input_dir: str
    output_dir: str
    config_folder: str
    config_filename: str
    # add adjustable config


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/run-model")
def run_model(request: AIRequest):
    """Triggers LeapVO model processing."""
    try:

        print('Running LEAP-VO with the following parameters:')
        print(f'    input dir: {request.input_dir}')
        print(f'    output dir: {request.output_dir}')
        print(f'    calib file: {request.calib_file}')
        print(f'    config file: {request.config_folder}/{request.config_filename}')

        command = [
            "python", "main/eval.py",
            f"--config-path={request.config_folder}",
            f"--config-name={request.config_filename}",
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
