from modal import Image, Stub, gpu, asgi_app
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from leffa_app.leffa_wrapper import LeffaPredictor

stub = Stub("leffa-api")

app_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy", "Pillow", "huggingface_hub", "fastapi", "uvicorn", "torch", "torchvision", "onnxruntime", "opencv-python", "scipy")
    .run_commands(
        "git clone https://github.com/AhmedAbubaker98/Leffa-mod /root/leffa_app",
        "cd /root/leffa_app && mkdir ckpts && huggingface-cli download franciszzj/Leffa --local-dir ./ckpts --repo-type model",
    )
)

app = FastAPI()

@stub.function(image=app_image, gpu=gpu.A100(), timeout=600)
@asgi_app()
def fastapi_app():
    predictor = LeffaPredictor()

    @app.post("/virtual-tryon/")
    async def virtual_tryon(person_image: UploadFile = File(...), cloth_image: UploadFile = File(...)):
        temp_dir = f"/tmp/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        person_path = os.path.join(temp_dir, "person.png")
        cloth_path = os.path.join(temp_dir, "cloth.png")
        output_path = os.path.join(temp_dir, "result.png")

        with open(person_path, "wb") as f:
            shutil.copyfileobj(person_image.file, f)
        with open(cloth_path, "wb") as f:
            shutil.copyfileobj(cloth_image.file, f)

        image, _, _ = predictor.leffa_predict_vt(
            src_image_path=person_path,
            ref_image_path=cloth_path,
            ref_acceleration=False,
            step=30,
            scale=2.5,
            seed=42,
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=False,
            preprocess_garment=True
        )

        Image.fromarray(image).save(output_path)
        return FileResponse(output_path, media_type="image/png")

    return app
