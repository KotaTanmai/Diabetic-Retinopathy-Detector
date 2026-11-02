
import os
import uuid
import numpy as np

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained CNN model
model = load_model("model/retinopathy_model.keras")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    image_path = None
    error = None

    if request.method == "POST":

        if "file" not in request.files:
            error = "Please select an image."

        else:
            file = request.files["file"]

            if file.filename == "":
                error = "Please select an image."

            elif not allowed_file(file.filename):
                error = "Please upload a PNG, JPG or JPEG image."

            else:
                filename = secure_filename(file.filename)
                unique_name = str(uuid.uuid4()) + "_" + filename

                save_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_name
                )

                file.save(save_path)

                try:
                    # Preprocess image
                    image = load_img(
                        save_path,
                        target_size=(224, 224)
                    )

                    image = img_to_array(image) / 255.0
                    image = np.expand_dims(image, axis=0)

                    # Predict
                    prediction = float(
                        model.predict(image, verbose=0)[0][0]
                    )

                    if prediction >= 0.5:
                        result = "DR Detected"
                        confidence = prediction * 100
                    else:
                        result = "No DR"
                        confidence = (1 - prediction) * 100

                    image_path = "/" + save_path.replace("\\", "/")

                except Exception as e:
                    error = "Could not process this image. Please try another image."
                    print("Prediction error:", e)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
