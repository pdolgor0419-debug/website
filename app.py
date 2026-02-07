from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ======= Моделуудыг ачааллах =======
model = joblib.load("model/model.pkl")
le_duureg = joblib.load("model/duureg.pkl")
le_bairlal = joblib.load("model/bairlal.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    duuregs = le_duureg.classes_.tolist()
    bairlals = le_bairlal.classes_.tolist()

    if request.method == "POST":
        # Хэрэглэгчийн оруулсан утгуудыг авах
        year = int(request.form["year"])
        total_floors = int(request.form["total_floors"])
        area = float(request.form["area"])
        floor = int(request.form["floor"])
        windows = int(request.form["windows"])
        duureg = request.form["duureg"]
        bairlal = request.form["bairlal"]

        # LabelEncoder ашиглан кодлох
        duureg_encoded = le_duureg.transform([duureg])[0]
        bairlal_encoded = le_bairlal.transform([bairlal])[0]

        # Prediction dataframe бэлдэх
        input_data = pd.DataFrame({
            "Ашиглалтанд орсон он": [year],
            "Барилгын давхар": [total_floors],
            "Талбай": [area],
            "Хэдэн давхарт": [floor],
            "Дүүрэг": [duureg_encoded],
            "Цонхны тоо": [windows],
            "Байрлал": [bairlal_encoded]
        })

        # Үнэ таамаглах
        predicted_price = model.predict(input_data)[0]
        prediction = f"{predicted_price:,.0f} ₮"

    return render_template("index.html", prediction=prediction, duuregs=duuregs, bairlals=bairlals)

if __name__ == "__main__":
    app.run(debug=True)
