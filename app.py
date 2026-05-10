from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import re

from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)
app.secret_key = "replace_this_with_a_strong_secret"

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

DB_PATH = "users.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def add_user(username, phone, email, password):
    pw_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, phone, email, password_hash) VALUES (?, ?, ?, ?)",
            (username, phone, email, pw_hash)
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, username, phone, email, password_hash FROM users WHERE email = ?",
        (email,)
    )
    row = c.fetchone()
    conn.close()
    return row



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not phone or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if not re.fullmatch(r"\d{10}", phone):
            flash("Phone must be 10 digits.", "error")
            return redirect(url_for("signup"))

        if not email.endswith("@gmail.com"):
            flash("Use Gmail only.", "error")
            return redirect(url_for("signup"))

        success, err = add_user(username, phone, email, password)
        if not success:
            flash(f"Error: {err}", "error")
            return redirect(url_for("signup"))

        flash("Signup successful!", "success")
        return redirect(url_for("signin"))

    return render_template("signup.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user_by_email(email)
        if not user:
            flash("User not found.", "error")
            return redirect(url_for("signup"))

        user_id, username, _, _, pw_hash = user

        if not check_password_hash(pw_hash, password):
            flash("Wrong password.", "error")
            return redirect(url_for("signin"))

        session["user_id"] = user_id
        session["username"] = username

        return redirect(url_for("home"))

    return render_template("signin.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/home")
def home():
    if "user_id" not in session:
        return redirect(url_for("signin"))
    return render_template("home.html", username=session["username"])



@app.route("/train")
def train():
    return render_template("train.html")

@app.route("/train_process", methods=["POST"])
def train_process():

    file = request.files["csv_file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path)

    
    df.fillna(df.mean(), inplace=True)

    
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    input_features = list(X.columns)
    output_label = "Potability"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

   
    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Neural Network": MLPClassifier(max_iter=500)
    }

    model_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        model_scores[name] = round(acc, 4)

       
        if name == "Random Forest":
            pickle.dump(model, open("model.pkl", "wb"))
            pickle.dump(scaler, open("scaler.pkl", "wb"))

  

    sns.countplot(x=y)
    plt.title("Potability Distribution")
    plt.savefig(f"{PLOT_FOLDER}/count.png")
    plt.close()

    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation")
    plt.savefig(f"{PLOT_FOLDER}/corr.png")
    plt.close()

    plt.bar(model_scores.keys(), model_scores.values())
    plt.title("Model Accuracy")
    plt.savefig(f"{PLOT_FOLDER}/accuracy.png")
    plt.close()

    return render_template(
        "train_result.html",
        input_features=input_features,
        output_label=output_label,
        model_scores=model_scores
    )



@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/detect_process", methods=["POST"])
def detect_process():

    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    values = np.array([[
        float(request.form["ph"]),
        float(request.form["Hardness"]),
        float(request.form["Solids"]),
        float(request.form["Chloramines"]),
        float(request.form["Sulfate"]),
        float(request.form["Conductivity"]),
        float(request.form["Organic_carbon"]),
        float(request.form["Trihalomethanes"]),
        float(request.form["Turbidity"])
    ]])

    scaled = scaler.transform(values)
    prediction = model.predict(scaled)[0]

    
    if prediction == 1:
        result = "Drinkable Water ✅"
        is_safe = True
    else:
        result = "Not Drinkable ❌"
        is_safe = False

    return render_template(
        "detect_result.html",
        prediction=result,
        is_safe=is_safe
    )



if __name__ == "__main__":
    app.run(debug=True)
