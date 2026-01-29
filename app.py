import textwrap
from flask import Flask, render_template, request
from ultralytics import YOLO
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import cv2
import os
import uuid
import json
import pandas as pd
from datetime import datetime
from flask import Response, jsonify, send_from_directory, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

HISTORY_FILE = "history.json"

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

VIDEO_FOLDER = "videos"
RESULT_VIDEO_FOLDER = "results_videos"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(RESULT_VIDEO_FOLDER, exist_ok=True)

stream_people_count = 0

model = YOLO("yolov8m.pt")


def save_request(record):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def draw_people_only(image, results):
    people_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            conf = float(box.conf[0])
            if conf < 0.45:
                continue

            if label != "person":
                continue

            people_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                "person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    return image, people_count


def generate_pdf(filename="report.pdf"):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    margin_x = 40
    y = height - 40

    c.setFont("DejaVu", 20)
    c.drawString(margin_x, y, "Отчёт по распознаванию людей")
    y -= 30

    c.setFont("DejaVu", 11)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin_x, y, f"Дата формирования: {now}")
    y -= 30

    c.setFont("DejaVu", 12)
    c.line(margin_x, y, width - margin_x, y)
    y -= 18

    headers = ["Дата", "Тип", "Файл", "Люди"]
    x_positions = [margin_x, 170, 260, 480]

    for header, x in zip(headers, x_positions):
        c.drawString(x, y, header)

    y -= 10
    c.line(margin_x, y, width - margin_x, y)
    y -= 20

    c.setFont("DejaVu", 11)
    max_widths = [120, 80, 200, 50]

    for rec in history:
        if y < 60:
            c.showPage()
            c.setFont("DejaVu", 11)
            y = height - 60

        row = [
            rec.get("timestamp", ""),
            rec.get("type", ""),
            rec.get("filename", ""),
            str(rec.get("people_count", ""))
        ]

        wraped_texts = []
        for i, text in enumerate(row):
            wrapped = textwrap.wrap(text, width=30 if i == 2 else 50)  # для файла шире перенос
            wraped_texts.append(wrapped)

        max_lines = max(len(w) for w in wraped_texts)

        for line_idx in range(max_lines):
            for col_idx, x in enumerate(x_positions):
                try:
                    line_text = wraped_texts[col_idx][line_idx]
                except IndexError:
                    line_text = ""
                c.drawString(x, y, line_text)
            y -= 15

        y -= 5

    c.save()


def generate_excel(filename="report.xlsx"):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    df = pd.DataFrame(history)
    df.to_excel(filename, index=False)


@app.route("/history")
def show_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    return render_template("history.html", history=history)


@app.route("/download_pdf")
def download_pdf():
    generate_pdf("report.pdf")
    return send_file("report.pdf", as_attachment=True)


@app.route("/download_excel")
def download_excel():
    generate_excel("report.xlsx")
    return send_file("report.xlsx", as_attachment=True)


@app.route("/results/<filename>")
def get_result_image(filename):
    return send_from_directory("results", filename)


@app.route("/results_videos/<filename>")
def get_result_video(filename):
    return send_from_directory("results_videos", filename)


@app.route("/stream_count")
def stream_count():
    return jsonify({
        "people": stream_people_count
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files["video"]

    filename = str(uuid.uuid4()) + ".mp4"
    video_path = os.path.join(VIDEO_FOLDER, filename)
    result_path = os.path.join(RESULT_VIDEO_FOLDER, filename)

    file.save(video_path)

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264
    out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

    last_people_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated, people_count = draw_people_only(frame, results)
        last_people_count = people_count
        out.write(annotated)

    cap.release()
    out.release()

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "video",
        "filename": filename,
        "people_count": last_people_count
    }
    save_request(record)

    return render_template(
        "index.html",
        video_path="/results_videos/" + filename,
        video_count=last_people_count
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    filename = str(uuid.uuid4()) + ".jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, filename)

    file.save(upload_path)

    image = cv2.imread(upload_path)
    results = model(image)

    annotated, people_count = draw_people_only(image, results)
    cv2.imwrite(result_path, annotated)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "image",
        "filename": filename,
        "people_count": people_count
    }

    save_request(record)

    return render_template(
        "index.html",
        count=people_count,
        result_image="/results/" + filename

    )


@app.route("/results/<path:filename>")
def results_file(filename):
    return app.send_static_file("../results/" + filename)


def generate_camera_stream():
    global stream_people_count

    STREAM_URL = "rtsp://127.0.0.1:8554/myvideo"
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Не удалось подключиться к RTSP видеопотоку")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        annotated, people_count = draw_people_only(frame, results)
        stream_people_count = people_count

        _, buffer = cv2.imencode(".jpg", annotated)
        frame_bytes = buffer.tobytes()

        yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/camera")
def camera():
    return Response(
        generate_camera_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
