import cv2
import tkinter as tk
from tkinter import Button, Label, filedialog
from PIL import Image, ImageTk

# Načtení Haar kaskád pro detekci obličeje, očí a úsměvu
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Načtení obrázku brýlí
glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
if glasses is None:
    raise FileNotFoundError("Obrázek brýlí 'glasses.png' nebyl nalezen!")

# Inicializace videa (0 pro webkameru nebo zadejte cestu k videu)
capture = cv2.VideoCapture(0)

# Proměnná pro statický obrázek
static_image = None
using_static_image = False


def overlay_image(background, overlay, x, y, w, h):
    if w <= 0 or h <= 0:
        return
    overlay = cv2.resize(overlay, (w, h))
    h_bg, w_bg, _ = background.shape  # Získání rozměrů pozadí

    # Oříznutí brýlí, pokud přesahují rámec obrazu
    x1, x2 = max(0, x), min(w_bg, x + w)
    y1, y2 = max(0, y), min(h_bg, y + h)
    overlay = overlay[:y2 - y1, :x2 - x1]

    for i in range(y1, y2):
        for j in range(x1, x2):
            if overlay[i - y1, j - x1][3] != 0:  # Pokud není průhledný pixel
                background[i, j] = overlay[i - y1, j - x1][:3]


def select_static_image():
    global static_image, using_static_image
    file_path = filedialog.askopenfilename(
        title="Vyberte obrázek",
        filetypes=[("Obrázky", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        static_image = cv2.imread(file_path)
        using_static_image = True
        process_and_display_image()


def switch_to_webcam():
    global using_static_image
    using_static_image = False
    update_frame()


def process_and_display_image():
    if static_image is None or not using_static_image:
        return

    frame = static_image.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Změna barvy ohraničení obličeje na oranžovou (0, 165, 255) ve formátu BGR
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Vylepšené parametry pro detekci očí
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=10,
                                            minSize=(30, 30),
                                            maxSize=(80, 80))

        # Filtrování očí podle pozice (očí jsou obvykle v horní polovině obličeje)
        filtered_eyes = []
        face_height = h
        for (ex, ey, ew, eh) in eyes:
            # Pouze oči v horní polovině obličeje
            if ey < face_height * 0.55:
                filtered_eyes.append((ex, ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Označení očí

        if len(filtered_eyes) >= 2:
            eye_1, eye_2 = filtered_eyes[:2]
            # Seřazení očí zleva doprava
            if eye_1[0] > eye_2[0]:
                eye_1, eye_2 = eye_2, eye_1

            glasses_x = eye_1[0] - 20
            glasses_y = eye_1[1] - 10
            glasses_width = max(1, (eye_2[0] + eye_2[2]) - eye_1[0] + 40)
            glasses_height = max(1, int(glasses_width * 0.5))
            overlay_image(roi_color, glasses, glasses_x, glasses_y, glasses_width, glasses_height)

        # Detekce úsměvu
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)  # Označení úsměvu

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)
    label_video.imgtk = frame_tk
    label_video.configure(image=frame_tk)


def update_frame():
    if using_static_image:
        return

    ret, frame = capture.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Změna barvy ohraničení obličeje na oranžovou (0, 165, 255) ve formátu BGR
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Vylepšené parametry pro detekci očí
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=10,
                                            minSize=(30, 30),
                                            maxSize=(80, 80))

        # Filtrování očí podle pozice (očí jsou obvykle v horní polovině obličeje)
        filtered_eyes = []
        face_height = h
        for (ex, ey, ew, eh) in eyes:
            # Pouze oči v horní polovině obličeje
            if ey < face_height * 0.55:
                filtered_eyes.append((ex, ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Označení očí

        if len(filtered_eyes) >= 2:
            eye_1, eye_2 = filtered_eyes[:2]
            # Seřazení očí zleva doprava
            if eye_1[0] > eye_2[0]:
                eye_1, eye_2 = eye_2, eye_1

            glasses_x = eye_1[0] - 20
            glasses_y = eye_1[1] - 10
            glasses_width = max(1, (eye_2[0] + eye_2[2]) - eye_1[0] + 40)
            glasses_height = max(1, int(glasses_width * 0.5))
            overlay_image(roi_color, glasses, glasses_x, glasses_y, glasses_width, glasses_height)

        # Detekce úsměvu
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)  # Označení úsměvu

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_pil)
    label_video.imgtk = frame_tk
    label_video.configure(image=frame_tk)
    label_video.after(10, update_frame)


def take_screenshot():
    if using_static_image and static_image is not None:
        frame = static_image.copy()
    else:
        ret, frame = capture.read()
        if not ret:
            return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Vylepšené parametry pro detekci očí
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=10,
                                            minSize=(30, 30),
                                            maxSize=(80, 80))

        # Filtrování očí podle pozice
        filtered_eyes = []
        face_height = h
        for (ex, ey, ew, eh) in eyes:
            if ey < face_height * 0.55:
                filtered_eyes.append((ex, ey, ew, eh))

        if len(filtered_eyes) >= 2:
            eye_1, eye_2 = filtered_eyes[:2]
            # Seřazení očí zleva doprava
            if eye_1[0] > eye_2[0]:
                eye_1, eye_2 = eye_2, eye_1

            glasses_x = eye_1[0] - 20
            glasses_y = eye_1[1] - 10
            glasses_width = max(1, (eye_2[0] + eye_2[2]) - eye_1[0] + 40)
            glasses_height = max(1, int(glasses_width * 0.5))
            overlay_image(roi_color, glasses, glasses_x, glasses_y, glasses_width, glasses_height)

    cv2.imwrite("Out/screenshot_with_glasses.png", frame)
    print("Snímek uložen jako screenshot_with_glasses.png")


def take_detection_screenshot():
    if using_static_image and static_image is not None:
        frame = static_image.copy()
    else:
        ret, frame = capture.read()
        if not ret:
            return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Změna barvy ohraničení obličeje na oranžovou (0, 165, 255) ve formátu BGR
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Vylepšené parametry pro detekci očí
        eyes = eye_cascade.detectMultiScale(roi_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=10,
                                            minSize=(30, 30),
                                            maxSize=(80, 80))

        # Filtrování očí podle pozice
        filtered_eyes = []
        face_height = h
        for (ex, ey, ew, eh) in eyes:
            if ey < face_height * 0.55:
                filtered_eyes.append((ex, ey, ew, eh))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    cv2.imwrite("Out/screenshot_detection.png", frame)
    print("Snímek uložen jako screenshot_detection.png")


# Vytvoření hlavního okna
root = tk.Tk()
root.title("Detekce obličeje a brýlí")
root.geometry("1200x700")

# Rám pro video
label_video = Label(root)
label_video.pack(side="left", padx=20, pady=20)

# Rám pro ovládací prvky
frame_controls = tk.Frame(root)
frame_controls.pack(side="right", padx=20, pady=20)

# Tlačítka pro výběr zdroje obrazu
btn_select_image = Button(frame_controls, text="Vybrat obrázek", command=select_static_image, height=2, width=20)
btn_select_image.pack(pady=10)

btn_webcam = Button(frame_controls, text="Použít webkameru", command=switch_to_webcam, height=2, width=20)
btn_webcam.pack(pady=10)

# Původní tlačítka
btn_screenshot = Button(frame_controls, text="Pořídit snímek s brýlemi", command=take_screenshot, height=2, width=20)
btn_screenshot.pack(pady=10)

btn_detection_screenshot = Button(frame_controls, text="Pořídit snímek detekce", command=take_detection_screenshot,
                                  height=2, width=20)
btn_detection_screenshot.pack(pady=10)

btn_exit = Button(frame_controls, text="Ukončit", command=root.quit, height=2, width=20)
btn_exit.pack(pady=10)

update_frame()
root.mainloop()

capture.release()
cv2.destroyAllWindows()