import json, os, cv2, numpy as np
import mediapipe as mp

# Root folder containing subfolders 01, 02, 03, ...
ROOT_DIR = "images"

mp_face_mesh = mp.solutions.face_mesh


def save_landmarks(folder_path, landmarks):
    """
    Save landmarks.txt in literal-evaluable dict format:
    {'landmarks': [[{'x': ..., 'y': ...}, ...]]}
    """
    out = {"landmarks": [landmarks]}
    output_path = os.path.join(folder_path, "landmarks.txt")

    # Write as Python literal, not JSON (so ast.literal_eval can read)
    with open(output_path, "w") as f:
        f.write(str(out))

    print(f"✅ Saved: {output_path}")


def process_image(folder_path, face_mesh):
    """Process original_image.png inside a folder."""
    img_path = os.path.join(folder_path, "original_image.png")
    if not os.path.exists(img_path):
        print(f"⚠️ Skipping {folder_path} (no original_image.png)")
        return

    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"⚠️ Failed to read image: {img_path}")
        return

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        print(f"❌ No face detected in {folder_path}")
        return

    # Get the first detected face (if multiple)
    face = res.multi_face_landmarks[0]

    # Convert to pixel coordinates
    landmarks = [{"x": lm.x * w, "y": lm.y * h} for lm in face.landmark]

    save_landmarks(folder_path, landmarks)


def main():
    opts = dict(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_face_mesh.FaceMesh(**opts) as face_mesh:
        for subdir in sorted(os.listdir(ROOT_DIR)):
            folder_path = os.path.join(ROOT_DIR, subdir)
            if not os.path.isdir(folder_path):
                continue

            print(f"📸 Processing folder: {folder_path}")
            process_image(folder_path, face_mesh)


if __name__ == "__main__":
    main()
