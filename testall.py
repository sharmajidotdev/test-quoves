import ast
import base64
import time
from pathlib import Path

import requests

API_VERSION = "v2"  # Change to "v2" to target API v2
BASE_URL = "http://localhost:8000"
if API_VERSION == "v1":
    SUBMIT_ENDPOINT = f"{BASE_URL}/api/v1/frontal/crop/submit"
    STATUS_ENDPOINT = f"{BASE_URL}/api/v1/frontal/crop/status"
elif API_VERSION == "v2":
    SUBMIT_ENDPOINT = f"{BASE_URL}/api/v2/jobs"
    STATUS_ENDPOINT = f"{BASE_URL}/api/v2/jobs"
else:
    raise ValueError("Unsupported API_VERSION. Use 'v1' or 'v2'.")
POLL_INTERVAL = 2
SAMPLE_ROOT = Path("samples/images")


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_payload(folder: Path) -> dict:
    payload = {
        "image": b64(folder / "original_image.png"),
        "segmentation_map": b64(folder / "segmentation_map.png"),
        "landmarks": ast.literal_eval((folder / "landmarks.txt").read_text())["landmarks"][0],
    }
    if API_VERSION == "v2":
        payload["regions"] = [
            "forehead",
            "nose",
            "right_undereye",
            "left_undereye",
            "low_face",
        ]
    return payload


def submit_jobs() -> dict:
    jobs = {}
    for folder in sorted(p for p in SAMPLE_ROOT.iterdir() if p.is_dir()):
        output_svg = folder / "output.svg"
        if output_svg.exists():
            try:
                output_svg.unlink()
                print(f"🗑️ Deleted old {output_svg}")
            except Exception as exc:
                print(f"⚠️ Could not delete {output_svg}: {exc}")

        payload = build_payload(folder)
        response = requests.post(SUBMIT_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        job_id = response.json()["id"]
        jobs[job_id] = {"folder": folder, "status": "pending"}
        print(f"Submitted {folder.name}: job {job_id}")
    return jobs


def poll_jobs(jobs: dict):
    pending = set(jobs.keys())
    while pending:
        completed = []
        for job_id in list(pending):
            status_url = (
                f"{STATUS_ENDPOINT}/{job_id}"
                if API_VERSION == "v1"
                else f"{STATUS_ENDPOINT}/{job_id}"
            )
            resp = requests.get(status_url, timeout=30)
            resp.raise_for_status()
            status_payload = resp.json()

            svg_payload = status_payload.get("svg")
            if svg_payload:
                print(f"{job_id}: completed")
                folder = jobs[job_id]["folder"]
                (folder / "output.svg").write_bytes(base64.b64decode(svg_payload))
                print(f"Saved SVG to {folder / 'output.svg'}")
                completed.append(job_id)
                continue

            state = status_payload.get("status", "unknown")
            print(f"{job_id}: {state}")
            if state in {"failed"}:
                completed.append(job_id)

        for job_id in completed:
            pending.discard(job_id)
        if pending:
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    if not SAMPLE_ROOT.exists():
        raise SystemExit(f"Sample directory not found: {SAMPLE_ROOT}")
    job_map = submit_jobs()
    if not job_map:
        print("No sample folders found.")
    else:
        poll_jobs(job_map)
