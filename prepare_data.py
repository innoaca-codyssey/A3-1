"""DocUNet 원본을 검증하고 평가 사본20장을 재현한다."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import cv2
import numpy as np

URL = "http://vision.cs.stonybrook.edu/~kema/docwarp/original.zip"
ARCHIVE_SHA256 = "ec226874d6845934c88b047c37e4e1a99cb2a8b9c42836739bc09015bcdc4405"


def transform(image, group):
    """원본 보존 후 크기/명암/평면 내 회전만 통제한다."""
    h, w = image.shape[:2]
    scale = min(900 / w, 900 / h, 1)
    image = cv2.resize(image, (round(w * scale), round(h * scale)))
    if group == "shadow":
        h, w = image.shape[:2]
        x = np.arange(w) / w
        factor = 0.25 + 0.75 / (1 + np.exp(-(x - 0.55) / 0.05))
        image = np.clip(image.astype(float) * factor[None, :, None], 0, 255).astype(
            np.uint8
        )
    if group == "rotation":
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 55, 1)
        c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = int(w * c + h * s), int(h * c + w * s)
        matrix[:, 2] += np.array([nw / 2 - w / 2, nh / 2 - h / 2])
        border = np.median(
            np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]), axis=0
        )
        image = cv2.warpAffine(
            image, matrix, (nw, nh), borderValue=tuple(int(v) for v in border)
        )
        h, w = image.shape[:2]
        scale = min(900 / w, 900 / h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
    return image


def prepare(archive, manifest, output, originals=None):
    """파일명과 원본 해시를 고정해 같은 입력 사본을 만든다."""
    with open(archive, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("원본 ZIP SHA-256 불일치")
    records = json.loads(Path(manifest).read_text())["cases"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        lookup = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        for record in records:
            data = z.read(lookup[record["source"]])
            if hashlib.sha256(data).hexdigest() != record["source_sha256"]:
                raise ValueError("사진 SHA-256 불일치: " + record["source"])
            if originals is not None:
                Path(originals).mkdir(parents=True, exist_ok=True)
                (Path(originals) / record["source"]).write_bytes(data)
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("사진 디코딩 실패")
            image = transform(image, record["group"])
            if not cv2.imwrite(str(output / record["file"]), image):
                raise OSError("평가 사본 저장 실패")
    print(
        json.dumps(
            {"archive_sha256": digest, "cases": len(records), "output": str(output)}
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/original.zip"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("data/cases"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--originals", type=Path, default=Path("data/original"))
    args = parser.parse_args()
    if not args.archive.exists() and args.download:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        part = args.archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as response, part.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        part.replace(args.archive)
    prepare(args.archive, args.manifest, args.output, args.originals)


if __name__ == "__main__":
    main()
