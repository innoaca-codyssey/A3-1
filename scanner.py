"""OpenCV 문서 검출과 투시 변환 스캐너."""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


MAX_OUTPUT_EDGE = 4000


def _cross(a, b):
    """2차원 벡터의 외적에 해당하는 스칼라를 계산한다."""
    return a[0] * b[1] - a[1] * b[0]


def order_points(points):
    """볼록 사각형을 TL,TR,BR,BL로 정렬한다.

    TL은 x+y가 최소인 점이고 동률이면 x가 작은 점이다.
    이후 중심각 순서로 연결하므로 마름모에서도 같은 점을 중복 선택하지 않는다.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("유한한 4x2 좌표가 필요합니다")
    if len(np.unique(pts, axis=0)) != 4:
        raise ValueError("중복된 꼭짓점입니다")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    polygon = pts[np.argsort(angles)]
    turns = np.array(
        [
            _cross(
                polygon[(i + 1) % 4] - polygon[i],
                polygon[(i + 2) % 4] - polygon[(i + 1) % 4],
            )
            for i in range(4)
        ]
    )
    if not (np.all(turns > 1e-8) or np.all(turns < -1e-8)):
        raise ValueError("볼록하지 않거나 퇴화한 사각형입니다")
    signed_area = np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    if abs(signed_area) < 1e-8:
        raise ValueError("면적이 없는 사각형입니다")
    if signed_area < 0:
        polygon = polygon[::-1]
    start = np.lexsort((polygon[:, 1], polygon[:, 0], polygon.sum(axis=1)))[0]
    return np.roll(polygon, -int(start), axis=0).astype(np.float32)


def _gaussian_kernel(kernel_size, sigma):
    """정규화한 가우시안 커널을 계산한다."""
    if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be positive odd int")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    r = kernel_size // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _conv2d_reflect_loop(arr, kernel):
    """반사 경계에서 이중 반복문 컨볼루션을 계산한다."""
    k = kernel.shape[0]
    r = k // 2
    if arr.ndim == 2:
        pad = np.pad(arr, ((r, r), (r, r)), mode="reflect")
        out = np.empty(arr.shape, dtype=np.float64)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                out[y, x] = np.sum(pad[y : y + k, x : x + k] * kernel)
        return out
    pad = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.empty(arr.shape, dtype=np.float64)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            out[y, x] = np.sum(
                pad[y : y + k, x : x + k, :] * kernel[:, :, None], axis=(0, 1)
            )
    return out


def _conv2d_reflect_vectorized(arr, kernel):
    """반사 경계에서 배열 컨볼루션을 계산한다."""
    k = kernel.shape[0]
    r = k // 2
    if arr.ndim == 2:
        pad = np.pad(arr, ((r, r), (r, r)), mode="reflect")
        win = sliding_window_view(pad, (k, k))
        return np.einsum("ijkl,kl->ij", win, kernel, optimize=True)
    pad = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="reflect")
    win = sliding_window_view(pad, (k, k), axis=(0, 1))
    return np.einsum("ijckl,kl->ijc", win, kernel, optimize=True)


def numpy_gaussian_blur(image, kernel_size=5, sigma=1.0, vectorized=True):
    """NumPy만으로 가우시안 블러를 수행한다."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim not in (2, 3) or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("image must be 2D or 3D")
    kernel = _gaussian_kernel(kernel_size, sigma)
    return (_conv2d_reflect_vectorized if vectorized else _conv2d_reflect_loop)(
        arr, kernel
    )


def gaussian_blur_speed_test(image, kernel_size=5, sigma=1.0, repeats=3):
    """루프와 벡터화 버전의 실제 시간을 비교한다."""
    arr = np.asarray(image, dtype=np.float64)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    out = {}
    for name, flag in (("loop", False), ("vectorized", True)):
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = numpy_gaussian_blur(
                arr, kernel_size=kernel_size, sigma=sigma, vectorized=flag
            )
            dt = time.perf_counter() - t0
            best = dt if best is None or dt < best else best
        out[name] = {"seconds": float(best), "shape": list(res.shape)}
    return out


def find_document(
    image,
    blur_kernel=5,
    canny_low=50,
    canny_high=150,
    min_area_ratio=0.02,
    morph_close=True,
    selection="scored",
):
    """닫힌 윤곽의 면적/주변 대비/중심성을 비교해 문서 후보를 찾는다.

    largest는 초기 외곽/최대면적 방식을 재현하는 비교용 설정이다.
    """
    if image is None or image.size == 0:
        raise ValueError("빈 이미지입니다")
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("uint8 Grayscale/BGR 이미지가 필요합니다")
    if (
        not isinstance(blur_kernel, (int, np.integer))
        or blur_kernel <= 0
        or blur_kernel % 2 == 0
    ):
        raise ValueError("블러 커널은 양의 홀수 정수여야 합니다")
    if canny_low < 0 or canny_high < canny_low or not 0 < min_area_ratio <= 1:
        raise ValueError("Canny/면적 임계값을 확인하세요")
    if selection not in ("largest", "scored"):
        raise ValueError("지원하지 않는 후보 선택 방식입니다")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    if morph_close:
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mode = cv2.RETR_EXTERNAL if selection == "largest" else cv2.RETR_LIST
    contours, _ = cv2.findContours(edges, mode, cv2.CHAIN_APPROX_SIMPLE)
    total = image.shape[0] * image.shape[1]
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(view, contours, -1, (0, 255, 255), 1)
    best = None
    best_score = -1.0
    seen = set()
    epsilons = (0.02,) if selection == "largest" else (0.015, 0.02, 0.03, 0.04, 0.05)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < total * min_area_ratio:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon in epsilons:
            approximation = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue
            try:
                points = order_points(approximation.reshape(4, 2))
            except ValueError:
                continue
            key = tuple(points.ravel())
            if key in seen:
                continue
            seen.add(key)
            if selection == "largest":
                score = area
            else:
                quad_area = abs(float(cv2.contourArea(points)))
                if quad_area / max(area, 1.0) < 0.8:
                    continue
                mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mask, points.astype(np.int32), 255)
                band = cv2.subtract(cv2.dilate(mask, np.ones((15, 15), np.uint8)), mask)
                if not np.any(band):
                    continue
                inside = cv2.mean(gray, mask=mask)[0]
                outside = cv2.mean(gray, mask=band)[0]
                contrast = abs(inside - outside) / 255.0
                center = points.mean(axis=0) / np.array(
                    [image.shape[1], image.shape[0]]
                )
                centrality = 1 / (1 + 2 * np.linalg.norm(center - 0.5))
                score = (
                    (contrast * np.sqrt(area / total) + 0.03 * area / total)
                    * (0.5 + inside / 255)
                    * centrality
                )
            if score > best_score:
                best, best_score = points, score
    if best is not None:
        cv2.polylines(view, [best.astype(np.int32)], True, (0, 255, 0), 2)
    return {"gray": gray, "preprocessed": blur, "edges": edges, "contours": view}, best


def warp_document(image, corners):
    """원근 보정과 이진 스캔 결과를 생성한다."""
    pts = order_points(corners).astype(np.float32)
    tl, tr, br, bl = pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(round(max(width_a, width_b)))
    max_h = int(round(max(height_a, height_b)))
    if max_w < 2 or max_h < 2:
        raise ValueError("document too small")
    scale = min(1.0, MAX_OUTPUT_EDGE / float(max(max_w, max_h)))
    out_w = max(2, int(round(max_w * scale)))
    out_h = max(2, int(round(max_h * scale)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, m, (out_w, out_h))
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    binary = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
    )
    return warped, binary


def _ensure_dir(path):
    """출력 디렉터리를 준비한다."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_stage_images(output_dir, stem, original, stages, warped=None, binary=None):
    """실제 처리 단계 이미지를 저장한다."""
    _ensure_dir(output_dir)
    paths = {}
    items = {
        "original": original,
        "preprocessed": stages.get("preprocessed"),
        "edges": stages.get("edges"),
        "contours": stages.get("contours"),
    }
    if warped is not None:
        items["warped"] = warped
    if binary is not None:
        items["binary"] = binary
    for name, img in items.items():
        if img is None:
            continue
        p = os.path.join(output_dir, f"{stem}_{name}.png")
        if not cv2.imwrite(p, img):
            raise OSError(f"파일 저장 실패: {p}")
        paths[name] = p
    return paths


def scan_image(path, output_dir, parameters=None):
    """이미지 파일을 스캔하고 결과 파일을 저장한다."""
    params = dict(parameters or {})
    _ensure_dir(output_dir)
    stem = Path(path).stem or "image"
    result = {"input_path": str(path), "detected": False, "corners": None, "files": {}}
    if not os.path.exists(path):
        result["status"] = "not_found"
        return result
    image = cv2.imread(str(path))
    if image is None:
        result["status"] = "read_failed"
        return result
    stages, corners = find_document(
        image,
        blur_kernel=int(params.get("blur_kernel", 5)),
        canny_low=int(params.get("canny_low", 50)),
        canny_high=int(params.get("canny_high", 150)),
        min_area_ratio=float(params.get("min_area_ratio", 0.02)),
        selection=params.get("selection", "scored"),
        morph_close=bool(params.get("morph_close", True)),
    )
    warped = binary = None
    if corners is not None:
        warped, binary = warp_document(image, corners)
        result["detected"] = True
        result["corners"] = [[float(x), float(y)] for x, y in corners]
        result["status"] = "ok"
    else:
        result["status"] = "not_detected"
    result["files"] = _write_stage_images(
        output_dir, stem, image, stages, warped, binary
    )
    return result


def export_pdf(image_paths, pdf_path):
    """보너스 PDF 내보내기를 수행한다."""
    if not image_paths:
        raise ValueError("image_paths is empty")
    from PIL import Image

    pages = []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        pages.append(im)
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
    for im in pages:
        im.close()
    return pdf_path


def run_gui(
    image_path=None, camera=False, output_dir="output", parameters=None, pdf_path=None
):
    """트랙바와 단계별 창을 사용해 파일 또는 웹캠을 처리한다."""
    params = dict(parameters or {})
    capture = None
    saved = 0
    try:
        if camera:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                return 2
            frame = None
        else:
            frame = cv2.imread(str(image_path)) if image_path else None
            if frame is None:
                return 2
        cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "blur",
            "controls",
            int(params.get("blur_kernel", 5)),
            31,
            lambda value: None,
        )
        cv2.createTrackbar(
            "canny_low",
            "controls",
            int(params.get("canny_low", 50)),
            255,
            lambda value: None,
        )
        cv2.createTrackbar(
            "canny_high",
            "controls",
            int(params.get("canny_high", 150)),
            255,
            lambda value: None,
        )
        while True:
            if capture is not None:
                ok, frame = capture.read()
                if not ok:
                    return 2
            blur = max(1, cv2.getTrackbarPos("blur", "controls"))
            blur += int(blur % 2 == 0)
            low = cv2.getTrackbarPos("canny_low", "controls")
            high = max(low, cv2.getTrackbarPos("canny_high", "controls"))
            stages, corners = find_document(
                frame,
                blur_kernel=blur,
                canny_low=low,
                canny_high=high,
                min_area_ratio=float(params.get("min_area_ratio", 0.02)),
                selection=params.get("selection", "scored"),
            )
            warped = binary = None
            if corners is not None:
                warped, binary = warp_document(frame, corners)
            cv2.imshow("original", frame)
            cv2.imshow("preprocessing", stages["preprocessed"])
            cv2.imshow("edges", stages["edges"])
            cv2.imshow("contours", stages["contours"])
            cv2.imshow(
                "result",
                binary
                if binary is not None
                else np.zeros(frame.shape[:2], dtype=np.uint8),
            )
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord("s"):
                stem = "camera" if camera else Path(image_path).stem
                paths = _write_stage_images(
                    output_dir, f"{stem}-{saved:03d}", frame, stages, warped, binary
                )
                if pdf_path and binary is not None:
                    export_pdf([paths["binary"]], pdf_path)
                saved += 1
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def build_arg_parser():
    """CLI 인자 파서를 생성한다."""
    p = argparse.ArgumentParser(description="OpenCV document scanner")
    p.add_argument("--image")
    p.add_argument("--camera", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", default="output")
    p.add_argument("--blur", type=int, default=5)
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=150)
    p.add_argument("--min-area-ratio", type=float, default=0.02)
    p.add_argument("--selection", choices=["scored", "largest"], default="scored")
    p.add_argument("--pdf")
    return p


def main(argv=None):
    """CLI 진입점이다."""
    args = build_arg_parser().parse_args(argv)
    if bool(args.image) == bool(args.camera):
        return 2
    params = {
        "blur_kernel": args.blur,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "min_area_ratio": args.min_area_ratio,
        "selection": args.selection,
    }
    if args.headless:
        if not args.image:
            return 2
        result = scan_image(args.image, args.output, params)
        if args.pdf and result.get("detected") and result["files"].get("binary"):
            export_pdf([result["files"]["binary"]], args.pdf)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("detected") else 1
    return run_gui(
        image_path=args.image,
        camera=args.camera,
        output_dir=args.output,
        parameters=params,
        pdf_path=args.pdf,
    )


if __name__ == "__main__":
    raise SystemExit(main())
