"""사전에 주석한 실제 사진의 검출/변환 성능을 측정한다."""

import argparse
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from scanner import find_document, warp_document


def geometric_score(predicted, expected, shape):
    """픽셀 IoU와 최소 순환 꼭짓점 RMSE를 계산한다."""
    if predicted is None:
        return {"iou": 0.0, "corner_rmse": None, "success": False}
    h, w = shape[:2]
    expected = np.asarray(expected, dtype=np.float32) * [w, h]
    masks = []
    for points in (predicted, expected):
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.asarray(points, dtype=np.int32), 1)
        masks.append(mask)
    union = np.logical_or(*masks).sum()
    iou = float(np.logical_and(*masks).sum() / union) if union else 0.0
    errors = []
    for pts in (expected, expected[::-1]):
        for shift in range(4):
            errors.append(
                np.mean(np.sum((predicted - np.roll(pts, shift, axis=0)) ** 2, axis=1))
            )
    rmse = float(np.sqrt(min(errors)) / np.hypot(w, h))
    return {
        "iou": iou,
        "corner_rmse": rmse,
        "success": bool(iou >= 0.75 and rmse <= 0.04),
    }


def evaluate_cases(case_root, manifest, parameters, output=None):
    """동일 설정으로20장의 실제 결과를 집계한다."""
    cases = json.loads(Path(manifest).read_text())["cases"]
    results = []
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    for record in cases:
        image = cv2.imread(str(Path(case_root) / record["file"]))
        if image is None:
            raise ValueError("입력 사진을 읽지 못했습니다: " + record["file"])
        stages, corners = find_document(image, **parameters)
        score = geometric_score(corners, record["quad_normalized"], image.shape)
        result = {
            "file": record["file"],
            "group": record["group"],
            "detected": corners is not None,
            "corners": corners.tolist() if corners is not None else None,
            **score,
        }
        if output:
            target = Path(output) / Path(record["file"]).stem
            target.mkdir(exist_ok=True)
            cv2.imwrite(str(target / "input.jpg"), image)
            cv2.imwrite(str(target / "contours.jpg"), stages["contours"])
            cv2.imwrite(str(target / "edges.png"), stages["edges"])
            if corners is not None:
                warped, binary = warp_document(image, corners)
                cv2.imwrite(str(target / "warped.jpg"), warped)
                cv2.imwrite(str(target / "binary.png"), binary)
        results.append(result)
    groups = {}
    for group in sorted({r["group"] for r in results}):
        selected = [r for r in results if r["group"] == group]
        passed = sum(r["success"] for r in selected)
        groups[group] = {
            "tests": len(selected),
            "success": passed,
            "failure": len(selected) - passed,
            "success_rate": passed / len(selected),
        }
    return {
        "parameters": parameters,
        "groups": groups,
        "success": sum(r["success"] for r in results),
        "tests": len(results),
        "results": results,
    }


def main():
    """기본 평가 또는 고정 파라미터 비교를 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/cases"))
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/cases.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.json"))
    parser.add_argument("--stages", type=Path)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--low", type=int, default=50)
    parser.add_argument("--high", type=int, default=150)
    parser.add_argument("--min-area", type=float, default=0.08)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        results = []
        for blur, thresholds, area in itertools.product(
            (3, 5, 9), ((20, 60), (50, 150), (100, 250)), (0.02, 0.05, 0.08)
        ):
            parameters = {
                "blur_kernel": blur,
                "canny_low": thresholds[0],
                "canny_high": thresholds[1],
                "min_area_ratio": area,
            }
            report = evaluate_cases(args.cases, args.manifest, parameters)
            results.append(report)
            print(
                json.dumps(
                    {
                        "parameters": parameters,
                        "success": report["success"],
                        "tests": report["tests"],
                    }
                ),
                flush=True,
            )
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    else:
        parameters = {
            "blur_kernel": args.blur,
            "canny_low": args.low,
            "canny_high": args.high,
            "min_area_ratio": args.min_area,
        }
        result = evaluate_cases(args.cases, args.manifest, parameters, args.stages)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "results"},
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
