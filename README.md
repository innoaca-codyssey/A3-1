# 문서 스캐너

문서 사진의 검출/투시 변환을 구현하고 실제 이미지 조건별 성공과 실패를 분석합니다. NumPy 블러, 종횡비 보정과 PDF 출력을 포함합니다.

## 실행 환경

미션 전용 격리 환경에서 수행합니다. 실제 버전과 데이터 출처는 준비 후 기록합니다.

## 진행 상태

데이터 확보와 환경 준비 중입니다. 실제 실행 결과를 항목별로 추가합니다.

## 문서 사진

[DocUNet 원 소유자](https://www3.cs.stonybrook.edu/~cvl/docunet.html)가 공개한 표준 벤치마크의 실제 문서 사진을 확보했습니다. 직접 촬영 대안으로 사용하며 LMS의 cv19_test_images.zip과 같은 파일이라고 주장하지 않습니다.

```text
DocUNet original photos: 130
archive bytes: 344449749
SHA-256: ec226874d6845934c88b047c37e4e1a99cb2a8b9c42836739bc09015bcdc4405
ZIP CRC: PASS
```

원본은 별도로 보존하고 평가할 20장의 조건/파일명을 선정한 뒤 검출 결과를 확인합니다. HTTPS 배포 서버의 인증서 체인 오류가 있어 원 소유자 페이지에 게시된 원래 HTTP 주소에서 내려받았습니다.

## 스캐너 기본 구현

직접 구현한 좌표 순서, NumPy 블러의 두 방식, 투시 변환, 검출 실패와 파일 저장 경계를 확인했습니다. 마름모에서 같은 꼭짓점을 중복 선택하지 않도록 중심각 정렬과 일관된 시작점 규칙을 사용했습니다.

```text
$ python -m unittest discover -s tests -v
{"input_path": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/doc.png", "detected": true, "corners": [[118.0, 69.0], [491.0, 108.0], [452.0, 361.0], [89.0, 322.0]], "files": {"original": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/out/doc_original.png", "preprocessed": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/out/doc_preprocessed.png", "edges": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/out/doc_edges.png", "contours": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/out/doc_contours.png", "warped": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/out/doc_warped.png", "binary": "/var/folders/d1/s_rf51vn1gd_08wtq55ss94r0000gn/T/tmpgb378phd/out/doc_binary.png"}, "status": "ok"}
test_export_pdf_magic (test_scanner.ScannerTests.test_export_pdf_magic) ... ok
test_find_document_blank_failure (test_scanner.ScannerTests.test_find_document_blank_failure) ... ok
test_find_document_success (test_scanner.ScannerTests.test_find_document_success) ... ok
test_main_bad_args (test_scanner.ScannerTests.test_main_bad_args) ... ok
test_main_headless_success (test_scanner.ScannerTests.test_main_headless_success) ... ok
test_numpy_blur_loop_vectorized_equal_gray (test_scanner.ScannerTests.test_numpy_blur_loop_vectorized_equal_gray) ... ok
test_numpy_blur_loop_vectorized_equal_rgb (test_scanner.ScannerTests.test_numpy_blur_loop_vectorized_equal_rgb) ... ok
test_numpy_blur_matches_opencv_same_kernel_and_boundary (test_scanner.ScannerTests.test_numpy_blur_matches_opencv_same_kernel_and_boundary) ... ok
test_order_points_axis_aligned (test_scanner.ScannerTests.test_order_points_axis_aligned) ... ok
test_order_points_degenerate_rejected (test_scanner.ScannerTests.test_order_points_degenerate_rejected) ... ok
test_order_points_diamond (test_scanner.ScannerTests.test_order_points_diamond) ... ok
test_order_points_duplicate_rejected (test_scanner.ScannerTests.test_order_points_duplicate_rejected) ... ok
test_order_points_rotated_convex (test_scanner.ScannerTests.test_order_points_rotated_convex) ... ok
test_scan_image_not_found (test_scanner.ScannerTests.test_scan_image_not_found) ... ok
test_scan_image_saves_outputs (test_scanner.ScannerTests.test_scan_image_saves_outputs) ... ok
test_speed_test_has_measured_values (test_scanner.ScannerTests.test_speed_test_has_measured_values) ... ok
test_warp_document_outputs (test_scanner.ScannerTests.test_warp_document_outputs) ... ok

----------------------------------------------------------------------
Ran 17 tests in 1.755s

OK
```

## 평가 입력과 경계 주석

DocUNet의 서로 다른 실제 사진20장을 선택했습니다. 단순 배경5장, 그림자5장, 기울어짐5장, 복잡한 배경5장으로 구성했습니다. 그림자 그룹에는 알려진 강도의 부드러운 명암 변화를, 회전 그룹에는 평면 내55도 회전을 통제 변형으로 적용했습니다. 실제 촬영 카메라의3차원 각도를 알고 있다고 해석하지 않습니다. 원본 사진은 변경하지 않고 실습 사본에 변형했습니다.

검출기를 실행하기 전에 원본/변형 이미지를 보고 문서의 네 꼭짓점을 정규화 좌표로 주석하고, 경계 그림20장을 확인했습니다. 곡면 종이의 바깥 경계는 근사 사각형이며 이 주석의 한계를 보고서에서 구분합니다. 기하학적 검출은 주석과의IoU0.75 이상/정규화 꼭짓점RMSE0.04 이하로 측정하고, 실제 스캔본의 판독/왜곡 여부는 따로 검토합니다.
