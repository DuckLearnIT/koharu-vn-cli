# Koharu-VN CLI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DuckLearnIT/koharu-vn-cli/blob/main/koharu_colab.ipynb)

Công cụ dịch truyện tranh / doujinshi / sáng tác Artist CG tiếng Việt bằng **ONNX thuần + Gemini Vision**:
- **Detector**: `ogkalu/comic-text-and-bubble-detector` (RT-DETRv2) + `mayocream/comic-text-detector` (YOLOv5 `blk` + UNet `seg` + DBNet `det`)
- **Inpainting**: `mayocream/lama-manga-onnx` (`anime-manga-big-lama`) + solid-fill thông minh
- **AI Vision / Translation**: Gemini Dual-model (Chính làm Story Bible/Vision, Phụ dịch/sáng tác) + xoay vòng nhiều API key

## Cài đặt lại tự động trên máy mới (sau khi reset Windows)

Mở **Command Prompt (`cmd`)** và dán 1 lệnh duy nhất sau (tự cài `Git`, `GitHub CLI`, đăng nhập GitHub, clone repo về `%USERPROFILE%\koharu-vn-cli`, cài `uv` + Python 3.12 + lệnh `koharu-vn` toàn cục và tải sẵn 3 model ONNX):

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements; winget install -e --id GitHub.cli --accept-package-agreements --accept-source-agreements; $env:Path=[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User'); gh auth login; gh repo clone DuckLearnIT/koharu-vn-cli \"$env:USERPROFILE\koharu-vn-cli\"; & \"$env:USERPROFILE\koharu-vn-cli\setup.cmd\""
```

Nếu máy đã có `git` và `gh` (đã clone sẵn), chỉ cần chạy trong thư mục dự án:

```cmd
setup.cmd
```

## Sử dụng nhanh

Mở `cmd` mới và chạy:

```cmd
koharu-vn
```

Hoặc kiểm tra test:

```cmd
uv run pytest -v
```
