#!/usr/bin/env python3
"""
Crawl dữ liệu STDN từ Google Drive về local theo (MM, YYYY).

Giống crawl.py nhưng phần ACTUAL lấy NHIỀU file (vd MM_volume.csv + MM_flow.parquet).
Cấu hình mặc định: configSTDN.yaml.

Dùng:
    python crawlSTDN.py --month 4 --year 2024
    python crawlSTDN.py            # sẽ hỏi MM, YYYY ở terminal
"""

import argparse
import io
import os
import sys

import yaml

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"


def log(msg=""):
    """In log và flush ngay để theo dõi real-time."""
    print(msg, flush=True)


def human_size(nbytes):
    """Đổi byte -> chuỗi dễ đọc (KB/MB/GB)."""
    try:
        n = float(nbytes)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


# --------------------------------------------------------------------------- #
# Config & xác thực
# --------------------------------------------------------------------------- #
def load_config(path="configSTDN.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_drive_service(cfg):
    gd = cfg["google_drive"]
    creds_file = gd["credentials_file"]
    if not os.path.exists(creds_file):
        sys.exit(f"[LỖI] Không thấy file credentials '{creds_file}'.")

    log("[1/4] Đang xác thực với Google Drive...")
    if gd.get("use_service_account"):
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            creds_file, scopes=SCOPES
        )
        log("      ✓ Xác thực bằng service account.")
    else:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        token_file = gd.get("token_file", "token.json")
        creds = None
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                log("      ↻ Token hết hạn, đang làm mới...")
                creds.refresh(Request())
            else:
                log("      ⚠ Cần đăng nhập: trình duyệt sẽ mở ra, hãy chọn tài khoản")
                log("        đã được chia sẻ folder rồi bấm Cho phép (Allow).")
                log("        Đang chờ bạn hoàn tất trên trình duyệt...")
                flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_file, "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        log("      ✓ Đã xác thực (token lưu tại token.json).")

    return build("drive", "v3", credentials=creds, cache_discovery=False)


# --------------------------------------------------------------------------- #
# Điều hướng folder / file
# --------------------------------------------------------------------------- #
def _list(service, query, fields="files(id, name, size)"):
    resp = service.files().list(
        q=query, fields=fields, pageSize=50,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return resp.get("files", [])


def find_folder(service, name, parent_id=None):
    """Tìm folder theo tên (tùy chọn trong parent_id). Trả về id hoặc None."""
    q = f"name = '{name}' and mimeType = '{FOLDER_MIME}' and trashed = false"
    if parent_id:
        q = f"'{parent_id}' in parents and " + q
    folders = _list(service, q, fields="files(id, name)")
    return folders[0]["id"] if folders else None


def find_file_in_folder(service, folder_id, filename):
    """Tìm file theo tên trong folder_id. Trả về dict hoặc None."""
    q = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    files = _list(service, q)
    return files[0] if files else None


def resolve_root(service, cfg):
    gd = cfg["google_drive"]
    if gd.get("root_folder_id"):
        return gd["root_folder_id"]

    # Đi từ folder top, ví dụ "TLC_project"
    top = gd["root_folder_name"]
    log(f"      • tìm folder gốc '{top}'...")
    fid = find_folder(service, top)
    if not fid:
        sys.exit(
            f"[LỖI] Không tìm thấy folder '{top}'. "
            "Kiểm tra tài khoản đăng nhập đã được chia sẻ folder này chưa."
        )

    # Đi tiếp qua root_subpath, ví dụ ["data"]
    subpath = gd.get("root_subpath", []) or []
    cur = fid
    walked = [top]
    for part in subpath:
        nxt = find_folder(service, part, cur)
        walked.append(part)
        if not nxt:
            sys.exit(f"[LỖI] Không tìm thấy folder gốc theo đường dẫn '{'/'.join(walked)}'.")
        cur = nxt
    log(f"      ✓ Root = {'/'.join(walked)}")
    return cur


def resolve_subpath(service, root_id, parts):
    """Đi lần lượt qua các folder con (parts) từ root. Trả về folder id cuối."""
    cur = root_id
    for part in parts:
        nxt = find_folder(service, part, cur)
        if not nxt:
            log(f"        ✗ thiếu folder '{part}'")
            return None
        cur = nxt
    return cur


# --------------------------------------------------------------------------- #
# Tải file (chunked, hỗ trợ file lớn)
# --------------------------------------------------------------------------- #
def download_file(service, file_meta, dest_path, chunk_size_mb=50, overwrite=False):
    import time

    if os.path.exists(dest_path) and not overwrite:
        log(f"  ↷ Bỏ qua (đã có): {dest_path}")
        return

    total = int(file_meta.get("size", 0) or 0)
    log(f"  ↓ Bắt đầu tải '{file_meta['name']}' ({human_size(total)}) -> {dest_path}")

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    request = service.files().get_media(fileId=file_meta["id"])
    buffer = io.FileIO(dest_path + ".part", "wb")
    downloader = MediaIoBaseDownload(
        buffer, request, chunksize=chunk_size_mb * 1024 * 1024
    )
    start = time.time()
    done = False
    try:
        while not done:
            status, done = downloader.next_chunk()
            if status:
                got = status.resumable_progress
                pct = int(status.progress() * 100)
                elapsed = max(time.time() - start, 1e-6)
                speed = got / elapsed  # byte/giây
                # in đè trên cùng dòng để thấy tiến độ chạy
                print(
                    f"    {pct:3d}%  {human_size(got)}/{human_size(total)}"
                    f"  ({human_size(speed)}/s)        ",
                    end="\r", flush=True,
                )
    finally:
        buffer.close()
    os.replace(dest_path + ".part", dest_path)
    elapsed = time.time() - start
    log(f"    ✓ Xong: {dest_path}  ({human_size(total)} trong {elapsed:.1f}s)")


# --------------------------------------------------------------------------- #
# Logic chính
# --------------------------------------------------------------------------- #
def crawl_month(service, cfg, year, month):
    year, month = f"{int(year):04d}", f"{int(month):02d}"
    local = cfg["local"]
    dl = cfg.get("download", {})
    chunk, overwrite = dl.get("chunk_size_mb", 50), dl.get("overwrite", False)

    log("[2/4] Xác định folder gốc trên Drive...")
    root_id = resolve_root(service, cfg)

    def fetch(target, dest_subdir, section):
        subpath = [p.format(year=year, month=month) for p in target.get("subpath", [])]
        filename = target["filename"].format(year=year, month=month)
        # save_as: tên lưu ở local (mặc định giữ nguyên tên trên Drive)
        save_as = target.get("save_as") or target["filename"]
        save_as = save_as.format(year=year, month=month)

        rel = "/".join(subpath) or "<root>"
        log(f"  • [{section}] đường dẫn '{rel}', tìm file '{filename}'...")
        folder_id = resolve_subpath(service, root_id, subpath)
        if not folder_id:
            log(f"  ✗ Không thấy đường dẫn '{rel}' ({section}).")
            return
        meta = find_file_in_folder(service, folder_id, filename)
        if not meta:
            log(f"  ✗ Không thấy file '{filename}' trong '{rel}' ({section}).")
            return
        dest = os.path.join(local["incoming_dir"], dest_subdir, save_as)
        download_file(service, meta, dest, chunk, overwrite)

    def as_list(value):
        """Cho phép actual/context khai báo 1 dict hoặc danh sách dict."""
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    # ----- actual (có thể nhiều file: volume + flow ...) -----
    log(f"\n[3/4] === ACTUAL ({year}-{month}) ===")
    for a in as_list(cfg["targets"]["actual"]):
        fetch(a, local["actual_dir"], "actual")

    # ----- context -----
    log(f"\n[4/4] === CONTEXT ({year}) ===")
    for c in as_list(cfg["targets"]["context"]):
        fetch(c, local["context_dir"], "context")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Crawl dữ liệu STDN từ Google Drive theo tháng.")
    p.add_argument("--month", "-m", type=int, help="Tháng (MM), ví dụ 4")
    p.add_argument("--year", "-y", type=int, help="Năm (YYYY), ví dụ 2024")
    p.add_argument("--config", "-c", default="configSTDN.yaml", help="Đường dẫn config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    month = args.month if args.month is not None else input("Nhập MM (tháng): ").strip()
    year = args.year if args.year is not None else input("Nhập YYYY (năm): ").strip()
    if not (1 <= int(month) <= 12):
        sys.exit("[LỖI] MM phải trong khoảng 1-12.")

    log(f"==> Bắt đầu crawl STDN tháng {int(month):02d}/{int(year):04d}")
    service = get_drive_service(cfg)
    try:
        crawl_month(service, cfg, year, month)
    except HttpError as e:
        sys.exit(f"[LỖI Drive API] {e}")
    log("\n==> HOÀN TẤT.")


if __name__ == "__main__":
    main()
