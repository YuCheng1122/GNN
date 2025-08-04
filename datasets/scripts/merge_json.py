#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

def merge_directories(src_dirs, dst_dir, pattern="*.json"):
    """
    將多個來源資料夾中的所有符合 pattern 的檔案合併到目標資料夾。
    - src_dirs: list of Path，來源資料夾清單
    - dst_dir: Path，目標資料夾
    - pattern: 檔名篩選（預設 *.json）
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dirs:
        if not src.exists():
            print(f"來源資料夾不存在，跳過: {src}")
            continue
        for file_path in src.rglob(pattern):
            # 計算相對於 src 的子路徑
            rel_path = file_path.relative_to(src)
            target_path = dst_dir / rel_path

            # 建立子目錄
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果檔案已存在，就跳過
            if target_path.exists():
                print(f"[跳過] 檔案已存在: {target_path}")
            else:
                shutil.copy2(file_path, target_path)
                print(f"[複製] {file_path} -> {target_path}")

if __name__ == "__main__":
    # 請自行確認以下路徑是否正確，並可依需新增或刪除資料夾
    src_dirs = [
        Path("/home/tommy/datasets/cross-architecture/results"),
        Path("/home/tommy/datasets/cross-architecture/results_0407"),
        Path("/home/tommy/datasets/cross-architecture/results_0428"),
        Path("/home/tommy/datasets/cross-architecture/results_0506"),
    ]
    dst_dir = Path("/home/tommy/datasets/cross-architecture/results_merged")

    merge_directories(src_dirs, dst_dir)
    print("合併完成！所有 .json 檔案已複製到：", dst_dir)
