import argparse
import os
from multiprocessing import Pool

from objathor.utils.download_utils import download_with_progress_bar

ALL_CKPT_IDS = ["text_nav", "box_nav", "text_box_nav"]

CKPT_KEY_VALUES = {
    "text_nav": "text_nav_DINOv2s__stage_02__steps_000600463848.pt",
    "box_nav": "box_nav_DINOv2s__stage_02__steps_000301747410.pt",
    "text_box_nav": "text_box_nav_DINOv2s__stage_02__steps_000309476886.pt",
}


def download_ckpt(info):
    url = info["url"]
    save_dir = info["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(save_dir, "model.ckpt")
    download_with_progress_bar(
        url=url,
        save_path=ckpt_path,
        desc=f"Downloading: {url}",
    )


def main():
    parser = argparse.ArgumentParser(description="Trained ckpt downloader.")
    parser.add_argument("--save_dir", required=True, help="Directory to save the downloaded files.")
    parser.add_argument(
        "--ckpt_ids",
        default=None,
        nargs="+",
        help=f"List of ckpt names to download, by default this will include all ids. Should be a subset of: {ALL_CKPT_IDS}",
    )
    parser.add_argument("--num", "-n", default=1, type=int, help="Number of parallel downloads.")
    args = parser.parse_args()

    if args.ckpt_ids is None:
        args.ckpt_ids = ALL_CKPT_IDS

    os.makedirs(args.save_dir, exist_ok=True)

    download_args = []
    for ckpt_id in args.ckpt_ids:
        save_dir = os.path.join(args.save_dir, ckpt_id)
        download_args.append(
            dict(
                url=f"https://pub-4194bc6e8ed3420491581242f2531a56.r2.dev/{CKPT_KEY_VALUES[ckpt_id]}",
                save_dir=save_dir,
            )
        )

    with Pool(args.num) as pool:
        pool.map(download_ckpt, download_args)


if __name__ == "__main__":
    main()
