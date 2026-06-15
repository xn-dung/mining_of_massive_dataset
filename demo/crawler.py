from pathlib import Path

from . import config
from .utils.file_utils import ensure_dir


def _import_crawl_module():
    try:
        from .CrawlSTDN import crawlSTDN
    except ImportError as exc:
        raise RuntimeError(
            "CrawlSTDN dependencies are missing. Install demo/CrawlSTDN/requirements.txt "
            "and requirements.txt, then run again."
        ) from exc
    return crawlSTDN


def _month_raw_dir(period):
    return ensure_dir(config.CRAWLED_DIR / period)


def _require_crawled_files(raw_dir):
    raw_dir = Path(raw_dir)
    checks = {
        "volume csv": any(path.is_file() and "volume" in path.name.lower() and path.suffix.lower() == ".csv" for path in raw_dir.iterdir()),
        "flow parquet": any(path.is_file() and "flow" in path.name.lower() and path.suffix.lower() in {".parquet", ".pq"} for path in raw_dir.iterdir()),
        "weather file": any(path.is_file() and "weather" in path.name.lower() and path.suffix.lower() in {".csv", ".parquet", ".pq"} for path in raw_dir.iterdir()),
        "holiday file": any(path.is_file() and "holiday" in path.name.lower() and path.suffix.lower() in {".csv", ".parquet", ".pq"} for path in raw_dir.iterdir()),
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        raise FileNotFoundError(
            f"CrawlSTDN did not produce required files in {raw_dir}. Missing: {', '.join(missing)}"
        )


def crawl_month_to_raw_dir(period):
    period = str(period)[:7]
    year, month = period.split("-")
    raw_dir = _month_raw_dir(period)
    crawlSTDN = _import_crawl_module()

    cfg = crawlSTDN.load_config(config.CRAWL_STDN_CONFIG_PATH)
    cfg["google_drive"]["credentials_file"] = str(config.CRAWL_STDN_DIR / cfg["google_drive"]["credentials_file"])
    cfg["google_drive"]["token_file"] = str(config.CRAWL_STDN_DIR / cfg["google_drive"].get("token_file", "token.json"))
    cfg["local"]["incoming_dir"] = str(raw_dir)
    cfg["local"]["actual_dir"] = ""
    cfg["local"]["context_dir"] = ""

    service = crawlSTDN.get_drive_service(cfg)
    crawlSTDN.crawl_month(service, cfg, year, month)
    _require_crawled_files(raw_dir)
    return raw_dir
