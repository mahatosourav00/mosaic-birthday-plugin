"""Engine side of the Birthday plugin: turn the form values into film.py's settings and render the film.

film.py (typography, grade, music, encode) and world.py (the 3D night-sky world, built on mosaicvid.animation) are the
original standalone birthday project, run here through runpy with a progress hook. All inputs come from the job:
no hard-coded paths.
"""
from __future__ import annotations

import contextlib
import json
import re
import runpy
import sys
from pathlib import Path


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name.strip()).strip("_") or "birthday"


def run(ctx) -> dict:
    v = ctx.values
    out = ctx.out_dir
    nested = ctx.nested()
    paras = [p.strip() for p in re.split(r"\n\s*\n", v["message"].replace("\r\n", "\n")) if p.strip()]
    if not paras:
        raise ValueError("The message is empty.")
    if not v["people.her_name"].strip():
        raise ValueError("Enter the birthday person's name.")
    music = v["files.music"].strip()
    if music and not Path(music).is_file():
        raise ValueError(f"The song file was not found: {music}")

    msg = out / "birthday_message.txt"
    msg.write_text("\n\n".join(" ".join(p.split()) for p in paras), encoding="utf-8")
    hero = v["files.hero_photos_dir"].strip()
    nested.pop("message", None)
    nested.setdefault("files", {}).update({
        "scene_json": str(ctx.scene), "scene_cache": str(ctx.cache_dir), "music": music,
        "hero_photos_dir": hero if hero and Path(hero).is_dir() else str(out / "_no_featured_photos"),
        "main_photo": ctx.main_image, "tiles_dir": ctx.tiles_dir})
    nested.setdefault("text", {})["message_file"] = str(msg)
    nested["text"]["opening_lines"] = [l for l in nested["text"].get("opening_lines", []) if l.strip()]
    nested["text"]["mosaic_caption"] = [l for l in nested["text"].get("mosaic_caption", []) if l.strip()]
    settings_path = out / "birthday_settings.json"
    settings_path.write_text(json.dumps(nested, indent=2), encoding="utf-8")

    pv = (ctx.job.get("plugin") or {}).get("preview") if ctx.preview else None
    name = f"{_safe(v['people.her_name'])}_birthday" + ("_preview" if pv else "")
    video = out / f"{name}.mp4"
    argv = ["film.py", "render", str(video)]
    if pv:
        t0 = float(pv.get("start", 40))
        argv += [str(t0), str(t0 + float(pv.get("seconds", 20)))]

    ctx.stage("render", "Rendering the birthday film" + (" (short preview, no music)" if pv else ""))
    log = open(out / "birthday_log.txt", "w", encoding="utf-8")      # film.py prints a lot; the job's stdout is for events
    env_old = {}
    import os
    for k, val in (("BIRTHDAY_SETTINGS", str(settings_path)), ("BIRTHDAY_WORK", str(out / "_work"))):
        env_old[k] = os.environ.get(k)
        os.environ[k] = val
    argv_old = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            runpy.run_path(str(Path(__file__).with_name("film.py")), run_name="__main__",
                           init_globals={"PROGRESS_CB": lambda i, n: ctx.progress("render", i, n)})
    except Exception:
        log.flush()
        tail = (out / "birthday_log.txt").read_text(encoding="utf-8", errors="replace")[-1200:]
        raise RuntimeError("The film could not be rendered.\n" + tail) from None
    finally:
        sys.argv = argv_old
        for k, val in env_old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
        log.close()
    return {"video": video}
