# Birthday video: a Mosaic Studio plugin

A personal birthday film for [Mosaic Studio](https://github.com/mahatosourav00/mosaic-video), as one continuous 3D shot
(vertical 1080 x 1920, 30 fps):

1. A dark night sky and an opening line.
2. The birthday person's name draws itself as a gold constellation, and the camera flies through it.
3. Their photos gather into a mosaic portrait, using one of Mosaic Studio's animation styles
   (**Swarm**, **Vortex**, **Wave** or **Zoom**).
4. Your message plays over the dimmed mosaic; beside each paragraph one framed photo lifts out of it.
5. A swoop onto the portrait, a pull-back, and a finale: "Happy Birthday, ..." and your sign-off. Music throughout.

Everything runs offline on your PC. Your photos never leave it.

## Use it

You need Mosaic Studio (the packaged app, or a source checkout).

1. Download or clone this folder anywhere.
2. In Mosaic Studio click **Add a plugin...** on the home screen and pick this folder.
   From a source checkout you can instead run
   `python -m mosaicvid plugins link "path\to\this\folder"`.
3. Choose **Birthday video** under *Made for a purpose*, then follow the steps:
   - **Pictures:** the folder of photos that become the mosaic tiles, and the main picture (usually a portrait).
   - **Names and message:** the birthday person's name, yours, an optional date line and the message (blank line between
     paragraphs; a first paragraph ending in a comma, like `Priya,`, is shown as a large greeting).
   - **Wording:** opening lines, captions and the finale text.
   - **Photos and music:** an optional folder of featured photos (named `01_...`, `02_...` in the order you want; without
     it good photos are chosen automatically) and a song.
   - **Look and pacing:** the mosaic animation style, colours, reading speed and grade.
4. On the review screen tick **Preview a short slice only** to render a few seconds without music first, then create the film.

The mosaic is built once and reused when you only change the message, music or look, so re-rendering takes about two
minutes instead of redoing everything.

## How it works

| File | Role |
|---|---|
| `plugin.json` | what Mosaic Studio shows: home card, form steps and fields, review summary, preview slice |
| `plugin.py` | `run(ctx)`: turns the form values into the settings `film.py` reads, then runs it with progress reporting |
| `film.py` | typography, grade, bloom, music and the encode |
| `world.py` | the 3D night-sky world, built on Mosaic Studio's `mosaicvid.animation` plan and camera for the chosen style |
| `settings.defaults.json` | every default; the form overrides them (delete a key to fall back to it) |
| `hero_photos/` | optional featured photos (see its README) |

The plugin format is documented at the top of
[`mosaicvid/plugins.py`](https://github.com/mahatosourav00/mosaic-video/blob/main/mosaicvid/plugins.py).

### Run it without the window

Put your values in a `settings.json` next to `film.py` (same keys as `settings.defaults.json`, plus `files.scene_json` and
`files.scene_cache` pointing at a mosaic built by Mosaic Studio), then with Mosaic Studio's Python environment:

```
python film.py render out.mp4              # the whole film
python film.py render preview.mp4 40 70    # only seconds 40 to 70, no audio
python film.py stills 12 31 47             # single frames into work\stills
```

## Notes

- Windows only (it uses fonts from `C:\Windows\Fonts` and OpenGL 3.3).
- Not committed on purpose: your photos, message, settings, music and rendered films (see `.gitignore`).
- Not tested yet: a packaged Mosaic Studio rendering a full-length film (short slices work).
