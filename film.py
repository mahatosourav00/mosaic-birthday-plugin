"""Night Sky film: continuous 3D world (world.py) + typography + grade.   Settings: settings.json

  stills:  python film.py stills 3 8 20 ...                  -> work/stills/n_XXXX.png
  render:  python film.py render out.mp4 [t0 t1]             (t0 t1 = quick preview slice, no audio)

Part of the "birthday" Mosaic Studio plugin: plugin.py runs this file with runpy and injects PROGRESS_CB.
BIRTHDAY_SETTINGS = settings file to use, BIRTHDAY_WORK = scratch folder for stills.
"""
import os, sys, json, math, subprocess, threading, queue, time
import numpy as np, cv2, moderngl
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("BIRTHDAY_WORK") or os.path.join(os.environ.get("TEMP", ROOT), "birthday_work")
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
PROGRESS_CB = globals().get("PROGRESS_CB") or (lambda done, total: None)      # injected by plugin.py
FD = r"C:\Windows\Fonts"
W, H, FPS = 1080, 1920, 30


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_settings():
    d = json.load(open(os.path.join(ROOT, "settings.defaults.json"), encoding="utf-8"))
    p = os.environ.get("BIRTHDAY_SETTINGS", os.path.join(ROOT, "settings.json"))
    return deep_merge(d, json.load(open(p, encoding="utf-8")))


C = load_settings()


def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def path_of(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


IVORY, GOLD = hexrgb(C["colors"]["ivory"]), hexrgb(C["colors"]["gold"])
HER, ME = C["people"]["her_name"], C["people"]["my_name"]
TX, FN, TL, MO, GR, NS = C["text"], C["fonts"], C["timeline"], C["motion"], C["grade"], C["nightsky"]
MSG = path_of(TX["message_file"])
SERIF, SERIF_I, SANS_L = FN["serif"], FN["serif_italic"], FN["sans_light"]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    return smooth((t - a) / (b - a))


# ================================================================ GPU: text + post
ctx = moderngl.create_standalone_context()
print("GL:", ctx.info["GL_RENDERER"], flush=True)

QUAD_VS = """
#version 330
in vec2 p; uniform vec4 rect; uniform vec2 res; out vec2 uv;
void main(){ vec2 px = rect.xy + p*rect.zw; gl_Position = vec4(px.x/res.x*2.-1., 1.-px.y/res.y*2., 0., 1.); uv = p; }
"""
TEX_FS = """
#version 330
in vec2 uv; uniform sampler2D tex; uniform int mode; uniform float alpha; uniform vec3 tint; uniform float shine; uniform vec2 res;
out vec4 o;
void main(){
  if(mode==0){ vec4 c = texture(tex, uv); vec3 rgb = c.rgb*tint;
    if(shine > -1.){ vec2 sp = vec2(gl_FragCoord.x/res.x, 1.-gl_FragCoord.y/res.y); float d = (sp.x + sp.y*0.25) - shine; rgb += exp(-d*d/0.0016)*vec3(.9,.8,.55); }
    o = vec4(rgb, c.a*alpha); }
  else if(mode==1){ o = vec4(0.,0.,0., 0.55*alpha*texture(tex, uv).r); }
  else if(mode==4){ o = vec4(tint, texture(tex, uv).r*alpha); }
  else if(mode==5){ float sy = pow(clamp((gl_FragCoord.y/res.y - 0.0), 0., 1.), 1.); o = vec4(0.,0.,0., alpha*pow(clamp(1.-gl_FragCoord.y/res.y*1.15,0.,1.),1.6)*0.0 + alpha*smoothstep(0.62,0.0,gl_FragCoord.y/res.y)); }
  else { o = vec4(tint, alpha); }
}
"""
BLIT_FS = """
#version 330
uniform sampler2D tex; uniform vec2 res; out vec4 o;
void main(){ o = vec4(texelFetch(tex, ivec2(gl_FragCoord.xy), 0).rgb, 1.); }
"""
POST_VS = """
#version 330
in vec2 p; out vec2 uv; void main(){ gl_Position = vec4(p*2.-1., 0., 1.); uv = p; }
"""
POST_FS = """
#version 330
in vec2 uv; uniform sampler2D acc; uniform float fade; uniform float seed;
uniform vec3 warm; uniform vec3 teal; uniform float vig; uniform float grain; uniform float bloom; uniform float bthr; uniform float ca;
out vec4 o;
float h(vec2 x){ return fract(sin(dot(x, vec2(12.9898,78.233)))*43758.5453); }
void main(){
  vec2 u = uv; vec2 c = u-.5; vec3 f;
  f.r = textureLod(acc, .5 + c*(1.+ca), 0.).r; f.g = textureLod(acc, u, 0.).g; f.b = textureLod(acc, .5 + c*(1.-ca), 0.).b;
  f *= 255.;
  vec3 b = (textureLod(acc, u, 3.).rgb*.45 + textureLod(acc, u, 5.).rgb*.55)*255.;
  f += max(b - bthr, 0.)*bloom*vec3(1.,.93,.82);
  float lum = (f.r+f.g+f.b)/765.;
  f *= warm; f += (1.-lum)*teal;
  float r = length(uv*2.-1.);
  f *= clamp(1.-vig*pow(clamp(r-.35,0.,1.),1.4), .5, 1.);
  vec2 g = gl_FragCoord.xy;
  f += (h(g+seed) + h(g*1.37+seed*3.1) - 1.)*7.8*grain;
  f *= 1.-fade;
  o = vec4(clamp(f,0.,255.)/255., 1.);
}
"""
YUV_FS = """
#version 330
uniform sampler2D src; uniform vec2 size; uniform int plane; out vec4 o;
void main(){
  vec3 c = texture(src, vec2(gl_FragCoord.x/size.x, 1.-gl_FragCoord.y/size.y)).rgb; float y = dot(c, vec3(.2126,.7152,.0722)); float v;
  if(plane==0) v = (16.+219.*y)/255.;
  else if(plane==1) v = (128.+224.*(c.b-y)/1.8556)/255.;
  else v = (128.+224.*(c.r-y)/1.5748)/255.;
  o = vec4(v,0.,0.,1.);
}
"""
_quad = ctx.buffer(np.array([0, 0, 1, 0, 0, 1, 1, 1], np.float32).tobytes())


def mk(vs, fs):
    p = ctx.program(vertex_shader=vs, fragment_shader=fs)
    return p, ctx.vertex_array(p, [(_quad, "2f", "p")])


p_tex, vao_tex = mk(QUAD_VS, TEX_FS)
p_blit, vao_blit = mk(QUAD_VS, BLIT_FS)
p_post, vao_post = mk(POST_VS, POST_FS)
p_yuv, vao_yuv = mk(POST_VS, YUV_FS)
for p in (p_tex, p_blit):
    p["res"].value = (W, H)
p_tex["shine"].value = -5.0


def make_tex(arr, comps, dtype="f1"):
    t = ctx.texture((arr.shape[1], arr.shape[0]), comps, arr.tobytes(), dtype=dtype)
    t.filter = (moderngl.LINEAR, moderngl.LINEAR)
    t.repeat_x = t.repeat_y = False
    return t


ACC_T = ctx.texture((W, H), 4, dtype="f2")
ACC_T.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
ACC_T.repeat_x = ACC_T.repeat_y = False
ACC = ctx.framebuffer(color_attachments=[ACC_T])
OUT_T = ctx.texture((W, H), 4)
OUT_T.filter = (moderngl.LINEAR, moderngl.LINEAR)
OUT = ctx.framebuffer(color_attachments=[OUT_T])
YUV_FBO = []
for _w, _h in ((W, H), (W // 2, H // 2), (W // 2, H // 2)):
    _t = ctx.texture((_w, _h), 1)
    YUV_FBO.append((ctx.framebuffer(color_attachments=[_t]), _w, _h))


def blend_over():
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


def draw_layer(L, cx, y, alpha=1.0, dy=0.0, blur=0.0, shine=-5.0):
    if alpha <= 0.002:
        return
    blend_over()
    p_tex["rect"].value = (int(cx - L["w"] / 2), int(y) + dy, L["w"], L["h"])
    p_tex["tint"].value = (1, 1, 1)
    p_tex["alpha"].value = alpha * (1 - 0.7 * blur)
    p_tex["mode"].value = 1
    L["sh"].use(0); p_tex["tex"].value = 0
    vao_tex.render(moderngl.TRIANGLE_STRIP)
    if blur > 0.02:
        p_tex["mode"].value = 4; p_tex["tint"].value = L["color"]; p_tex["alpha"].value = alpha * blur * 0.9
        vao_tex.render(moderngl.TRIANGLE_STRIP)
    p_tex["mode"].value = 0; p_tex["tint"].value = (1, 1, 1); p_tex["alpha"].value = alpha * (1 - blur)
    p_tex["shine"].value = shine
    L["tex"].use(0)
    vao_tex.render(moderngl.TRIANGLE_STRIP)
    p_tex["shine"].value = -5.0


def draw_scrim(alpha):
    if alpha <= 0.002:
        return
    blend_over()
    p_tex["rect"].value = (0, 0, W, H)
    p_tex["mode"].value = 5
    p_tex["alpha"].value = alpha
    vao_tex.render(moderngl.TRIANGLE_STRIP)


def F(name, size):
    return ImageFont.truetype(os.path.join(FD, name), size)


def tracked(text, fnt, color, tracking=0):
    widths = [fnt.getlength(c) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    asc, desc = fnt.getmetrics()
    pad = 44
    img = Image.new("RGBA", (int(total) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad
    for c, w in zip(text, widths):
        d.text((x, pad), c, font=fnt, fill=color + (255,))
        x += w + tracking
    return img


def line_layer(text, fnt, color, tracking=0):
    img = tracked(text, fnt, color, tracking)
    a = np.array(img)
    sh = cv2.GaussianBlur(a[..., 3].astype(np.float32) / 255, (0, 0), 11)
    return dict(tex=make_tex(a, 4), sh=make_tex((sh * 255).astype(np.uint8)[..., None], 1),
                w=img.width, h=img.height, color=tuple(c / 255 for c in color))


def wrap(text, fnt, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if fnt.getlength(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def reveal(t, start, dur=1.1, blur_amt=None):
    p = ramp(t, start, start + dur)
    ba = MO["word_blur_in"] if blur_amt is None else blur_amt
    return p, (1 - p) * 14, (1 - ramp(t, start, start + dur * 0.9)) * ba


# ================================================================ timeline
FIN_D = TL["finale_length"]
STAG = MO["word_reveal_stagger"]
T_NAME_IN, T_NAME_DRAW, T_HOLD_END = NS["name_in"], NS["name_draw"], NS["name_hold_end"]
T_E0, D_FORM = NS["formation_start"], NS["formation_length"]     # the mosaic engine's formation animation runs T_E0 .. T_E0 + D_FORM
T_GREET = T_E0 + D_FORM - 3.0
T_MSG0 = T_GREET + NS["greeting_length"]


def build_cards():
    paras = [" ".join(p.split()) for p in open(MSG, encoding="utf-8-sig").read().split("\n\n") if p.strip()]
    cards, t = [], T_MSG0 + 0.8
    for i, p in enumerate(paras):
        sal = i == 0 and p.endswith(",")
        fnt = F(SERIF_I, FN["salutation_size"] if sal else FN["message_size"])
        color = GOLD if sal else IVORY
        lines = [p] if sal else wrap(p, fnt, 880)
        lh = 170 if sal else int(fnt.size * 1.64)
        words, k = [], 0
        for li, line in enumerate(lines):
            x0 = (W - fnt.getlength(line)) / 2
            pos = 0
            for w in line.split():
                idx = line.index(w, pos)
                pos = idx + len(w)
                L = line_layer(w, fnt, color)
                words.append(dict(L=L, cx=x0 + fnt.getlength(line[:idx]) + L["w"] / 2 - 44, line=li, k=k))
                k += 1
        nw = len(p.split())
        dur = TL["salutation_seconds"] if sal else 0.3 + STAG * nw + 0.5 + TL["reading_seconds_per_word"] * nw + 0.7
        cards.append(dict(t0=t, t1=t + dur, lh=lh, nlines=len(lines), words=words, sal=sal))
        t += dur
    return cards


CARDS = build_cards()
T_MSG1 = CARDS[-1]["t1"]
T_FIN0 = T_MSG1 + 0.2
TOTAL = T_FIN0 + FIN_D
lifts = []
for j, c in enumerate([c for c in CARDS if not c["sal"]]):
    ny = 0.60 if j % 2 == 0 else -0.60
    lifts.append((c["t0"] - 0.6, c["t1"] + 0.9, 0.0, ny, 0.86))
TLW = dict(name_in=T_NAME_IN, name_draw=T_NAME_DRAW, name_hold_end=T_HOLD_END, form0=T_E0, form_len=D_FORM,
           msg0=T_MSG0, msg1=T_MSG1, fin0=T_FIN0, end=TOTAL,
           name_z=NS["name_z"], name_y=NS["name_y"], name_width=NS["name_width"],
           message_exposure=NS["message_exposure"], final_width=NS["final_mosaic_width"], lifts=lifts,
           style=NS.get("style", "swarm"))

from world import World  # noqa: E402  (needs the GL context above to exist first)

WORLD = World(ctx, W, H, TLW, path_of(C["files"]["scene_json"]), path_of(C["files"]["scene_cache"]), HER,
              os.path.join(FD, SERIF), grade=C["mosaic_grade"], hero_dir=path_of(C["files"]["hero_photos_dir"]))
print(f"cards={len(CARDS)} lifts={len(lifts)} formation={T_E0:.1f}-{T_E0 + D_FORM:.1f}s msg={T_MSG0:.1f}-{T_MSG1:.1f}s TOTAL={TOTAL:.1f}s", flush=True)

# ================================================================ typography assets
OPEN_LINES = [line_layer(l, F(SERIF_I, 58), IVORY) for l in TX["opening_lines"]]
DATE_L = line_layer(C["people"]["date_text"], F(SANS_L, 40), IVORY, 14)
SUB_L = line_layer(TX["subtitle"], F(SERIF_I, 40), IVORY, 3)
CAP = [line_layer(l, F(SERIF_I, 66), IVORY) for l in TX["mosaic_caption"]]
GREET_A = line_layer(TX["mosaic_greeting"], F(SERIF_I, 76), IVORY)
_nf = F(SERIF, 170)
while _nf.getlength(HER) > 900:
    _nf = F(SERIF, _nf.size - 6)
GREET_B = line_layer(HER, _nf, GOLD)
FIN_A = line_layer(TX["finale_line1"], F(SERIF_I, 66), IVORY)
FIN_B = line_layer(TX["finale_line2"], F(SERIF_I, 136), GOLD)
FIN_C = line_layer(TX["signoff_lead"], F(SERIF_I, 40), IVORY)
FIN_D_L = line_layer(ME, F(SERIF_I, 70), GOLD)


# ================================================================ frame
def draw_text(t):
    # opening line in the dark sky
    fo = 1 - ramp(t, T_NAME_IN - 1.0, T_NAME_IN + 0.8)
    for i, L in enumerate(OPEN_LINES):
        a, dy, bl = reveal(t, 1.6 + 0.6 * i, 1.8)
        draw_layer(L, W / 2, 880 + i * 100, a * fo, dy, bl)
    # date under the name constellation (fades as the camera starts to fly through it)
    fo2 = 1 - ramp(t, T_HOLD_END + 1.5, T_HOLD_END + 4.0)
    a, dy, bl = reveal(t, T_HOLD_END - 4.0, 1.6)
    draw_layer(DATE_L, W / 2, 975, a * fo2, dy, bl)
    a, dy, bl = reveal(t, T_HOLD_END - 3.0, 1.6)
    draw_layer(SUB_L, W / 2, 1065, a * fo2 * 0.8, dy, bl)
    # greeting once the mosaic has formed
    g = T_GREET
    draw_scrim(0.55 * ramp(t, g, g + 1.5) * (1 - ramp(t, T_MSG0 - 2.0, T_MSG0 - 0.3)))
    fc = 1 - ramp(t, g + 6.2, g + 7.4)
    for i, L in enumerate(CAP):
        a, dy, bl = reveal(t, g + 1.0 + 0.4 * i, 1.4)
        draw_layer(L, W / 2, 1400 + i * 100, a * fc, dy, bl)
    fg = 1 - ramp(t, T_MSG0 - 1.8, T_MSG0 - 0.3)
    a, dy, bl = reveal(t, g + 7.6, 1.5)
    draw_layer(GREET_A, W / 2, 1330, a * fg, dy, bl)
    a, dy, bl = reveal(t, g + 8.4, 1.8)
    sh = -0.3 + 1.6 * ramp(t, g + 10.0, g + 12.0) if (MO["title_shine"] > 0 and g + 10.0 < t < g + 12.0) else -5.0
    draw_layer(GREET_B, W / 2, 1400, a * fg, dy, bl, sh)
    # message
    for c in CARDS:
        if c["t0"] - 0.1 <= t <= c["t1"] + 0.1:
            top = H * 0.47 - (c["nlines"] * c["lh"]) / 2
            fo3 = 1 - ramp(t, c["t1"] - 0.9, c["t1"])
            for wd in c["words"]:
                a, dy, bl = reveal(t, c["t0"] + 0.4 + STAG * wd["k"], 1.0)
                draw_layer(wd["L"], wd["cx"], top + wd["line"] * c["lh"], a * fo3, dy - (1 - fo3) * 10, bl)
    # finale
    u = t - T_FIN0                         # camera pulls back until u = 17, so the dedication appears after that
    a, dy, bl = reveal(u, 13.0, 1.6)
    draw_layer(FIN_A, W / 2, 1385, a, dy, bl)
    a, dy, bl = reveal(u, 14.0, 2.0)
    shb = -0.3 + 1.6 * ramp(u, 16.0, 18.2) if (MO["title_shine"] > 0 and 16.0 < u < 18.2) else -5.0
    draw_layer(FIN_B, W / 2, 1450, a, dy, bl, shb)
    a, dy, bl = reveal(u, 17.0, 1.6)
    draw_layer(FIN_C, W / 2, 1660, a * 0.85, dy, bl)
    a, dy, bl = reveal(u, 17.8, 1.8)
    draw_layer(FIN_D_L, W / 2, 1695, a, dy, bl)


def render(t, yuv=False):
    tex = WORLD.render(t)
    ACC.use()
    ctx.viewport = (0, 0, W, H)
    ctx.disable(moderngl.BLEND | moderngl.DEPTH_TEST)
    tex.use(0)
    p_blit["tex"].value = 0
    p_blit["rect"].value = (0, 0, W, H)
    vao_blit.render(moderngl.TRIANGLE_STRIP)
    draw_text(t)
    ACC_T.build_mipmaps()
    OUT.use()
    ctx.disable(moderngl.BLEND)
    ACC_T.use(0)
    p_post["acc"].value = 0
    p_post["fade"].value = ramp(t, TOTAL - 3.5, TOTAL - 0.2)
    p_post["seed"].value = float(int(t * FPS) % 6) * 7.31
    p_post["warm"].value = tuple(GR["warmth"]); p_post["teal"].value = tuple(GR["shadow_tint"])
    p_post["vig"].value, p_post["grain"].value = GR["vignette"], GR["grain"]
    p_post["bloom"].value, p_post["bthr"].value, p_post["ca"].value = GR["bloom"], GR["bloom_threshold"], GR["chromatic_aberration"]
    vao_post.render(moderngl.TRIANGLE_STRIP)
    if not yuv:
        return OUT.read(components=3)[:]
    OUT_T.use(0)
    p_yuv["src"].value = 0
    planes = []
    for i, (fbo, w, h) in enumerate(YUV_FBO):
        fbo.use()
        ctx.viewport = (0, 0, w, h)
        p_yuv["size"].value = (w, h)
        p_yuv["plane"].value = i
        vao_yuv.render(moderngl.TRIANGLE_STRIP)
        planes.append(fbo.read(components=1))
    ctx.viewport = (0, 0, W, H)
    return b"".join(planes)


# ================================================================ audio + output
def audio_filter():
    """Song plays from 0:00; at the end of assembly it jumps (bar-aligned, crossfaded) to the song's climax so the
    portrait reveal lands on it; the rest follows continuously through the quiet ending of the track."""
    M = C["music"]
    if M.get("straight", False):
        # plain playback from 0:00, no jump and no tempo change: the song's own build (about 1:40-2:05) carries the later lines
        return (f"[1:a]atrim=0:{TOTAL:.3f},asetpts=PTS-STARTPTS,afade=t=in:st=0:d={M['fade_in']},"
                f"afade=t=out:st={TOTAL - M['fade_out']:.3f}:d={M['fade_out']},volume={M['volume']}[a]")
    XF = 2.4
    bar0, bar = 0.706, 2.424
    k_cut = round((T_GREET - 0.3 - bar0) / bar)
    T1 = bar0 + bar * k_cut                           # video time of the cut (song time == video time before it)
    S2 = bar0 + bar * M["climax_bar"]                 # song time we jump to
    need = TOTAL - (T1 - XF / 2)
    avail = M["song_length"] - (S2 - XF / 2)
    tempo = min(1.0, avail / need) if need > 0 else 1.0
    seg1 = f"[1:a]atrim=0:{T1 + XF / 2:.3f},asetpts=PTS-STARTPTS[a0];"
    seg2 = f"[1:a]atrim={S2 - XF / 2:.3f},asetpts=PTS-STARTPTS,atempo={tempo:.4f}[a1];"
    return (seg1 + seg2 + f"[a0][a1]acrossfade=d={XF}:c1=tri:c2=tri[ax];"
            f"[ax]afade=t=in:st=0:d={M['fade_in']},afade=t=out:st={TOTAL - M['fade_out']:.3f}:d={M['fade_out']},"
            f"volume={M['volume']}[a]")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "stills":
        os.makedirs(WORK + r"\stills", exist_ok=True)
        for s in sys.argv[2:]:
            t = float(s)
            Image.frombytes("RGB", (W, H), render(t)).transpose(Image.FLIP_TOP_BOTTOM).save(WORK + rf"\stills\n_{int(round(t * 10)):04d}.png")
            print("saved", t, flush=True)
    else:
        out = sys.argv[2]
        O = C["output"]
        rest = sys.argv[3:]
        t0, t1 = (float(rest[0]), float(rest[1])) if len(rest) >= 2 else (0.0, TOTAL)
        preview = len(rest) >= 2
        mr = O["maxrate_mbps"]
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(O["crf"]), "-maxrate", f"{mr}M", "-bufsize", f"{mr * 2}M"]
        cmd = [FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
        tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
        if preview:
            cmd += vcodec + tags + [out]
        elif not C["files"].get("music"):                                # no song chosen: a silent film
            cmd += vcodec + tags + ["-an", "-t", f"{TOTAL:.3f}", "-movflags", "+faststart", out]
        else:
            cmd += ["-i", path_of(C["files"]["music"]), "-filter_complex", audio_filter(), "-map", "0:v", "-map", "[a]"] + vcodec + tags + \
                   ["-c:a", "aac", "-b:a", "192k", "-t", f"{TOTAL:.3f}", "-movflags", "+faststart", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        q = queue.Queue(maxsize=12)

        def writer():
            while True:
                b = q.get()
                if b is None:
                    break
                p.stdin.write(b)

        th = threading.Thread(target=writer)
        th.start()
        i0, n = int(t0 * FPS), int(t1 * FPS)
        st = time.time()
        for i in range(i0, n):
            q.put(render(i / FPS, yuv=True))
            if (i - i0) % 15 == 0:
                PROGRESS_CB(i - i0, n - i0)
            if (i - i0) % 150 == 0:
                print(f"{i - i0}/{n - i0}  {(i - i0 + 1) / (time.time() - st):.1f} fps", flush=True)
        q.put(None)
        th.join()
        p.stdin.close()
        p.wait()
        PROGRESS_CB(n - i0, n - i0)
        if p.returncode:
            raise RuntimeError(f"ffmpeg stopped with code {p.returncode}")
        print(f"done {out} in {time.time() - st:.0f}s", flush=True)
