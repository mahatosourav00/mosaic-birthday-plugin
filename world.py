"""Night-sky world: one continuous 3D scene (GPU / ModernGL).

Timeline:  name constellation  ->  camera flies through it  ->  the mosaic engine's own formation animation
(tiles fly in large on arcs, showcase photos sweep past, orbiting camera)  ->  message with photos that lift out
of the mosaic  ->  swoop in and pull back on the finished portrait.

The formation uses mosaicvid.animation.swarm exactly as built earlier; this file only adds the night sky around it,
the star state before each tile flies, and a continuous camera that hands over smoothly.
render(t) draws everything into self.tex (RGBA16F). Typography + grade are done by film.py on top.
"""
import sys, math, os, copy
import numpy as np, cv2, moderngl
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.interpolate import PchipInterpolator

from mosaicvid.solver import load_scene, blend_alpha                                         # noqa: E402
from mosaicvid.animation import (make_plan, _world, _subject_centroid, full_distance,  # noqa: E402
                                 FOV_DEG, TAN)
from mosaicvid.render_gl import perspective, look_at                                         # noqa: E402
from mosaicvid.preprocess import load_tiles                                                  # noqa: E402
from mosaicvid.settings import Settings                                                      # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def smooth(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3 - 2 * u)


def smoother(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)       # zero velocity AND zero acceleration at both ends


def ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_io(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ================================================================ shaders
TILE_COMMON = """
vec3 tile_pos(out float p, out float e, out float sc) {
    p = clamp((u_time - i_time.x) / i_time.y, 0.0, 1.0);
    bool hero = i_ext.z >= 0.5;
    float pp = p + 0.8 * sin(6.2831853 * p) / 6.2831853;
    e = hero ? pp * pp * (3.0 - 2.0 * pp) : 1.0 - pow(1.0 - p, 3.0);
    vec3 idle = 0.7 * vec3(sin(u_time * 0.31 + i_x.x), cos(u_time * 0.27 + i_x.x * 1.7),
                           sin(u_time * 0.23 + i_x.x * 0.6)) * (1.0 - smoothstep(0.0, 0.08, p));
    vec3 E = vec3(i_end.xy, 0.0);
    sc = mix(i_ext.x, 1.0, smoothstep(i_ext.y, 1.0, p));
    return (1.0 - e) * (1.0 - e) * (i_start.xyz + idle) + 2.0 * (1.0 - e) * e * i_mid + e * e * E;
}
"""
TILE_VS = """
#version 330
in vec2 in_vert;
in vec4 i_end; in vec4 i_start; in vec4 i_time; in vec3 i_off; in vec3 i_ext; in vec3 i_mid; in vec3 i_x;
uniform mat4 u_vp; uniform float u_time; uniform float u_gap;
uniform vec3 u_eye; uniform vec3 u_fwd; uniform float u_focus; uniform float u_dof;
out vec3 v_uv; out vec3 v_off; out float v_light; out float v_blur; out float v_fog; out float v_a;
flat out float v_hero;
""" + TILE_COMMON + """
void main() {
    if (u_time < i_time.x || (u_time > i_x.y && u_time < i_x.z)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_uv = vec3(0.0); v_off = vec3(0.0); v_light = 1.0; v_blur = 0.0; v_fog = 0.0;
        v_a = 0.0; v_hero = -1.0; return; }
    bool hero = i_ext.z >= 0.5;
    v_hero = hero ? i_ext.z - 1.0 : -1.0;
    float p, e, sc;
    vec3 pos = tile_pos(p, e, sc);
    float ang = i_start.w * (1.0 - e);
    float tilt = i_time.w * (1.0 - e);
    float side = max(i_end.z - u_gap * (i_end.z >= 0.99 ? 1.0 : 0.5), 0.02);
    vec3 l = vec3(in_vert * side * sc, 0.0);
    float ct = cos(tilt), st = sin(tilt);
    l = vec3(ct * l.x + st * l.z, l.y, -st * l.x + ct * l.z);
    float ca = cos(ang), sa = sin(ang);
    l = vec3(ca * l.x - sa * l.y, sa * l.x + ca * l.y, l.z);
    vec3 wp = pos + l;
    gl_Position = u_vp * vec4(wp, 1.0);
    float vx = in_vert.x + 0.5;
    if (i_time.z > 0.5) vx = 1.0 - vx;
    v_uv = vec3(vx, 0.5 - in_vert.y, i_end.w);
    v_off = i_off;
    v_a = smoothstep(0.0, 0.2, p);
    float ts = u_time - (i_time.x + i_time.y);
    v_light = 1.0 + (ts > 0.0 ? 0.35 * exp(-ts * 6.0) : 0.0);
    float depth = dot(wp - u_eye, u_fwd);
    float off = abs(depth - u_focus) / u_focus;
    v_blur = hero ? 0.0 : clamp((off - 0.10) * 2.0, 0.0, 1.0) * 2.6 * u_dof;
    v_fog = hero ? 0.0 : clamp((depth - u_focus) / (u_focus * 0.9), 0.0, 1.0) * 0.55 * u_dof;
}
"""
TILE_FS = """
#version 330
in vec3 v_uv; in vec3 v_off; in float v_light; in float v_blur; in float v_fog; in float v_a;
flat in float v_hero;
uniform sampler2DArray u_tiles; uniform sampler2DArray u_hero; uniform vec3 u_bgc; uniform float u_expo; uniform float u_sat;
uniform vec3 u_tint; uniform float u_contrast; uniform float u_yellow;
out vec4 f_color;
vec3 deyellow(vec3 c, float k) {      // calm yellow/chartreuse walls without touching orange skin tones
    float y = smoothstep(0.03, 0.16, min(c.r, c.g) - c.b) * (1.0 - smoothstep(0.04, 0.22, abs(c.r - c.g)));
    float l = dot(c, vec3(.2126, .7152, .0722));
    return mix(c, vec3(l) * vec3(0.98, 1.0, 1.06), k * y);
}
void main() {
    if (v_a < 0.02) discard;
    vec3 c;
    if (v_hero > -0.5) c = texture(u_hero, vec3(v_uv.xy, floor(v_hero + 0.5))).rgb;
    else c = textureLod(u_tiles, v_uv, v_blur + 0.5).rgb;
    c = deyellow(clamp((c + v_off) * v_light, 0.0, 1.5), u_yellow) * u_tint;
    float l = dot(c, vec3(.2126, .7152, .0722));
    c = max((mix(vec3(l), c, u_sat) - 0.4) * u_contrast + 0.4, 0.0) * u_expo;
    f_color = vec4(mix(c, u_bgc, v_fog), v_a);
}
"""
HALO_VS = """
#version 330
in vec2 in_vert;
in vec4 i_end; in vec4 i_start; in vec4 i_time; in vec3 i_off; in vec3 i_ext; in vec3 i_mid; in vec3 i_x;
uniform mat4 u_vp; uniform float u_time; uniform float u_gap; uniform vec3 u_right; uniform vec3 u_up;
uniform sampler2DArray u_tiles; uniform float u_stars; uniform vec3 u_eye; uniform vec3 u_fwd;
out vec2 v_q; out float v_a; out vec3 v_col;
""" + TILE_COMMON + """
void main() {
    float p, e, sc;
    vec3 pos = tile_pos(p, e, sc);
    float star = 1.0 - smoothstep(0.0, 0.30, p);
    float tw = 0.6 + 0.4 * sin(u_time * (1.1 + fract(i_x.x) * 1.3) + i_x.x * 9.0);
    float depth = dot(pos - u_eye, u_fwd);
    float mag = 0.45 + 1.5 * pow(fract(i_x.x * 0.731), 3.0);          // few bright stars, many faint
    float vis = step(fract(i_x.x * 0.137), 0.30);                     // 30% of the photos are visible stars from the start
    float appear = vis > 0.5 ? 1.0 : smoothstep(i_time.x - 3.5, i_time.x - 0.5, u_time);
    float r = depth * 0.0046 * mag * (0.75 + 0.5 * tw) * (1.0 + 0.7 * smoothstep(0.0, 0.3, p));
    v_a = star * appear * u_stars * (0.22 + 0.5 * tw) * (0.55 + 0.45 * mag) * smoothstep(3.0, 20.0, depth);
    if (i_ext.z >= 0.5 || v_a < 0.01 || (u_time > i_x.y && u_time < i_x.z)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_q = vec2(0.0); v_col = vec3(0.0); return; }
    vec3 avg = textureLod(u_tiles, vec3(0.5, 0.5, i_end.w), 8.0).rgb;
    v_col = mix(avg * 1.6, vec3(1.0, 0.90, 0.74), 0.55);
    v_q = in_vert * 2.0;
    gl_Position = u_vp * vec4(pos + (u_right * in_vert.x + u_up * in_vert.y) * r * 2.0, 1.0);
}
"""
HALO_FS = """
#version 330
in vec2 v_q; in float v_a; in vec3 v_col; out vec4 f_color;
void main() {
    float r = length(v_q);
    float core = exp(-r * r * 22.0) * 1.4;
    float halo = exp(-r * r * 4.0) * 0.30;
    float spike = (exp(-abs(v_q.x) * 30.0) * exp(-abs(v_q.y) * 2.6) + exp(-abs(v_q.y) * 30.0) * exp(-abs(v_q.x) * 2.6)) * 0.55;
    f_color = vec4(v_col * (core + halo + spike) * v_a, 1.0);
}
"""
SPRITE_VS = """
#version 330
in vec2 in_vert; in vec4 s_a; in vec4 s_b;      // s_a: pos.xyz, px size ; s_b: phase, brightness, appear time, kind
uniform mat4 u_vp; uniform float u_time; uniform vec2 u_res; uniform float u_gain; uniform float u_mode; uniform vec3 u_cam; uniform vec3 u_col;
out vec2 v_q; out float v_a; out vec3 v_c;
void main() {
    vec4 c = u_vp * vec4(s_a.xyz, 1.0);
    float tw = 0.65 + 0.35 * sin(u_time * (0.7 + fract(s_b.x * 3.1) * 1.6) + s_b.x * 6.28);
    float appear = smoothstep(s_b.z, s_b.z + 0.6, u_time);
    float near = smoothstep(2.0, 24.0, distance(s_a.xyz, u_cam));          // fade out right at the lens
    float sz = s_a.w * (u_mode > 0.5 ? 1.0 : (0.7 + 0.5 * tw));
    c.xy += in_vert * sz * 2.0 / u_res * c.w * (appear > 0.0 ? 1.0 : 0.0);
    v_q = in_vert * 2.0; v_a = u_gain * s_b.y * (u_mode > 0.5 ? 1.0 : tw) * appear * near;
    v_c = mix(u_col, vec3(1.0, 0.80, 0.55), fract(s_b.x * 7.7) * 0.6);
    gl_Position = c;
}
"""
SPRITE_FS = """
#version 330
in vec2 v_q; in float v_a; in vec3 v_c; uniform float u_mode; out vec4 f_color;
void main() {
    float r = length(v_q);
    float a;
    if (u_mode > 0.5) a = (1.0 - smoothstep(0.55, 1.0, r)) * 0.55;                     // soft out-of-focus disc
    else a = exp(-r * r * 9.0) + 0.5 * exp(-(abs(v_q.x) * 22.0 + abs(v_q.y) * 3.0)) + 0.5 * exp(-(abs(v_q.y) * 22.0 + abs(v_q.x) * 3.0));
    f_color = vec4(v_c * a * v_a, 1.0);
}
"""
LINE_VS = """
#version 330
in vec2 in_vert;        // x: 0 = start, 1 = end ; y: -0.5 / +0.5 side
in vec4 l_a; in vec4 l_b;   // l_a: p0.xyz, appear ; l_b: p1.xyz, unused
uniform mat4 u_vp; uniform float u_time; uniform vec2 u_res; uniform float u_width; uniform vec3 u_cam;
out float v_a; out float v_s;
void main() {
    float g = smoothstep(l_a.w, l_a.w + 0.45, u_time);
    vec3 pe = mix(l_a.xyz, l_b.xyz, g);
    vec4 c0 = u_vp * vec4(l_a.xyz, 1.0), c1 = u_vp * vec4(pe, 1.0);
    vec2 s0 = c0.xy / c0.w * u_res * 0.5, s1 = c1.xy / c1.w * u_res * 0.5;
    vec2 d = s1 - s0; float len = max(length(d), 1e-4); vec2 n = vec2(-d.y, d.x) / len;
    vec4 c = mix(c0, c1, in_vert.x);
    c.xy += n * in_vert.y * u_width * 2.0 / u_res * c.w;
    gl_Position = c;
    float near = smoothstep(4.0, 30.0, distance(mix(l_a.xyz, pe, 0.5), u_cam));
    v_a = (g > 0.001 ? 1.0 : 0.0) * near; v_s = in_vert.y;
}
"""
LINE_FS = """
#version 330
in float v_a; in float v_s; uniform float u_gain; uniform vec3 u_col; out vec4 f_color;
void main() { float a = exp(-v_s * v_s * 10.0); f_color = vec4(u_col * a * v_a * u_gain, 1.0); }
"""
BACK_VS = """
#version 330
in vec2 in_vert; out vec2 v_uv;
void main() { gl_Position = vec4(in_vert * 2.0, 0.999, 1.0); v_uv = in_vert + 0.5; }
"""
BACK_FS = """
#version 330
in vec2 v_uv; uniform float u_time; uniform vec2 u_shift; uniform float u_glow; uniform vec2 u_gpos; uniform vec2 u_res;
out vec4 f_color;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float noise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y); }
float fbm(vec2 p) { float a = 0.5, s = 0.0; for (int i = 0; i < 5; i++) { s += a * noise(p); p = p * 2.03 + 11.7; a *= 0.5; } return s; }
void main() {
    vec2 uv = v_uv;
    vec3 top = vec3(0.010, 0.016, 0.045), mid = vec3(0.016, 0.050, 0.074), bot = vec3(0.006, 0.012, 0.022);
    vec3 col = mix(bot, top, smoothstep(0.0, 1.0, uv.y));
    col = mix(col, mid, exp(-pow((uv.y - 0.45) * 2.2, 2.0)));
    float n = fbm(uv * vec2(2.2, 3.6) + u_shift * 3.0 + vec2(0.0, u_time * 0.004));
    float n2 = fbm(uv * vec2(4.0, 6.0) - u_shift * 5.0 + 7.3 + u_time * 0.003);
    col += vec3(0.020, 0.075, 0.085) * pow(n, 2.6) * 1.7 + vec3(0.055, 0.028, 0.070) * pow(n2, 3.0) * 0.9;
    vec2 d = uv - u_gpos;
    col += vec3(0.90, 0.60, 0.26) * 0.075 * u_glow * exp(-dot(d * vec2(1.0, 0.55), d * vec2(1.0, 0.55)) * 6.0);
    col *= mix(0.55, 1.0, smoothstep(0.95, 0.15, length((uv - 0.5) * vec2(1.0, 1.1))));
    col += (hash(uv * u_res + fract(u_time)) - 0.5) * (4.0 / 255.0);
    f_color = vec4(col, 1.0);
}
"""
OVER_VS = """
#version 330
in vec2 in_vert; uniform mat4 u_vp; uniform vec2 u_size; out vec2 v_uv;
void main() { gl_Position = u_vp * vec4(in_vert * u_size, 0.05, 1.0); v_uv = vec2(in_vert.x + 0.5, 0.5 - in_vert.y); }
"""
OVER_FS = """
#version 330
in vec2 v_uv; uniform sampler2D u_main; uniform float u_ramp; uniform float u_expo; uniform float u_sat; uniform vec3 u_tint; uniform float u_contrast;
uniform float u_yellow; uniform float u_shine;
out vec4 f_color;
vec3 deyellow(vec3 c, float k) {
    float y = smoothstep(0.03, 0.16, min(c.r, c.g) - c.b) * (1.0 - smoothstep(0.04, 0.22, abs(c.r - c.g)));
    float l = dot(c, vec3(.2126, .7152, .0722));
    return mix(c, vec3(l) * vec3(0.98, 1.0, 1.06), k * y);
}
void main() {
    vec4 m = texture(u_main, v_uv); float a = m.a * u_ramp;
    vec3 c = deyellow(m.rgb, u_yellow) * u_tint;
    float l = dot(c, vec3(.2126, .7152, .0722));
    c = max((mix(vec3(l), c, u_sat) - 0.4) * u_contrast + 0.4, 0.0) * u_expo;
    float d = v_uv.x * 0.55 + v_uv.y * 0.45;
    float band = exp(-pow((d - u_shine) / 0.05, 2.0));
    f_color = vec4(mix(c, vec3(1.0), band * 0.7), max(a, band * 0.30));
}
"""
GLOW_VS = """
#version 330
in vec2 in_vert; uniform mat4 u_vp; uniform vec2 u_quad; out vec2 v_p;
void main() { v_p = in_vert * u_quad; gl_Position = u_vp * vec4(v_p, -0.08, 1.0); }
"""
GLOW_FS = """
#version 330
in vec2 v_p; uniform vec2 u_half; uniform float u_scale; uniform float u_mode; uniform float u_fade; uniform vec3 u_accent;
out vec4 f_color;
float sdf(vec2 p) { vec2 q = abs(p) - u_half; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0); }
void main() {
    if (u_mode < 0.5) { float d = sdf(v_p + vec2(0.0, 0.018 * u_scale)); float a = 0.75 * exp(-max(d, 0.0) / (0.05 * u_scale));
        f_color = vec4(0.0, 0.0, 0.0, a * u_fade); }
    else { float d = sdf(v_p); float g = exp(-max(d, 0.0) / (0.14 * u_scale)) * step(0.0, d);
        float rim = exp(-pow((d - 0.004 * u_scale) / (0.0035 * u_scale), 2.0));
        f_color = vec4((u_accent * 1.0 * g + vec3(1.0, 0.95, 0.88) * rim * 0.55) * u_fade, 1.0); }
}
"""
HERO_VS = """
#version 330
in vec2 in_vert; uniform mat4 u_vp; uniform vec3 u_pos; uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f;
uniform float u_size; uniform float u_rx; uniform float u_ry; uniform float u_scale;
out vec2 v_uv; out vec2 v_q;
void main() {
    vec3 l = vec3(in_vert * u_size * u_scale, 0.0);
    float cy = cos(u_ry), sy = sin(u_ry); l = vec3(cy * l.x + sy * l.z, l.y, -sy * l.x + cy * l.z);
    float cx = cos(u_rx), sx = sin(u_rx); l = vec3(l.x, cx * l.y - sx * l.z, sx * l.y + cx * l.z);
    gl_Position = u_vp * vec4(u_pos + u_r * l.x + u_u * l.y - u_f * l.z, 1.0);
    v_uv = vec2(in_vert.x + 0.5, 0.5 - in_vert.y); v_q = in_vert * 2.0;
}
"""
HERO_FS = """
#version 330
in vec2 v_uv; in vec2 v_q; uniform sampler2DArray u_hero; uniform float u_layer; uniform vec3 u_off; uniform float u_alpha; uniform float u_mode;
uniform float u_border; out vec4 f_color;
void main() {
    if (u_mode < 0.5) {                                              // soft shadow
        float d = max(abs(v_q.x), abs(v_q.y)); f_color = vec4(0.0, 0.0, 0.0, u_alpha * 0.55 * (1.0 - smoothstep(0.55, 1.0, d))); }
    else if (u_mode < 1.5) {                                         // warm glow
        float d = length(v_q); f_color = vec4(vec3(1.0, 0.78, 0.45) * u_alpha * 0.26 * exp(-d * d * 3.2), 1.0); }
    else {
        float b = u_border; vec2 q = v_uv;
        bool inside = q.x > b && q.x < 1.0 - b && q.y > b && q.y < 1.0 - b;
        vec3 c = inside ? texture(u_hero, vec3((q - b) / (1.0 - 2.0 * b), u_layer)).rgb + u_off : vec3(0.96, 0.93, 0.86);
        f_color = vec4(clamp(c, 0.0, 1.0), u_alpha);
    }
}
"""


def load_user_photos(folder, px=512):
    """Square-crop every image in `folder` (sorted by name) for use as hero photos. Missing folder / no images -> []."""
    if not folder or not os.path.isdir(folder):
        return [], []
    out, names = [], []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(IMG_EXT):
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(os.path.join(folder, f))).convert("RGB")
        except Exception:
            continue
        out.append(np.array(ImageOps.fit(im, (px, px), Image.LANCZOS, centering=(0.5, 0.4))))
        names.append(f)
    return out, names


class World:
    def __init__(self, ctx, W, H, tl, scene_json, cache_dir, name, name_font, grade=None, hero_dir=None, seed=7):
        self.ctx, self.W, self.H, self.tl = ctx, W, H, tl
        self.aspect = W / H
        g = grade or {}
        self.g_tint = tuple(float(v) * float(g.get("brightness", 1.0)) for v in g.get("tint", (1, 1, 1)))
        self.g_sat, self.g_contrast = float(g.get("saturation", 1.0)), float(g.get("contrast", 1.0))
        self.g_yellow = float(g.get("yellow_suppress", 0.0))
        self.rng = np.random.default_rng(seed)
        m, tiles, s = load_scene(scene_json)
        # look / animation settings come from the engine's current defaults, exactly like `mosaicvid video` does
        fresh = Settings()
        for k in ("hero_tiles", "hero_size", "flight_scale", "pulses", "pulse_depth", "dof", "frame_margin",
                  "backdrop", "glow", "particles", "anim_seed", "bg_color", "crf", "style"):
            setattr(s, k, getattr(fresh, k))
        s.style = tl.get("style", "swarm")            # any of the engine's animation styles: swarm | vortex | wave | zoom
        s.duration = float(tl["form_len"])
        s.video_aspect = "9:16"
        self.m, self.tiles, self.s = m, tiles, s
        self.plan = make_plan(m, tiles, s)
        wx, wy, size = _world(m)
        self.wx, self.wy, self.size = wx, wy, size
        self.N = len(wx)
        self.d_full = self.plan.d_full
        self._eng_state = self.plan.state              # the camera parameters of the chosen style (same camera the plan uses)
        self._tan = TAN
        self._build_camera()
        self._build_tiles(m, tiles)
        self._build_name(name, name_font)
        self._build_far()
        self._build_heroes(m, tiles, s, cache_dir, hero_dir)
        self._gl_setup(m, s)

    # ------------------------------------------------------------ continuous camera
    def _build_camera(self):
        tl = self.tl
        D, T0 = tl["form_len"], tl["form0"]
        tE = T0 + D
        th = tl["name_hold_end"]
        m0, m1, f0, end = tl["msg0"], tl["msg1"], tl["fin0"], tl["end"]
        INF = end + 120
        mk = lambda k: PchipInterpolator([a for a, _ in k], [b for _, b in k])
        self._A_R = mk([(0, 330), (th, 306), (INF, 306)])                     # name phase: slow dolly, dead-centre
        fin = self._eng_state(D)                                              # engine camera state at its last frame
        txf, tyf, Rf, yawf, pitf, rolf = fin
        R_final = self.d_full / tl.get("final_width", 0.68)
        self.R_final = R_final
        shift = (0.5 - 0.375) * 2 * R_final * TAN
        span = m1 - m0
        self._R = mk([(tE, Rf), (tE + 2, Rf), (m0, Rf * 0.97), (m1 - 6, 190), (m1, 200), (f0 + 5, 122), (f0 + 8, 126),
                      (f0 + 17, R_final), (INF, R_final)])
        self._yaw = mk([(tE, yawf), (tE + 2, yawf), (m0, 0.0), (m0 + span * 0.30, 0.30), (m0 + span * 0.65, -0.24), (m1, 0.10),
                        (f0 + 5, 0.0), (INF, 0.0)])
        self._pitch = mk([(tE, pitf), (tE + 2, pitf), (m0, 0.0), (m0 + span * 0.35, 0.10), (m0 + span * 0.7, -0.06), (m1, 0.03),
                          (f0 + 5, 0.0), (INF, 0.0)])
        self._tx = mk([(tE, txf), (INF, txf)])
        self._ty = mk([(tE, tyf), (f0 + 8, tyf), (f0 + 17, tyf - shift), (INF, tyf - shift)])
        self._roll = mk([(tE, rolf), (INF, rolf)])
        # the engine's own camera stops dead at 0.93 D; ease the last stretch into rest instead (cubic Hermite, zero end velocity)
        self._tA, self._tB = T0 + 0.80 * D, tE + 1.0
        h = 1.0 / 60
        sa = np.array(self._eng_state(0.80 * D), float)
        self._S_a = sa
        self._V_a = (np.array(self._eng_state(0.80 * D + h), float) - np.array(self._eng_state(0.80 * D - h), float)) / (2 * h)
        self._S_f = np.array(fin, float)
        me = tl.get("message_exposure", 0.36)
        kE = [(0, 1), (m0 - 0.5, 1), (m0 + 2.5, me), (m1 - 1.5, me), (f0 + 4.0, 1.0), (INF, 1.0)]
        kS = [(0, 1), (m0 - 0.5, 1), (m0 + 2.5, 0.72), (m1 - 1.5, 0.72), (f0 + 4.0, 1.0), (INF, 1.0)]
        self._expo, self._sat = mk(kE), mk(kS)

    def params(self, t):
        """-> (tx, ty, dist, yaw, pitch, roll). Name phase -> (blend) -> engine formation camera -> message/finale path."""
        tl = self.tl
        D, T0 = tl["form_len"], tl["form0"]
        tE, th = T0 + D, tl["name_hold_end"]
        BL = 2.0                                        # the blend runs past the engine start so velocities stay continuous
        if t < T0 + BL:
            s = max(t - T0, 0.0)
            tau = 2 * s * s / BL - s ** 3 / BL ** 2     # engine clock starts at rest and merges into normal speed at s = BL
            A = (0.0, 0.0, float(self._A_R(t)), 0.0, 0.0, 0.0)
            B = self._eng_state(tau)
            w = smoother((t - th) / (T0 + BL - th)) if t > th else 0.0
            return tuple(a * (1 - w) + b * w for a, b in zip(A, B))
        if t <= self._tA:
            return tuple(float(v) for v in self._eng_state(t - T0))
        if t <= self._tB:                               # smooth landing into the still frame
            dt = self._tB - self._tA
            x = (t - self._tA) / dt
            h00, h10, h01 = 2 * x ** 3 - 3 * x ** 2 + 1, x ** 3 - 2 * x ** 2 + x, -2 * x ** 3 + 3 * x ** 2
            v = h00 * self._S_a + h10 * dt * self._V_a + h01 * self._S_f
            return tuple(float(a) for a in v)
        hh = smooth(tE + 1, tE + 4, t) * (1 - smooth(tl["fin0"] + 8, tl["fin0"] + 14, t))
        tx, ty = float(self._tx(t)), float(self._ty(t))
        tx += hh * (0.35 * math.sin(t * 0.43) + 0.2 * math.sin(t * 1.1 + 1))
        ty += hh * (0.30 * math.sin(t * 0.37 + 2) + 0.15 * math.sin(t * 0.9))
        return (tx, ty, float(self._R(t)), float(self._yaw(t)), float(self._pitch(t)),
                float(self._roll(t)) + hh * 0.006 * math.sin(t * 0.29 + 0.5))

    def fov_deg(self, t):
        """Lens: constant in the name phase, then the style's own lens (vortex/wave start wide, zoom starts long),
        blended in exactly like the camera so there is no jump."""
        pf = self.plan.fov
        if pf is None:
            return FOV_DEG
        tl = self.tl
        T0, D, th, BL = tl["form0"], tl["form_len"], tl["name_hold_end"], 2.0
        if t < T0 + BL:
            s = max(t - T0, 0.0)
            tau = 2 * s * s / BL - s ** 3 / BL ** 2
            w = smoother((t - th) / (T0 + BL - th)) if t > th else 0.0
            return FOV_DEG * (1 - w) + float(pf(tau)) * w
        return float(pf(min(t - T0, D)))

    def camera(self, t):
        tx, ty, R, yaw, pitch, roll = self.params(t)
        target = np.array([tx, ty, 0.0])
        eye = target + R * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        return eye, target, roll

    def basis(self, eye, target, roll):
        f = target - eye; f = f / np.linalg.norm(f)
        up = np.array([math.sin(roll), math.cos(roll), 0.0])
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u

    def expo(self, t):
        return float(self._expo(t))

    def sat(self, t):
        return float(self._sat(t))

    def overlay_ramp(self, t):
        tl, T0 = self.tl, self.tl["form0"]
        r0, r1 = self.plan.ramp
        a = smooth(T0 + r0, T0 + r1, t) * (1 - smooth(tl["msg0"] - 4.0, tl["msg0"] - 0.5, t))
        b = smooth(tl["fin0"] + 3.0, tl["fin0"] + 8.0, t)
        return max(a, b)

    def shine_pos(self, t):
        T0, (s0, s1) = self.tl["form0"], self.plan.shine
        return -0.2 + 1.4 * (t - T0 - s0) / (s1 - s0) if T0 + s0 <= t <= T0 + s1 else -5.0

    def glow_fade(self, t):
        tl, T0, D = self.tl, self.tl["form0"], self.tl["form_len"]
        return smooth(T0 + 0.66 * D, T0 + 0.88 * D, t) * (0.55 + 0.45 * smooth(tl["fin0"] + 2, tl["fin0"] + 8, t))

    # ------------------------------------------------------------ tiles: the engine's formation + a star state
    def _build_tiles(self, m, tiles):
        N = self.N
        T0 = self.tl["form0"]
        inst = self.plan.inst.astype(np.float32).copy()        # (N, 21): the engine's per-tile flight parameters
        # ordinary tiles start ~1.3 s before the camera reaches the engine's opening pose, so the first photos take flight while the
        # camera is still rushing through stars (no empty beat). Showcase photos keep their camera-choreographed timing.
        hero = inst[:, 17] >= 1.0
        inst[:, 8] += np.where(hero, T0, T0 - self.tl.get("early_start", 1.3))
        extra = np.zeros((N, 3), np.float32)
        extra[:, 0] = self.rng.uniform(0, 100, N)              # star phase
        extra[:, 1:] = -1.0                                    # hide window (used while a photo is lifted out)
        self.inst = np.concatenate([inst, extra], 1)
        self.cell_tile = np.array([c.tile for c in m.cells])
        self.cell_flip = np.array([c.flip for c in m.cells])

    # ------------------------------------------------------------ name constellation
    def _build_name(self, name, font_path):
        tl = self.tl
        fnt = ImageFont.truetype(font_path, 420)
        bb = fnt.getbbox(name)
        Wm, Hm = bb[2] - bb[0] + 80, bb[3] - bb[1] + 80
        img = Image.new("L", (Wm, Hm), 0)
        ImageDraw.Draw(img).text((40 - bb[0], 40 - bb[1]), name, font=fnt, fill=255)
        mask = np.array(img)
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        width_units = tl.get("name_width", 28.0)
        sc = width_units / (bb[2] - bb[0])
        zn, cy = tl["name_z"], tl.get("name_y", 6.5)
        rng = self.rng
        lines, nodes = [], []
        t_in, dd = tl["name_in"], tl["name_draw"]
        for c in cnts:
            if cv2.contourArea(c) < 300:
                continue
            ap = cv2.approxPolyDP(c, 2.6, True)[:, 0, :].astype(np.float64)
            pts = np.stack([(ap[:, 0] - Wm / 2) * sc, -(ap[:, 1] - Hm / 2) * sc + cy, zn + rng.normal(0, 0.5, len(ap))], 1)
            n = len(pts)
            for i in range(n):
                a, b = pts[i], pts[(i + 1) % n]
                xm = ((ap[i, 0] + ap[(i + 1) % n, 0]) / 2) / Wm
                ta = t_in + xm * dd + rng.uniform(0, 0.25)
                lines.append([*a, ta, *b, 0.0])
                nodes.append([*a, 11.0 + rng.uniform(0, 6.0), rng.uniform(0, 6.28), rng.uniform(0.8, 1.0), ta - 0.1, 0.0])
        ys, xs = np.nonzero(mask)
        pick = rng.choice(len(xs), 520, replace=False)
        for k in pick:
            x, y = xs[k], ys[k]
            ta = t_in + (x / Wm) * dd + rng.uniform(0.2, 1.8)
            nodes.append([(x - Wm / 2) * sc, -(y - Hm / 2) * sc + cy, zn + rng.normal(0, 0.8), 2.6 + rng.uniform(0, 2.6),
                          rng.uniform(0, 6.28), rng.uniform(0.35, 0.8), ta, 0.0])
        self.lines = np.array(lines, np.float32)
        self.name_nodes = np.array(nodes, np.float32)

    # ------------------------------------------------------------ far stars + dust
    def _build_far(self):
        rng = self.rng
        n = 1500
        z = rng.uniform(-520, 300, n)
        dref = 330 - z
        x = rng.uniform(-1, 1, n) * (0.22 * dref + 12)
        y = rng.uniform(-1, 1, n) * (0.36 * dref + 12)
        px = rng.uniform(1.3, 3.2, n) * (rng.random(n) ** 3 * 1.6 + 1)
        a = np.stack([x, y, z, px], 1)
        b = np.stack([rng.uniform(0, 6.28, n), rng.uniform(0.25, 0.95, n), rng.uniform(0, 3, n), np.zeros(n)], 1)
        # approach stars: laid out along the camera's real flight path so the fly-through is a rush of stars, never empty sky
        th, T0 = self.tl["name_hold_end"], self.tl["form0"]
        na = 3600
        pa, pb = [], []
        for t_ in rng.uniform(th - 2, T0 + 5, na):
            eye, target, roll = self.camera(float(t_))
            f, r, u = self.basis(eye, target, roll)
            d = 16 + 230 * rng.random() ** 1.6
            x_, y_ = rng.uniform(-0.95, 0.95, 2)
            pos = eye + f * d + r * x_ * d * TAN * self.aspect + u * y_ * d * TAN
            # bigger and brighter than the sky dust, with a long tail of large stars (they are what a fly-through is made of)
            pa.append([*pos, rng.uniform(3.0, 6.0) * (rng.random() ** 2.2 * 2.6 + 1)])
            pb.append([rng.uniform(0, 6.28), rng.uniform(0.55, 1.0), 0.0, 0.0])
        a = np.concatenate([a, np.array(pa)], 0)
        b = np.concatenate([b, np.array(pb)], 0)
        self.far = np.concatenate([a, b], 1).astype(np.float32)
        nb = 26
        z = rng.uniform(40, 300, nb); dref = 330 - z
        a = np.stack([rng.uniform(-1, 1, nb) * (0.2 * dref + 10), rng.uniform(-1, 1, nb) * (0.34 * dref + 10), z,
                      rng.uniform(40, 110, nb)], 1)
        b = np.stack([rng.uniform(0, 6.28, nb), rng.uniform(0.06, 0.16, nb), np.full(nb, 2.0), np.zeros(nb)], 1)
        self.dust = np.concatenate([a, b], 1).astype(np.float32)

    # ------------------------------------------------------------ hero photos
    def _score_tiles(self, tiles, exclude):
        """Auto selection: sharp, colourful, well-exposed photos with people (skin tone) in the middle of the frame."""
        used = {}
        for i, (t, fl, sz, hv) in enumerate(zip(self.cell_tile, self.cell_flip, self.size, self.inst[:, 17] >= 1.0)):
            if sz >= 1 and not fl and not hv:
                used.setdefault(int(t), []).append(i)
        scored = []
        for ti, cells in used.items():
            if ti in exclude:
                continue
            g = cv2.cvtColor(tiles[ti].image, cv2.COLOR_RGB2GRAY)
            sharp = cv2.Laplacian(g, cv2.CV_64F).var()
            ycc = cv2.cvtColor(tiles[ti].image, cv2.COLOR_RGB2YCrCb)
            h, w = g.shape
            ctr = ycc[h // 5: 4 * h // 5, w // 5: 4 * w // 5]
            skin = ((ctr[..., 1] > 135) & (ctr[..., 1] < 175) & (ctr[..., 2] > 85) & (ctr[..., 2] < 130)).mean()
            col = tiles[ti].image.reshape(-1, 3).std()
            ok = 0.6 if (g.mean() < 55 or g.mean() > 210) else 1.0
            scored.append((ok * (min(sharp, 400) * (1 + 3.0 * min(skin, 0.5)) + col), ti))
        scored.sort(reverse=True)
        return [ti for _, ti in scored], used

    @staticmethod
    def _fingerprint(img):
        """Tiny colour thumbnail, zero-mean and unit-norm: two shots of the same scene/outfit correlate highly."""
        v = cv2.resize(img, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        return v / (np.linalg.norm(v) + 1e-6)

    def _diversify(self, ids, tiles, refs, need, thr=0.72):
        """Walk the quality-ranked list and keep a photo only if it is visibly different from every photo already
        used as a hero (showcase / user photos / earlier picks), so no near-duplicate shot shows up twice."""
        fps = [self._fingerprint(r) for r in refs]
        keep = []
        for ti in ids:
            fp = self._fingerprint(tiles[ti].image)
            if all(float(fp @ q) < thr for q in fps):
                keep.append(ti)
                fps.append(fp)
                if len(keep) >= need:
                    break
        if len(keep) < need:                                  # pool too small for the strict test: top up from the rest
            keep += [ti for ti in ids if ti not in keep][:need - len(keep)]
        print(f"hero diversity: kept {len(keep)} of {len(ids)} ranked photos (threshold {thr})", flush=True)
        return keep

    def _build_heroes(self, m, tiles, s, cache_dir, hero_dir):
        tl, inst = self.tl, self.inst
        px = 512
        s_hi = copy.copy(s); s_hi.cache_px = px
        # engine showcase photos (they fly in large during the formation): marker value - 1 = hi-res layer
        hmask = inst[:, 17] >= 1.0
        order_h = np.argsort(inst[hmask, 17])
        eng_ids = inst[hmask, 3][order_h].astype(int)
        if len(eng_ids):
            hi = load_tiles([m.tile_paths[i] for i in eng_ids], s_hi, cache_dir)
            eng_layers = [cv2.resize(t.image, (px, px), interpolation=cv2.INTER_AREA) for t in hi]
        else:
            eng_layers = []
        n_eh = len(eng_layers)
        user, user_names = load_user_photos(hero_dir, px)
        n_lift = len(tl["lifts"])
        # message photos first (the memories next to the text), the rest of the user's photos replace showcase photos
        lift_user, rest_user = user[:n_lift], user[n_lift:]
        for k, img in enumerate(rest_user[:n_eh]):
            eng_layers[k] = img
        auto_ids, used = self._score_tiles(tiles, set(int(i) for i in eng_ids))
        n_auto = max(0, n_lift - len(lift_user))
        auto_ids = self._diversify(auto_ids, tiles, list(eng_layers) + list(lift_user), n_auto)
        if n_lift:
            print(f"hero photos: {len(user)} from folder ({user_names[:n_lift]}) -> {len(lift_user)} message photos, "
                  f"{min(len(rest_user), n_eh)} showcase; {n_auto} message photos chosen automatically", flush=True)
        lift_layers, lift_cells, lift_off = [], [], []
        auto_i = 0
        taken = set()
        centre_cells = sorted([i for i in range(self.N) if self.size[i] >= 1 and not self.cell_flip[i] and inst[i, 17] < 1.0],
                              key=lambda i: abs(self.wx[i]) + abs(self.wy[i]) * 0.6)
        for j in range(n_lift):
            if j < len(lift_user):
                lift_layers.append(lift_user[j])
                ci = next(c for c in centre_cells[j * 7:] if c not in taken)       # a spread of cells near the middle
            else:
                ti = auto_ids[auto_i]; auto_i += 1
                hi = load_tiles([m.tile_paths[ti]], s_hi, cache_dir)
                lift_layers.append(cv2.resize(hi[0].image, (px, px), interpolation=cv2.INTER_AREA))
                cells = used[ti]
                ci = int(cells[int(np.argmin([abs(self.wx[c]) + abs(self.wy[c]) * 0.6 for c in cells]))])
            taken.add(ci)
            lift_cells.append(ci)
        self.hero_layers = np.stack((eng_layers + lift_layers) if (eng_layers or lift_layers) else [np.zeros((px, px, 3), np.uint8)])
        self.heroes = []
        for j, (ta, tb, nx, ny, nsize) in enumerate(tl["lifts"]):
            ci = lift_cells[j]
            self.inst[ci, 22], self.inst[ci, 23] = ta, tb
            self.heroes.append(dict(kind="lift", layer=n_eh + j, t0=ta, t1=tb, cell=ci, ndc=(nx, ny), nsize=nsize,
                                    off=self.inst[ci, 12:15].copy(), side=1 if j % 2 == 0 else -1))

    # ------------------------------------------------------------ GL objects
    def _gl_setup(self, m, s):
        ctx, W, H = self.ctx, self.W, self.H
        self.fbo_ms = ctx.framebuffer([ctx.renderbuffer((W, H), 4, samples=4, dtype="f2")], ctx.depth_renderbuffer((W, H), samples=4))
        self.tex = ctx.texture((W, H), 4, dtype="f2")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = False
        self.fbo = ctx.framebuffer([self.tex])
        tex_px = min(256, s.cache_px)
        layers = np.stack([cv2.resize(t.image, (tex_px, tex_px), interpolation=cv2.INTER_AREA) for t in self.tiles])
        rgba = np.concatenate([layers, np.full((*layers.shape[:3], 1), 255, np.uint8)], axis=3)
        self.tile_tex = ctx.texture_array((tex_px, tex_px, len(self.tiles)), 4, np.ascontiguousarray(rgba).tobytes())
        self.tile_tex.build_mipmaps()
        self.tile_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.tile_tex.anisotropy = 8.0
        self.tile_tex.repeat_x = self.tile_tex.repeat_y = False
        hl = self.hero_layers
        hr = np.concatenate([hl, np.full((*hl.shape[:3], 1), 255, np.uint8)], axis=3)
        self.hero_tex = ctx.texture_array((hl.shape[2], hl.shape[1], len(hl)), 4, np.ascontiguousarray(hr).tobytes())
        self.hero_tex.build_mipmaps()
        self.hero_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.hero_tex.anisotropy = 8.0
        self.hero_tex.repeat_x = self.hero_tex.repeat_y = False
        mw = 2048
        mh = int(round(mw * m.rows / m.cols))
        main = cv2.resize(m.main, (mw, mh), interpolation=cv2.INTER_AREA)
        alpha = (np.clip(blend_alpha(m, s, mw, mh), 0, 1) * 255 + 0.5).astype(np.uint8)
        self.main_tex = ctx.texture((mw, mh), 4, np.ascontiguousarray(np.concatenate([main, alpha], 2)).tobytes())
        self.main_tex.build_mipmaps()
        self.main_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        P = ctx.program
        self.p_tile = P(vertex_shader=TILE_VS, fragment_shader=TILE_FS)
        self.p_halo = P(vertex_shader=HALO_VS, fragment_shader=HALO_FS)
        self.p_sprite = P(vertex_shader=SPRITE_VS, fragment_shader=SPRITE_FS)
        self.p_line = P(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        self.p_back = P(vertex_shader=BACK_VS, fragment_shader=BACK_FS)
        self.p_over = P(vertex_shader=OVER_VS, fragment_shader=OVER_FS)
        self.p_glow = P(vertex_shader=GLOW_VS, fragment_shader=GLOW_FS)
        self.p_hero = P(vertex_shader=HERO_VS, fragment_shader=HERO_FS)
        quad = ctx.buffer(np.array([-.5, -.5, .5, -.5, -.5, .5, .5, .5], np.float32).tobytes())
        lquad = ctx.buffer(np.array([0, -.5, 1, -.5, 0, .5, 1, .5], np.float32).tobytes())
        self.inst_buf = ctx.buffer(self.inst.tobytes())
        fmt, names = "4f 4f 4f 3f 3f 3f 3f/i", ("i_end", "i_start", "i_time", "i_off", "i_ext", "i_mid", "i_x")
        self.vao_tile = ctx.vertex_array(self.p_tile, [(quad, "2f", "in_vert"), (self.inst_buf, fmt, *names)], skip_errors=True)
        self.vao_halo = ctx.vertex_array(self.p_halo, [(quad, "2f", "in_vert"), (self.inst_buf, fmt, *names)], skip_errors=True)
        far_buf = ctx.buffer(self.far.tobytes()); dust_buf = ctx.buffer(self.dust.tobytes()); node_buf = ctx.buffer(self.name_nodes.tobytes())
        self.vao_far = ctx.vertex_array(self.p_sprite, [(quad, "2f", "in_vert"), (far_buf, "4f 4f/i", "s_a", "s_b")])
        self.vao_dust = ctx.vertex_array(self.p_sprite, [(quad, "2f", "in_vert"), (dust_buf, "4f 4f/i", "s_a", "s_b")])
        self.vao_nodes = ctx.vertex_array(self.p_sprite, [(quad, "2f", "in_vert"), (node_buf, "4f 4f/i", "s_a", "s_b")])
        line_buf = ctx.buffer(self.lines.tobytes())
        self.vao_line = ctx.vertex_array(self.p_line, [(lquad, "2f", "in_vert"), (line_buf, "4f 4f/i", "l_a", "l_b")])
        self.vao_over = ctx.vertex_array(self.p_over, [(quad, "2f", "in_vert")])
        self.vao_back = ctx.vertex_array(self.p_back, [(quad, "2f", "in_vert")])
        self.vao_glow = ctx.vertex_array(self.p_glow, [(quad, "2f", "in_vert")])
        self.vao_hero = ctx.vertex_array(self.p_hero, [(quad, "2f", "in_vert")])
        self.n_far, self.n_dust, self.n_nodes, self.n_lines = len(self.far), len(self.dust), len(self.name_nodes), len(self.lines)
        self.proj = perspective(FOV_DEG, self.aspect, 0.5, 900.0)
        self.p_tile["u_tiles"].value = 0
        self.p_tile["u_hero"].value = 3
        self.p_halo["u_tiles"].value = 0
        self.p_hero["u_hero"].value = 3
        self.p_over["u_main"].value = 1
        self.p_tile["u_gap"].value = 0.0
        self.p_tile["u_dof"].value = 0.7
        self.p_tile["u_bgc"].value = (0.012, 0.030, 0.050)
        self.p_over["u_size"].value = (float(m.cols), float(m.rows))
        pad = 0.55 * max(m.cols, m.rows)
        self.p_glow["u_quad"].value = (m.cols + pad, m.rows + pad)
        self.p_glow["u_half"].value = (m.cols / 2, m.rows / 2)
        self.p_glow["u_scale"].value = float(m.cols)
        self.p_glow["u_accent"].value = (0.90, 0.62, 0.30)
        self.p_sprite["u_res"].value = (float(W), float(H))
        self.p_line["u_res"].value = (float(W), float(H))
        self.p_back["u_res"].value = (float(W), float(H))

    # ------------------------------------------------------------ per-frame
    def _hero_state(self, h, t, eye, f, r, u, target):
        """Photo lifting out of the mosaic -> (pos, size, rx, ry, alpha) or None"""
        if not (h["t0"] <= t <= h["t1"]):
            return None
        ci = h["cell"]
        d_out, d_back = 2.0, 1.8
        a_, b_ = h["t0"], h["t1"]
        if t < a_ + d_out:
            sgl = ease_out((t - a_) / d_out)
        elif t > b_ - d_back:
            sgl = 1 - ease_io((t - (b_ - d_back)) / d_back)
            sgl = sgl * sgl
        else:
            sgl = 1.0
        dh = 0.5 * float(np.linalg.norm(target - eye))
        visw, vish = 2 * dh * self._tan * self.aspect, 2 * dh * self._tan
        nx, ny = h["ndc"]
        tgt = eye + f * dh + r * (nx * 0.5 * visw) + u * (ny * 0.5 * vish)
        cell = np.array([self.wx[ci], self.wy[ci], 0.0])
        bul = np.sin(math.pi * sgl) * 0.18 * dh
        pos = cell * (1 - sgl) + tgt * sgl + (r * h["side"] + f * -0.4) * bul
        size = self.size[ci] * (1 - sgl) + (h["nsize"] * 0.5 * visw) * sgl
        ry = sgl * (0.30 * h["side"] + 0.07 * math.sin(t * 0.8))
        rx = sgl * (-0.10 + 0.04 * math.sin(t * 0.6 + 1))
        return pos, size, rx, ry, 1.0 - smooth(b_ - 0.5, b_, t)

    def render(self, t):
        ctx, W, H = self.ctx, self.W, self.H
        eye, target, roll = self.camera(t)
        view = look_at(eye, target, roll)
        f, r, u = self.basis(eye, target, roll)
        fov = self.fov_deg(t)
        self._tan = math.tan(math.radians(fov) / 2)
        vpb = (perspective(fov, self.aspect, 0.5, 900.0) @ view).T.astype("f4").tobytes()
        expo, sat = self.expo(t), self.sat(t)
        master = smooth(0.0, 3.2, t)
        self.inst_buf.write(self.inst.tobytes())
        self.fbo_ms.use()
        ctx.viewport = (0, 0, W, H)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        # 1) night sky
        pb = self.p_back
        pb["u_time"].value = t
        pb["u_shift"].value = (float(eye[0]) * 0.004, float(eye[1]) * 0.004)
        pb["u_glow"].value = master * float(np.clip(self.glow_fade(t) * 1.2 + 0.25 * smooth(18, 40, t), 0, 1.2))
        pb["u_gpos"].value = (0.5, 0.5)
        ctx.depth_mask = False
        self.vao_back.render(moderngl.TRIANGLE_STRIP)
        ctx.depth_mask = True
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        for pr in (self.p_tile, self.p_halo, self.p_sprite, self.p_line, self.p_over, self.p_glow, self.p_hero):
            pr["u_vp"].write(vpb)
        # 2) rim glow behind the mosaic
        gf = self.glow_fade(t) * master
        if gf > 0.001:
            ctx.depth_mask = False
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.p_glow["u_fade"].value = gf; self.p_glow["u_mode"].value = 0.0
            self.vao_glow.render(moderngl.TRIANGLE_STRIP)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.p_glow["u_mode"].value = 1.0
            self.vao_glow.render(moderngl.TRIANGLE_STRIP)
            ctx.depth_mask = True
        # 3) tiles
        pt = self.p_tile
        pt["u_time"].value = t
        pt["u_eye"].value = tuple(float(v) for v in eye); pt["u_fwd"].value = tuple(float(v) for v in f)
        pt["u_focus"].value = float(np.linalg.norm(target - eye))
        pt["u_expo"].value = expo * master; pt["u_sat"].value = sat * self.g_sat
        pt["u_tint"].value = self.g_tint; pt["u_contrast"].value = self.g_contrast; pt["u_yellow"].value = self.g_yellow
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.tile_tex.use(0)
        self.hero_tex.use(3)
        self.vao_tile.render(moderngl.TRIANGLE_STRIP, instances=self.N)
        # 4) additive light: photo-star halos, far stars, dust, name constellation
        ctx.depth_mask = False
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        ph = self.p_halo
        ph["u_time"].value = t; ph["u_stars"].value = 0.95 * master
        ph["u_right"].value = tuple(float(v) for v in r); ph["u_up"].value = tuple(float(v) for v in u)
        ph["u_eye"].value = tuple(float(v) for v in eye); ph["u_fwd"].value = tuple(float(v) for v in f)
        self.vao_halo.render(moderngl.TRIANGLE_STRIP, instances=self.N)
        ps = self.p_sprite
        ps["u_time"].value = t; ps["u_cam"].value = tuple(float(v) for v in eye)
        ps["u_mode"].value = 0.0; ps["u_col"].value = (0.80, 0.88, 1.0)
        ps["u_gain"].value = 0.95 * master * (1.0 - 0.35 * smooth(self.tl["msg0"], self.tl["msg0"] + 3, t) * (1 - smooth(self.tl["fin0"], self.tl["fin0"] + 4, t)))
        self.vao_far.render(moderngl.TRIANGLE_STRIP, instances=self.n_far)
        ps["u_mode"].value = 1.0; ps["u_col"].value = (0.95, 0.72, 0.42); ps["u_gain"].value = master * 0.9
        self.vao_dust.render(moderngl.TRIANGLE_STRIP, instances=self.n_dust)
        ps["u_mode"].value = 0.0; ps["u_col"].value = (1.0, 0.86, 0.60); ps["u_gain"].value = 1.35
        self.vao_nodes.render(moderngl.TRIANGLE_STRIP, instances=self.n_nodes)
        pl = self.p_line
        pl["u_time"].value = t; pl["u_width"].value = 3.2; pl["u_cam"].value = tuple(float(v) for v in eye)
        pl["u_col"].value = (1.0, 0.82, 0.55); pl["u_gain"].value = 1.4
        self.vao_line.render(moderngl.TRIANGLE_STRIP, instances=self.n_lines)
        # 5) portrait overlay + light sweep
        ramp, shine = self.overlay_ramp(t), self.shine_pos(t)
        ctx.disable(moderngl.DEPTH_TEST)
        if ramp > 0.002 or shine > -1:
            self.p_over["u_ramp"].value = ramp; self.p_over["u_expo"].value = expo * master; self.p_over["u_sat"].value = sat * self.g_sat
            self.p_over["u_tint"].value = self.g_tint; self.p_over["u_contrast"].value = self.g_contrast
            self.p_over["u_yellow"].value = self.g_yellow; self.p_over["u_shine"].value = shine
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.main_tex.use(1)
            self.vao_over.render(moderngl.TRIANGLE_STRIP)
        # 6) photos lifted out of the mosaic (framed prints beside the message)
        self.hero_tex.use(3)
        ph_ = self.p_hero
        for h in self.heroes:
            st = self._hero_state(h, t, eye, f, r, u, target)
            if st is None:
                continue
            pos, size, rx, ry, a = st
            ph_["u_pos"].value = tuple(float(v) for v in pos)
            ph_["u_r"].value = tuple(float(v) for v in r); ph_["u_u"].value = tuple(float(v) for v in u); ph_["u_f"].value = tuple(float(v) for v in f)
            ph_["u_size"].value = float(size); ph_["u_rx"].value = float(rx); ph_["u_ry"].value = float(ry)
            ph_["u_alpha"].value = float(a * master); ph_["u_layer"].value = float(h["layer"])
            ph_["u_off"].value = tuple(float(v) for v in h["off"])
            ph_["u_border"].value = 0.02
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ph_["u_mode"].value = 0.0; ph_["u_scale"].value = 1.25
            self.vao_hero.render(moderngl.TRIANGLE_STRIP)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            ph_["u_mode"].value = 1.0; ph_["u_scale"].value = 2.2
            self.vao_hero.render(moderngl.TRIANGLE_STRIP)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ph_["u_mode"].value = 2.0; ph_["u_scale"].value = 1.0
            self.vao_hero.render(moderngl.TRIANGLE_STRIP)
        ctx.depth_mask = True
        ctx.copy_framebuffer(self.fbo, self.fbo_ms)
        return self.tex
