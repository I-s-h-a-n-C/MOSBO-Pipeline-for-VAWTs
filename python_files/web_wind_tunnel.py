"""
web_wind_tunnel.py  (v4 - retro panel UI, true Gorlov helix)
=============================================================
MOSBO 2.0 - Interactive Web Wind Tunnel (browser companion to the
terminal `virtual_wind_tunnel`, which remains unchanged).

v4 changes:
  * TRUE GORLOV TWIST: the twist slider is the total helix sweep angle;
    each blade ORBITS around the rotor cylinder with height (azimuth
    advances 0 -> twist from bottom to top, all blades together), instead
    of pitching in place.
  * Sliders are now plain native <input type=range> elements with fully
    custom grey/black CSS (the Dash rc-slider widget that rendered
    purple/blue is no longer used anywhere).
  * Retro engineering-panel look (like the reference screenshot): grey
    panel background, black section headers, white value boxes with black
    borders, label -> box -> slider rows, results table on the right.
  * Only Cp (blue) and TSR (red) numbers carry color.
  * SPIN stays an on/off toggle; rotation speed follows TSR.

Extra dependency:  pip install dash   (dash >= 2.9)

Integration in main_engine_pipeline_update.py (unchanged):
    import web_wind_tunnel
    ...
    cp_builder, trf_builder = surrogates.build_surrogates(df_cp, df_trf)
    web_wind_tunnel.run_web_tunnel(cp_builder, trf_builder, df_cp,
                                   background=True)
    ...
    virtual_wind_tunnel(cp_builder, trf_builder)   # terminal version

Run directly (`python web_wind_tunnel.py`) for a synthetic-data demo.
"""

import threading
import webbrowser

import numpy as np

try:
    from dash import Dash, dcc, html, Input, Output, State, Patch
    import plotly.graph_objects as go
    _DASH_AVAILABLE = True
except ImportError:
    _DASH_AVAILABLE = False

try:
    import config as _cfg
except Exception:
    _cfg = None

LAMBDA_PENALTY   = float(getattr(_cfg, 'LAMBDA_PENALTY', 0.25))
TRF_PARETO_LIMIT = float(getattr(_cfg, 'TRF_PARETO_LIMIT', 1.5))
CP_EXPORT_GATE   = 0.35

# ---------------------- schematic rotor constants ----------------------
N_BLADES          = 3
ROTOR_RADIUS      = 1.0
ROTOR_HEIGHT      = 2.2
CHORD_PREVIEW_SCALE = 3.0   # preview-only chord magnification (disclosed)
SPIN_INTERVAL_MS  = 100
SPIN_RATE         = 0.5     # rad/s per unit TSR (spin speed follows TSR)

COLOR_BLADE = '#c9c9c9'     # QBlade-style light grey
COLOR_WIND  = '#a8a8a8'
COLOR_CP    = '#1d4ed8'     # allowed color: Cp number
COLOR_TSR   = '#c2410c'     # allowed color: TSR number
PANEL_BG    = '#d4d0c8'     # retro engineering panel grey
VIEW_BG     = '#f7f8fa'

__all__ = ['run_web_tunnel']


# =====================================================================
#  Geometry helpers
# =====================================================================
def _naca00_outline(thickness_pct, n_half=24):
    """Closed NACA 00xx outline in chord units, quarter-chord at origin."""
    t = max(0.02, thickness_pct) / 100.0
    beta = np.linspace(0.0, np.pi, n_half)
    x = (1.0 - np.cos(beta)) / 2.0
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)
    x_up, y_up = x[::-1], yt[::-1]
    x_lo, y_lo = x[1:-1], -yt[1:-1]
    xc = np.concatenate([x_up, x_lo]) - 0.25
    yc = np.concatenate([y_up, y_lo])
    return np.column_stack([xc, yc])


def _tube_triangulation(n_sections, n_pts):
    ii, jj, kk = [], [], []
    for s in range(n_sections - 1):
        a0, a1 = s * n_pts, (s + 1) * n_pts
        for p in range(n_pts):
            q = (p + 1) % n_pts
            ii += [a0 + p, a0 + q]
            jj += [a1 + p, a1 + p]
            kk += [a0 + q, a1 + q]
    return np.array(ii), np.array(jj), np.array(kk)


def _blade_mesh(profile, chord, radius, height, total_twist_deg,
                azimuth_deg, n_span=16):
    """
    Gorlov helical blade: the section ORBITS around the rotor cylinder.
    Azimuth advances from `azimuth_deg` at the bottom to
    `azimuth_deg + total_twist_deg` at the top (all blades sweep
    together). The airfoil stays tangential to the cylinder surface.
    """
    n = profile.shape[0]
    sections = []
    for z in np.linspace(0.0, height, n_span):
        frac = z / height
        az = np.deg2rad(azimuth_deg + total_twist_deg * frac)   # helix orbit
        caz, saz = np.cos(az), np.sin(az)
        u = profile[:, 1] * chord      # radial offset (thickness dir.)
        v = profile[:, 0] * chord      # tangential offset (chord dir.)
        r = radius + u
        sections.append(np.column_stack([r * caz - v * saz,
                                         r * saz + v * caz,
                                         np.full(n, z)]))
    sections = np.array(sections)
    verts = sections.reshape(-1, 3)
    ii, jj, kk = [list(a) for a in _tube_triangulation(n_span, n)]
    for sec, flip in ((0, True), (n_span - 1, False)):
        ci = len(verts)
        verts = np.vstack([verts, sections[sec].mean(axis=0)[None, :]])
        base = sec * n
        for p in range(n):
            q = (p + 1) % n
            if flip:
                ii.append(ci); jj.append(base + q); kk.append(base + p)
            else:
                ii.append(ci); jj.append(base + p); kk.append(base + q)
    return verts, (np.array(ii), np.array(jj), np.array(kk))


def _merge_meshes(meshes):
    verts_all, ii, jj, kk = [], [], [], []
    off = 0
    for v, (a, b, c) in meshes:
        verts_all.append(v)
        ii.append(np.asarray(a) + off)
        jj.append(np.asarray(b) + off)
        kk.append(np.asarray(c) + off)
        off += len(v)
    return np.vstack(verts_all), (np.concatenate(ii), np.concatenate(jj),
                                  np.concatenate(kk))


def _rotate_z(verts, angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    out = verts.copy()
    out[:, 0] = verts[:, 0] * c - verts[:, 1] * s
    out[:, 1] = verts[:, 0] * s + verts[:, 1] * c
    return out


def build_rotor_geometry(thickness_pct, twist_deg, solidity):
    """Three Gorlov-helical blades. Morphs with all three geometry sliders."""
    chord = max(0.02, solidity * ROTOR_RADIUS / N_BLADES) * CHORD_PREVIEW_SCALE
    prof = _naca00_outline(thickness_pct)
    blades = []
    for k in range(N_BLADES):
        blades.append(_blade_mesh(prof, chord, ROTOR_RADIUS, ROTOR_HEIGHT,
                                  twist_deg, 360.0 * k / N_BLADES))
    return _merge_meshes(blades)


def _rotor_at_angle(thickness, twist, solidity, angle_rad):
    v, t = build_rotor_geometry(thickness, twist, solidity)
    if angle_rad:
        v = _rotate_z(v, angle_rad)
    return v, t


# =====================================================================
#  Figure assembly (monochrome, QBlade-like)
# =====================================================================
def _mesh3d(verts, tris, color):
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=tris[0], j=tris[1], k=tris[2],
        color=color, flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.9, specular=0.25,
                      roughness=0.55, fresnel=0.1),
        lightposition=dict(x=-80, y=-60, z=140),
        hoverinfo='skip', showlegend=False)


def _wind_arrows():
    xs, ys, zs = [], [], []
    for zi in (0.55, 1.15, 1.75):
        for yi in (-0.85, 0.0, 0.85):
            xs.append(-2.35); ys.append(yi); zs.append(zi)
    return go.Cone(x=xs, y=ys, z=zs,
                   u=[1.4] * len(xs), v=[0.0] * len(xs), w=[0.0] * len(xs),
                   colorscale=[[0, COLOR_WIND], [1, COLOR_WIND]],
                   showscale=False, sizemode='absolute', sizeref=0.4,
                   opacity=0.9, hoverinfo='skip', showlegend=False)


def build_figure(thickness, twist, solidity, tsr, angle_rad=0.0,
                 show_wind=True):
    rotor_v, rotor_t = _rotor_at_angle(thickness, twist, solidity, angle_rad)
    traces = [_mesh3d(rotor_v, rotor_t, COLOR_BLADE)]
    if show_wind:
        traces.append(_wind_arrows())

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-2.9, 2.2]),
            yaxis=dict(visible=False, range=[-2.05, 2.05]),
            zaxis=dict(visible=False, range=[-0.4, ROTOR_HEIGHT + 0.4]),
            aspectmode='manual', aspectratio=dict(x=1.35, y=1.1, z=1.0),
            camera=dict(eye=dict(x=1.8, y=-1.9, z=0.75),
                        up=dict(x=0, y=0, z=1)),
            bgcolor=VIEW_BG),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor=VIEW_BG,
        showlegend=False,
        uirevision='mosbo-wt')
    return fig


# =====================================================================
#  Retro panel UI
# =====================================================================
BOX_IN = {'background': '#ffffff', 'border': '1px solid #000000',
          'padding': '3px 6px', 'fontSize': '11px'}
SEC_HEAD = {'background': '#000000', 'color': '#ffffff', 'fontSize': '11px',
            'fontWeight': 700, 'letterSpacing': '1px', 'padding': '3px 8px',
            'marginBottom': '10px'}
LBL = {'fontSize': '11px', 'textAlign': 'right'}

INDEX_STRING = '''<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
 html, body { background:#d4d0c8; }
 /* ---- native range sliders: fully grey/black, cannot turn purple ---- */
 input[type=range] { -webkit-appearance:none; appearance:none; width:100%;
                     height:4px; background:#b0b0b0; border:none; padding:0;
                     outline:none !important; margin:0; }
 input[type=range]::-webkit-slider-runnable-track { height:4px;
                     background:#b0b0b0; border:none; }
 input[type=range]::-webkit-slider-thumb { -webkit-appearance:none;
                     appearance:none; width:20px; height:10px; margin-top:-3px;
                     background:#808080; border:1px solid #404040;
                     border-radius:5px; cursor:pointer; }
 input[type=range]::-moz-range-track { height:4px; background:#b0b0b0;
                     border:none; }
 input[type=range]::-moz-range-thumb { width:18px; height:9px;
                     background:#808080; border:1px solid #404040;
                     border-radius:5px; cursor:pointer; }
 input[type=range]:focus, input[type=range]:active { outline:none !important; }
 input[type=range]::-webkit-slider-thumb:hover,
 input[type=range]::-webkit-slider-thumb:active { background:#606060; }
 input[type=range]::-moz-range-thumb:hover,
 input[type=range]::-moz-range-thumb:active { background:#606060; }
 input[type=checkbox] { accent-color:#000000; }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''


def _extract_bounds(df_cp):
    defaults = {'thickness': (13.0, 21.0), 'twist': (0.0, 150.0),
                'solidity': (0.15, 0.45), 'tsr': (1.5, 4.5)}
    out = dict(defaults)
    if df_cp is not None:
        for col in out:
            try:
                if col in df_cp.columns and len(df_cp) > 0:
                    lo, hi = float(df_cp[col].min()), float(df_cp[col].max())
                    if hi > lo:
                        out[col] = (lo, hi)
            except Exception:
                pass
    return out


def _param_row(label, sid, ro_id, lo, hi, step, value, ro_color='#000000'):
    return html.Div(style={'display': 'flex', 'alignItems': 'center',
                           'gap': '8px', 'marginBottom': '10px'}, children=[
        html.Div(label, style={**LBL, 'width': '86px'}),
        html.Div(id=ro_id, style={**BOX_IN, 'width': '52px',
                                  'textAlign': 'center', 'fontWeight': 700,
                                  'color': ro_color}),
        dcc.Input(id=sid, type='range', min=float(lo), max=float(hi),
                  step=float(step), value=float(value),
                  style={'flex': '1'}),
    ])


def _result_row(label, rid, color='#000000'):
    return html.Div(style={'display': 'flex', 'alignItems': 'center',
                           'gap': '8px', 'marginBottom': '8px'}, children=[
        html.Div(label, style={**LBL, 'width': '70px'}),
        html.Div(id=rid, style={**BOX_IN, 'flex': '1', 'color': color,
                                'fontWeight': 700}),
    ])


def _build_app(cp_builder, trf_builder, df_cp):
    app = Dash(__name__, title='MOSBO 2.0 - Virtual Wind Tunnel',
               index_string=INDEX_STRING)
    b = _extract_bounds(df_cp)

    th_lo, th_hi = b['thickness']
    th_step = 1.0 if (float(th_lo).is_integer()
                      and float(th_hi).is_integer()) else 0.5
    tw_lo, tw_hi = b['twist']
    tw_step = 1.0 if (tw_hi - tw_lo) > 20 else 0.5

    def mid(lo, hi, step):
        return round(round((lo + hi) / 2.0, 6) / step) * step

    left_col = html.Div(style={'flex': 1, 'minWidth': 0,
                               'marginRight': '14px'}, children=[
        html.Div(style={'border': '2px solid #000000',
                        'background': VIEW_BG}, children=[
            dcc.Graph(id='graph-3d', style={'height': '520px'},
                      config={'displaylogo': False, 'displayModeBar': False,
                              'responsive': True}),
        ]),
        html.Div(style={'display': 'flex', 'alignItems': 'center',
                        'gap': '18px', 'marginTop': '6px'}, children=[
            dcc.Checklist(id='chk-spin',
                          options=[{'label': ' SPIN', 'value': 'on'}],
                          value=[], style={'fontSize': '11px'}),
            dcc.Checklist(id='chk-wind',
                          options=[{'label': ' WIND', 'value': 'on'}],
                          value=['on'], style={'fontSize': '11px'}),
            html.Div('SPIN SPEED FOLLOWS TSR',
                     style={'fontSize': '10px', 'color': '#707070',
                            'marginLeft': 'auto'}),
        ]),
    ])

    right_col = html.Div(style={'width': '380px', 'flexShrink': 0}, children=[
        html.Div('DESIGN', style=SEC_HEAD),
        _param_row('Thickness %', 'sl-thickness', 'ro-thickness',
                   th_lo, th_hi, th_step, mid(th_lo, th_hi, th_step)),
        _param_row('Twist deg', 'sl-twist', 'ro-twist',
                   tw_lo, tw_hi, tw_step, mid(tw_lo, tw_hi, tw_step)),
        _param_row('Solidity', 'sl-solidity', 'ro-solidity',
                   b['solidity'][0], b['solidity'][1], 0.01,
                   mid(*b['solidity'], 0.01)),
        _param_row('TSR', 'sl-tsr', 'ro-tsr',
                   b['tsr'][0], b['tsr'][1], 0.1, mid(*b['tsr'], 0.1),
                   ro_color=COLOR_TSR),
        html.Div('RESULTS', style={**SEC_HEAD, 'marginTop': '14px'}),
        html.Div(style={'display': 'flex', 'gap': '14px'}, children=[
            html.Div(style={'flex': 1}, children=[
                _result_row('Cp', 'ro-cp', COLOR_CP),
                _result_row('sig Cp', 'ro-sc'),
                _result_row('CI Cp', 'ro-ci'),
            ]),
            html.Div(style={'flex': 1}, children=[
                _result_row('TRF', 'ro-trf'),
                _result_row('sig TRF', 'ro-st'),
                _result_row('EXPORT', 'ro-exp'),
            ]),
        ]),
        html.Div(id='ro-status', style={**BOX_IN, 'marginTop': '4px'}),
        html.Div('SCHEMATIC PREVIEW. CHORD x3 FOR VISIBILITY. '
                 'TWIST = TOTAL HELIX SWEEP AROUND THE ROTOR CYLINDER '
                 '(GORLOV).',
                 style={'fontSize': '10px', 'color': '#707070',
                        'marginTop': '8px'}),
    ])

    app.layout = html.Div(style={
        'background': PANEL_BG, 'minHeight': '100vh',
        'padding': '10px 14px 20px',
        'fontFamily': "Tahoma, Verdana, sans-serif"}, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between',
                        'alignItems': 'baseline', 'marginBottom': '10px'},
                 children=[
            html.Span('MOSBO 2.0 - VIRTUAL WIND TUNNEL',
                      style={'fontSize': '14px', 'fontWeight': 700,
                             'color': '#000000'}),
            html.Span('SURROGATE EXPLORER  ·  v4',
                      style={'fontSize': '10px', 'color': '#707070'}),
        ]),
        html.Div(style={'display': 'flex', 'alignItems': 'flex-start'},
                 children=[left_col, right_col]),
        dcc.Store(id='spin-angle', data=0.0),
        dcc.Interval(id='spin-interval', interval=SPIN_INTERVAL_MS,
                     disabled=True),
    ])

    @app.callback(
        Output('graph-3d', 'figure'),
        Output('ro-thickness', 'children'),
        Output('ro-twist', 'children'),
        Output('ro-solidity', 'children'),
        Output('ro-tsr', 'children'),
        Output('ro-cp', 'children'),
        Output('ro-sc', 'children'),
        Output('ro-ci', 'children'),
        Output('ro-trf', 'children'),
        Output('ro-st', 'children'),
        Output('ro-exp', 'children'),
        Output('ro-status', 'children'),
        Input('sl-thickness', 'value'),
        Input('sl-twist', 'value'),
        Input('sl-solidity', 'value'),
        Input('sl-tsr', 'value'),
        Input('chk-wind', 'value'),
        State('spin-angle', 'data'),
    )
    def _update(thickness, twist, solidity, tsr, wind_chk, angle):
        thickness = round(float(thickness), 2)
        twist = round(float(twist), 2)
        solidity = round(float(solidity), 3)
        tsr = round(float(tsr), 2)
        angle = float(angle or 0.0)

        fig = build_figure(thickness, twist, solidity, tsr,
                           angle_rad=angle, show_wind=bool(wind_chk))

        try:
            cp_mean, cp_sig = cp_builder.predict(
                np.array([[thickness, twist, solidity, tsr]]))
            trf_mean, trf_sig = trf_builder.predict(
                np.array([[thickness, twist, solidity]]))
            cp_m, cp_s = float(cp_mean[0]), float(cp_sig[0])
            trf_m, trf_s = float(trf_mean[0]), float(trf_sig[0])
        except Exception as exc:
            err = f'SURROGATE ERROR: {exc}'
            return (fig, f'{thickness:g}', f'{twist:g}', f'{solidity:.2f}',
                    f'{tsr:.1f}', '--', '--', '--', '--', '--', '--', err)

        conf = ('HIGH' if cp_s < 0.015
                else 'MEDIUM' if cp_s < 0.03 else 'LOW-NEEDS-CFD')
        cp_eff = cp_m - LAMBDA_PENALTY * cp_s
        trf_eff = trf_m + LAMBDA_PENALTY * trf_s
        cp_ok = cp_eff >= CP_EXPORT_GATE
        trf_ok = trf_eff <= TRF_PARETO_LIMIT

        status = (f'CP GATE {"PASS" if cp_ok else "FAIL"}  ·  '
                  f'TRF CEIL {"PASS" if trf_ok else "FAIL"}  ·  '
                  f'CONF {conf}  ·  LAMBDA {LAMBDA_PENALTY:g}')

        return (fig, f'{thickness:g}', f'{twist:g}', f'{solidity:.2f}',
                f'{tsr:.1f}',
                f'{cp_m:.4f}', f'{cp_s:.4f}',
                f'[{cp_m - 1.96 * cp_s:.3f}, {cp_m + 1.96 * cp_s:.3f}]',
                f'{trf_m:.4f}', f'{trf_s:.4f}',
                'ALLOWED' if (cp_ok and trf_ok) else 'BLOCKED',
                status)

    @app.callback(
        Output('graph-3d', 'figure', allow_duplicate=True),
        Output('spin-angle', 'data'),
        Input('spin-interval', 'n_intervals'),
        State('spin-angle', 'data'),
        State('sl-thickness', 'value'),
        State('sl-twist', 'value'),
        State('sl-solidity', 'value'),
        State('sl-tsr', 'value'),
        prevent_initial_call=True,
    )
    def _spin(_, angle, thickness, twist, solidity, tsr):
        angle = (float(angle or 0.0)
                 + SPIN_RATE * float(tsr) * (SPIN_INTERVAL_MS / 1000.0))
        angle %= 2.0 * np.pi
        rv, _rt = _rotor_at_angle(float(thickness), float(twist),
                                  float(solidity), angle)
        p = Patch()
        p['data'][0]['x'] = np.round(rv[:, 0], 4).tolist()
        p['data'][0]['y'] = np.round(rv[:, 1], 4).tolist()
        return p, angle

    @app.callback(Output('spin-interval', 'disabled'),
                  Input('chk-spin', 'value'))
    def _toggle_spin(value):
        return not value

    return app


# =====================================================================
#  Public entry point
# =====================================================================
def run_web_tunnel(cp_builder, trf_builder, df_cp=None, host='127.0.0.1',
                   port=8050, background=False, open_browser=True):
    """Launch the browser Virtual Wind Tunnel (terminal version untouched)."""
    if not _DASH_AVAILABLE:
        print("\n[Web Tunnel] Dash not installed. Run:  pip install dash")
        return None

    app = _build_app(cp_builder, trf_builder, df_cp)
    url = f'http://{host}:{port}/'

    def _serve():
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        try:
            runner = getattr(app, 'run', None) or getattr(app, 'run_server')
            runner(host=host, port=port, debug=False, use_reloader=False)
        except OSError as e:
            print(f"[Web Tunnel] Could not start server on {url}: {e}")

    print('\n' + '=' * 60)
    print(' WEB WIND TUNNEL (retro panel, Gorlov helix)')
    print('=' * 60)
    if background:
        threading.Thread(target=_serve, daemon=True).start()
        print(f' -> Serving in background at {url}')
    else:
        print(f' -> Serving at {url}  (Ctrl+C to stop)')

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    if not background:
        try:
            _serve()
        except KeyboardInterrupt:
            print('\n[Web Tunnel] Closed.')
    return app


# =====================================================================
#  Standalone demo:  python web_wind_tunnel.py
# =====================================================================
if __name__ == '__main__':
    import pandas as pd
    print('[Web Tunnel Demo] Building small synthetic surrogates...')
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

    class _DemoSurrogate:
        def __init__(self, X, y):
            self.sx = StandardScaler().fit(X)
            self.sy = StandardScaler().fit(y.reshape(-1, 1))
            gp = GaussianProcessRegressor(kernel=C(1.0) * Matern(nu=2.5),
                                          alpha=1e-3, n_restarts_optimizer=2,
                                          random_state=42)
            gp.fit(self.sx.transform(X),
                   self.sy.transform(y.reshape(-1, 1)).ravel())
            self.model = gp

        def predict(self, X):
            m, s = self.model.predict(self.sx.transform(np.asarray(X)),
                                      return_std=True)
            return (self.sy.inverse_transform(m.reshape(-1, 1)).ravel(),
                    s * self.sy.scale_[0])

    rng = np.random.default_rng(42)
    N = 400
    thickness = rng.choice([13.0, 15.0, 18.0, 21.0], N)
    twist = rng.uniform(0.0, 150.0, N)          # total Gorlov helix angle
    solidity = rng.uniform(0.15, 0.45, N)
    tsr = rng.uniform(1.5, 4.5, N)
    cp = (0.30 + 0.05 * tsr - 0.012 * (tsr - 2.8) ** 2
          + 0.0015 * (thickness - 15.0) + 0.0008 * (twist - 75.0) / 10.0
          - 1.2 * (solidity - 0.30) ** 2 + rng.normal(0.0, 0.004, N))
    trf = np.clip(1.30 - 1.5 * (solidity - 0.15)
                  - 0.004 * (twist - 30.0)
                  + 0.008 * (thickness - 13.0) + 0.06 * np.abs(tsr - 2.5)
                  + rng.normal(0.0, 0.02, N), 0.05, None)

    demo_df = pd.DataFrame({'thickness': thickness, 'twist': twist,
                            'solidity': solidity, 'tsr': tsr, 'cp': cp})
    run_web_tunnel(
        _DemoSurrogate(np.column_stack([thickness, twist, solidity, tsr]), cp),
        _DemoSurrogate(np.column_stack([thickness, twist, solidity]), trf),
        demo_df, background=False)