"""Experiment 9: Integrated 3D cosmic web visualization (no clustering rerun).

Reads saved outputs of Experiments 6, 7 and 8 in the project's data/ folder.
Requires: pandas, numpy, plotly. Run in Spyder.
Outputs: results/Cosmic_Web_Atlas.html and results/Atlas_Summary.csv

Important: PCA classes are geometric candidates, matched names are spatial
associations only, and empty spheres are NOT validated cosmic voids.
"""
from pathlib import Path
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from html import escape

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
ASSIGNMENTS = DATA / 'Full_Survey_Galaxy_Assignments.csv'
STRUCTURES = DATA / 'Full_Survey_Structure_Catalog.csv'
MATCHES = DATA / 'Experiment7_Known_Structure_Matches.csv'
VOIDS = DATA / 'Experiment8_Void_Candidates.csv'
OUTPUT = RESULTS / 'Cosmic_Web_Atlas.html'
SUMMARY = RESULTS / 'Atlas_Summary.csv'
SEED = 42
MAX_CONTEXT_PER_SURVEY = 25000
MAX_STRUCTURE_POINTS_PER_CLASS = 45000
MAX_REFERENCE_GROUP_POINTS = 5000
MAX_SPHERES = 60

MORPH_COLORS = {
    'Filament-like': '#ffac36',
    'Sheet-like': '#37d7c5',
    'Compact/round': '#ff65b5',
    'Complex/elongated': '#aa83ff',
}

# These are approximate literature reference anchors used in Experiment 7.
# They are NOT proof of physical membership or structure boundaries.
REFERENCES = [
    ('Coma Cluster (Abell 1656)', 194.9350, 27.9125, 0.0234),
    ('Leo Cluster (Abell 1367)', 176.0, 20.0, 0.0220),
    ('Hercules Cluster (Abell 2151)', 241.0, 18.0, 0.0354),
    ('Shapley region: Abell 3558', 202.0133, -31.5264, 0.0474),
]


def say(message):
    print(message, flush=True)


def require(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing input file: {path}\nPlace the four saved Experiment 6/7/8 CSV outputs in the data/ folder.')


def sample(frame, limit):
    return frame if len(frame) <= limit else frame.sample(n=limit, random_state=SEED)


def reference_xyz(ra_deg, dec_deg, redshift):
    radius = 299792.458 * redshift / 70.0
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return (radius*np.cos(dec)*np.cos(ra), radius*np.cos(dec)*np.sin(ra), radius*np.sin(dec))


def sphere_xyz(cx, cy, cz, r):
    theta = np.linspace(0, 2*np.pi, 16)
    phi = np.linspace(0, np.pi, 10)
    x = cx + r*np.outer(np.sin(phi), np.cos(theta))
    y = cy + r*np.outer(np.sin(phi), np.sin(theta))
    z = cz + r*np.outer(np.cos(phi), np.ones_like(theta))
    return x, y, z


def main():
    say('STEP 1/6: Checking Experiment 6, 7 and 8 CSV files...')
    for p in (ASSIGNMENTS, STRUCTURES, MATCHES, VOIDS):
        require(p)
    say('STEP 2/6: Loading saved results (DBSCAN will NOT run again)...')
    galaxies = pd.read_csv(ASSIGNMENTS, low_memory=False)
    structures = pd.read_csv(STRUCTURES, low_memory=False)
    matches = pd.read_csv(MATCHES, low_memory=False)
    voids = pd.read_csv(VOIDS, low_memory=False)
    for name, frame, columns in [
        ('assignments', galaxies, ['Survey','X_Mpc','Y_Mpc','Z_Mpc','Global_Group']),
        ('structure catalog', structures, ['Group_ID','Morphology_Candidate','Galaxy_Count']),
        ('void catalog', voids, ['Candidate_ID','Survey','X_Mpc','Y_Mpc','Z_Mpc','Empty_Sphere_Radius_Mpc']),
    ]:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f'{name}: missing {missing}. Available: {list(frame.columns)}')
    say(f'  {len(galaxies):,} galaxies; {len(structures):,} characterized groups; '
        f'{len(matches):,} proximity-match rows; {len(voids):,} candidate empty spheres.')

    say('STEP 3/6: Creating background and PCA morphology layers...')
    fig = go.Figure()
    kinds = []
    def add(trace, kind):
        fig.add_trace(trace)
        kinds.append(kind)

    for survey, color in [('SDSS','#76b8ff'),('6DFGS','#ff94c5')]:
        subset = sample(galaxies[galaxies.Survey.astype(str).str.upper() == survey], MAX_CONTEXT_PER_SURVEY)
        if len(subset):
            add(go.Scatter3d(x=subset.X_Mpc,y=subset.Y_Mpc,z=subset.Z_Mpc,
                mode='markers',name=f'{survey} galaxy context (sample)',
                marker=dict(size=1.2,color=color,opacity=.22),hoverinfo='skip',
                visible='legendonly'), 'context')

    lookup = structures.drop_duplicates('Group_ID').set_index('Group_ID')
    assigned = galaxies[galaxies.Global_Group.isin(lookup.index)].copy()
    assigned['Morphology_Candidate'] = assigned.Global_Group.map(lookup.Morphology_Candidate)
    for morphology, color in MORPH_COLORS.items():
        sub = sample(assigned[assigned.Morphology_Candidate == morphology], MAX_STRUCTURE_POINTS_PER_CLASS)
        if not len(sub):
            continue
        custom = np.stack([sub.Global_Group.astype(str).to_numpy()],axis=1)
        add(go.Scatter3d(x=sub.X_Mpc,y=sub.Y_Mpc,z=sub.Z_Mpc,mode='markers',
            name=f'{morphology} candidates',marker=dict(size=2.1,color=color,opacity=.85),
            customdata=custom,
            hovertemplate='DBSCAN group: %{customdata[0]}<br>X: %{x:.1f} Mpc<br>Y: %{y:.1f} Mpc<br>Z: %{z:.1f} Mpc<extra></extra>'), 'structures')

    say('STEP 4/6: Adding published cluster references and candidate matches...')
    ref_xyz = np.array([reference_xyz(ra,dec,z) for _,ra,dec,z in REFERENCES])
    add(go.Scatter3d(x=ref_xyz[:,0],y=ref_xyz[:,1],z=ref_xyz[:,2],
        mode='markers',name='Published cluster reference positions',
        marker=dict(size=7,color='#ffffff',symbol='diamond',line=dict(width=1,color='#222222')),
        hovertext=[f'{name}<br>Reference RA={ra:.2f}°, Dec={dec:.2f}°, z={z:.4f}<br>Approximate position; not a confirmed DBSCAN match'
                   for name,ra,dec,z in REFERENCES],hoverinfo='text'), 'references')

    # Optional prominent group centers for groups found near published references.
    group_column = next((c for c in ('Candidate_Group_ID','Group_ID','Candidate_Group','DBSCAN_Group','Global_Group') if c in matches.columns), None)
    if group_column is None:
        say('  Match CSV columns: ' + ', '.join(matches.columns))
        say('  Could not identify group ID field; showing published reference markers only.')
    else:
        matched = structures[structures.Group_ID.isin(matches[group_column].dropna().astype(str))].copy()
        coords = ['Center_X_Mpc','Center_Y_Mpc','Center_Z_Mpc']
        if len(matched) and all(c in matched for c in coords):
            add(go.Scatter3d(x=matched[coords[0]],y=matched[coords[1]],z=matched[coords[2]],
                mode='markers',name='Candidate DBSCAN group centers near references',
                marker=dict(size=5,color='#fdf25a',symbol='diamond'),
                text=[f'{r.Group_ID}: {r.Galaxy_Count:,} galaxies' for r in matched.itertuples()],
                hoverinfo='text'), 'references')

    say('STEP 5/6: Adding exploratory empty-sphere candidates...')
    for row in voids.head(MAX_SPHERES).itertuples(index=False):
        x,y,z = sphere_xyz(float(row.X_Mpc),float(row.Y_Mpc),float(row.Z_Mpc),
                           float(row.Empty_Sphere_Radius_Mpc))
        color = '#ffbd59' if str(row.Survey).upper() == 'SDSS' else '#4de3d0'
        add(go.Surface(x=x,y=y,z=z,showscale=False,opacity=.17,
            colorscale=[[0,color],[1,color]],surfacecolor=np.zeros_like(x),
            name=f'Candidate {row.Candidate_ID}',showlegend=False,
            hovertemplate=f'{row.Candidate_ID} ({row.Survey})<br>Empty sphere radius: {row.Empty_Sphere_Radius_Mpc:.1f} Mpc<br>NOT a validated cosmic void<extra></extra>',
            visible='legendonly'), 'voids')
    if len(voids):
        add(go.Scatter3d(x=voids.X_Mpc,y=voids.Y_Mpc,z=voids.Z_Mpc,
            mode='markers',name=f'Empty-sphere centers ({len(voids)} candidates)',
            marker=dict(size=4,color=['#ffbd59' if str(s).upper()=='SDSS' else '#4de3d0' for s in voids.Survey],symbol='diamond'),
            text=[f'{r.Candidate_ID} | {r.Survey} | radius {r.Empty_Sphere_Radius_Mpc:.1f} Mpc | NOT validated'
                  for r in voids.itertuples()],hoverinfo='text',visible='legendonly'), 'voids')

    # Full-width presentation layout. Buttons are below the heading and never overlap it.
    # Clicking the legend can further isolate individual structure categories.
    def mask(*enabled):
        return [kind in enabled for kind in kinds]

    button_sets = [
        ('Structures', ('structures', 'references')),
        ('Galaxies + structures', ('context', 'structures', 'references')),
        ('Candidate voids', ('voids', 'references')),
        ('All layers', ('context', 'structures', 'references', 'voids')),
    ]
    fig.update_layout(
        title=dict(
            text='<b>THE OBSERVED COSMIC WEB</b><br>'
                 '<span style="font-size:13px;color:#aebed2">'
                 f'{len(galaxies):,} galaxies  ·  DBSCAN density groups  ·  PCA morphology  ·  candidate underdensities'
                 '</span>',
            x=0.5, xanchor='center', y=0.925, yanchor='top',
            font=dict(size=23, color='#f5f8ff')
        ),
        template='plotly_dark',
        paper_bgcolor='#080e1b',
        plot_bgcolor='#080e1b',
        font=dict(color='#edf3ff', family='Arial'),
        height=760,
        autosize=True,
        margin=dict(l=8, r=8, t=175, b=65),
        legend=dict(
            x=0.015, y=0.97, xanchor='left', yanchor='top',
            bgcolor='rgba(12,22,38,0.83)',
            bordercolor='rgba(125,160,195,0.25)', borderwidth=1,
            font=dict(size=11), itemsizing='constant'
        ),
        scene=dict(
            xaxis=dict(title='X (Mpc)', backgroundcolor='#0c1829',
                       gridcolor='#34445d', zerolinecolor='#53647c'),
            yaxis=dict(title='Y (Mpc)', backgroundcolor='#0c1829',
                       gridcolor='#34445d', zerolinecolor='#53647c'),
            zaxis=dict(title='Z (Mpc)', backgroundcolor='#0c1829',
                       gridcolor='#34445d', zerolinecolor='#53647c'),
            aspectmode='data', bgcolor='#080e1b',
            camera=dict(eye=dict(x=1.6, y=1.55, z=0.95)),
            domain=dict(x=[0.07, 0.99], y=[0.02, 0.99])
        ),
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.5, xanchor='center', y=1.055, yanchor='bottom',
            showactive=False, active=0, pad=dict(t=5, b=5, l=4, r=4),
            bgcolor='#263a57', bordercolor='#4c6484',
            font=dict(color='#ffffff', size=12),
            buttons=[
                dict(label=label, method='update',
                     args=[{'visible': mask(*categories)}])
                for label, categories in button_sets
            ]
        )],
        annotations=[dict(
            x=0.5, y=-0.055, xref='paper', yref='paper',
            showarrow=False, xanchor='center',
            text='Exploratory identifications: groups, morphologies and empty spheres are not confirmed physical structures. '
                 'Click legend items to isolate layers; hover for details.',
            font=dict(size=11, color='#aebed2')
        )],
    )
    # Keep plot responsive to the browser window, rather than requiring a
    # specific screen resolution or opening inside Spyder's static plot pane.
    say('STEP 6/6: Saving HTML and summary...')
    # SKY VIEW: reconstruct equatorial sky directions from Cartesian positions.
    # This uses the same galaxy assignments and PCA morphology catalog as the 3D map.
    say('  Creating companion all-sky RA/Dec map...')
    sky = go.Figure()
    def add_sky(points, label, color, size=2, opacity=.65, custom=None):
        if len(points) == 0:
            return
        xyz = points[['X_Mpc','Y_Mpc','Z_Mpc']].to_numpy(dtype=float)
        radius = np.linalg.norm(xyz, axis=1)
        good = np.isfinite(radius) & (radius > 0)
        xyz = xyz[good]
        ra = (np.degrees(np.arctan2(xyz[:,1],xyz[:,0])) + 180) % 360 - 180
        dec = np.degrees(np.arcsin(np.clip(xyz[:,2] / radius[good], -1, 1)))
        kwargs = dict(lon=ra,lat=dec,mode='markers',name=label,
                      marker=dict(size=size,color=color,opacity=opacity),
                      hovertemplate='RA (wrapped): %{lon:.1f}°<br>Dec: %{lat:.1f}°<extra>'+label+'</extra>')
        if custom is not None:
            kwargs['customdata'] = np.asarray(custom)[good]
            kwargs['hovertemplate'] = ('Group: %{customdata[0]}<br>RA (wrapped): %{lon:.1f}°'
                                       '<br>Dec: %{lat:.1f}°<extra>'+label+'</extra>')
        sky.add_trace(go.Scattergeo(**kwargs))

    for survey, color in [('SDSS','#5279b3'),('6DFGS','#a36c91')]:
        subset = sample(galaxies[galaxies.Survey.astype(str).str.upper() == survey], 22000)
        add_sky(subset, f'{survey} observed galaxy positions (sample)', color, 1.6, .33)
    for morphology, color in MORPH_COLORS.items():
        subset = sample(assigned[assigned.Morphology_Candidate == morphology], 13000)
        if len(subset):
            ids = subset.Global_Group.astype(str).to_numpy().reshape(-1,1)
            add_sky(subset, morphology+' DBSCAN candidates', color, 2.5, .8, ids)

    sky.add_trace(go.Scattergeo(
        lon=[((ra+180)%360)-180 for _,ra,dec,z in REFERENCES],
        lat=[dec for _,ra,dec,z in REFERENCES],
        mode='markers', name='Published cluster reference positions',
        marker=dict(size=11,color='#ffffff',symbol='diamond',line=dict(color='#172235',width=1)),
        text=[f'{name}<br>RA {ra:.2f}°, Dec {dec:.2f}°<br>Approximate reference position'
              for name,ra,dec,z in REFERENCES],hoverinfo='text'))
    sky.update_layout(
        title=dict(text='<b>WHERE ARE THE STRUCTURES IN THE SKY?</b><br>'
                   '<span style="font-size:13px;color:#b7c7dc">'
                   'All-sky equatorial view · Right Ascension and Declination · Distance is not shown'
                   '</span>',x=.5,xanchor='center',y=.975,font=dict(size=23)),
        paper_bgcolor='#080e1b',font=dict(color='#f1f5ff',family='Arial'),
        height=780, margin=dict(l=40,r=40,t=145,b=135),
        legend=dict(orientation='h',x=.5,xanchor='center',y=-.10,yanchor='top',
                    bgcolor='rgba(12,22,38,.83)',font=dict(size=10),
                    entrywidth=190,entrywidthmode='pixels'),
        geo=dict(domain=dict(x=[.03,.97],y=[.08,.91]),projection=dict(type='mollweide'),
                 bgcolor='#080e1b',showland=False,showocean=False,
                 showcoastlines=False,showframe=True,framecolor='#5e7598',
                 lonaxis=dict(showgrid=True,gridcolor='#34445d',dtick=60),
                 lataxis=dict(showgrid=True,gridcolor='#34445d',dtick=30)),
        annotations=[dict(x=.5,y=-.29,xref='paper',yref='paper',showarrow=False,
             text='Mollweide projection of the celestial sphere · RA / Dec only · Survey gaps are not cosmic voids.',
             font=dict(size=11,color='#aebed2'))]
    )
    plot3d = pio.to_html(fig, include_plotlyjs=True, full_html=False,
                         config={'responsive':True,'displaylogo':False},div_id='cosmic-3d')
    plotsky = pio.to_html(sky, include_plotlyjs=False, full_html=False,
                          config={'responsive':True,'displaylogo':False},div_id='cosmic-sky')
    html = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>The Observed Cosmic Web | 3D and Sky View</title>
    <style>
    body{margin:0;background:#080e1b;color:#f1f5ff;font-family:Arial,sans-serif}
    .tabs{display:flex;justify-content:center;gap:10px;padding:12px 8px 0}
    .tabs button{background:#263a57;color:#fff;border:1px solid #4c6484;
      border-radius:5px;padding:11px 23px;font-weight:bold;cursor:pointer}
    .tabs button[aria-selected="true"]{background:#38577c;border-color:#7d9dc8}
    .panel{width:100%;} .hidden{display:none}
    .note{text-align:center;color:#b7c7dc;font-size:12px;padding:0 15px 12px}
    </style></head><body>
    <div class="tabs" role="tablist" aria-label="Cosmic web views">
    <button id="tab3d" role="tab" aria-selected="true" onclick="switchView('3d')">🌌 3D Cosmic Web</button>
    <button id="tabsky" role="tab" aria-selected="false" onclick="switchView('sky')">🔭 Sky View (RA / Dec)</button>
    </div>
    <div class="panel" id="panel3d">""" + plot3d + """</div>
    <div class="panel hidden" id="panelsky">""" + plotsky + """</div>
    <div class="note">The 3D view shows spatial distances; the Sky View shows angular directions only.</div>
    <script>
    function switchView(which){
      const sky=which==='sky';
      document.getElementById('panel3d').classList.toggle('hidden',sky);
      document.getElementById('panelsky').classList.toggle('hidden',!sky);
      document.getElementById('tab3d').setAttribute('aria-selected',String(!sky));
      document.getElementById('tabsky').setAttribute('aria-selected',String(sky));
      requestAnimationFrame(()=>Plotly.Plots.resize(document.getElementById(sky?'cosmic-sky':'cosmic-3d')));
    }
    </script></body></html>"""
    OUTPUT.write_text(html, encoding='utf-8')
    summary = pd.DataFrame([
        {'Metric':'Galaxies in input assignments','Value':len(galaxies)},
        {'Metric':'Characterized DBSCAN groups','Value':len(structures)},
        {'Metric':'Experiment 7 proximity-match rows','Value':len(matches)},
        {'Metric':'Experiment 8 exploratory empty-sphere candidates','Value':len(voids)},
        {'Metric':'Displayed morphology categories','Value':len(MORPH_COLORS)},
        {'Metric':'Status','Value':'Exploratory; not physically validated'},
    ])
    summary.to_csv(SUMMARY,index=False)
    say(f'Saved: {OUTPUT}')
    say(f'Saved: {SUMMARY}')
    webbrowser.open(OUTPUT.as_uri())
    say('COMPLETE! Interactive map should open in your browser.')


if __name__ == '__main__':
    main()