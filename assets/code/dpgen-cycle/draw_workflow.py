"""Generate the conceptual DP-GEN cycle SVG; this does not run DP-GEN."""
from pathlib import Path
import argparse
import html


def diagram():
    rows = [
        ('Training dataset', ['Seed + accepted additions']),
        ('1 · Train committee', ['Same data; independent seeds']),
        ('2 · Explore', ['One leader; all members inspect']),
        ('3 · Select', ['Spread + geometry + diversity', 'Within the labeling budget']),
        ('4 · Label', ['Same-protocol quantum chemistry']),
        ('5 · Validate and merge', ['Accept distinct, usable labels']),
    ]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="420" height="760" viewBox="0 0 420 760" role="img" aria-labelledby="title desc">',
             '<title id="title">The DP-GEN learning cycle</title>',
             '<desc id="desc">Training data feed a committee, which explores and selects new geometries. Quantum chemistry labels selected geometries. Only validated distinct labels expand the dataset for the next iteration.</desc>',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#416979"/></marker></defs>',
             '<rect width="420" height="760" fill="#fafbfc"/>',
             '<path d="M95 691 H35 V61 H95" fill="none" stroke="#416979" stroke-width="2" marker-end="url(#arrow)"/>',
             '<text x="22" y="376" transform="rotate(-90 22 376)" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#416979">Next iteration · expanded dataset</text>']
    for i, (title, notes) in enumerate(rows):
        y = 16 + 126 * i
        fill = '#e9f1f4' if i == 0 else '#ffffff'
        parts.append(f'<rect x="95" y="{y}" width="300" height="90" rx="7" fill="{fill}" stroke="#93abb4" stroke-width="1.3"/>')
        parts.append(f'<text x="245" y="{y+30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="19" font-weight="700" fill="#233e4d">{html.escape(title)}</text>')
        for j, note in enumerate(notes):
            parts.append(f'<text x="245" y="{y+55+19*j}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#465861">{html.escape(note)}</text>')
        if i < len(rows)-1:
            parts.append(f'<path d="M245 {y+108} V{y+122}" fill="none" stroke="#416979" stroke-width="2" marker-end="url(#arrow)"/>')
    return '\n'.join(parts) + '\n</svg>\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'img/dpgen-cycle/learning-loop.svg')
    out = p.parse_args().output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(diagram().encode('utf-8'))
    print('Wrote the workflow SVG.')
