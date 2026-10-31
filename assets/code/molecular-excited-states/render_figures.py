#!/usr/bin/env python3
"""Render the article's geometry, energy ladder and occupation diagram.

Run with Python 3 from any directory. Only the adjacent public JSON is read.
The completed reference label is reused; this script does not run PySCF.
"""
from pathlib import Path
import collections
import hashlib
import html
import json
import math

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'assets/img/molecular-excited-states'
INK, MUTED, BLUE, WARM, LINE = '#222222', '#666666', '#315f7d', '#a3654a', '#deded8'


def text(x, y, value, size=18, color=INK, anchor='start', weight='normal'):
    return f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(value))}</text>'


def line(x1, y1, x2, y2, color=LINE, width=1.5, dash=''):
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'


def save(name, body, title, description, width=640, height=440):
    raw=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
         f'<title id="title">{html.escape(title)}</title>\n<desc id="desc">{html.escape(description)}</desc>\n'
         '<rect width="100%" height="100%" fill="#fbfbfa"/>\n'
         '<g font-family="Arial, Helvetica, sans-serif">\n'+body+'\n</g>\n</svg>\n')
    (OUT/name).write_bytes(raw.encode('utf-8'))


def bonds(data):
    radii=data['display']['covalent_radii_angstrom'];result=[]
    for i,p in enumerate(data['positions']):
        for j,q in enumerate(data['positions'][:i]):
            distance=math.dist(p,q)
            cutoff=data['display']['bond_threshold_factor']*(radii[str(data['atomic_numbers'][i])]+radii[str(data['atomic_numbers'][j])])
            if 0.4<distance<cutoff:result.append((i,j))
    return result


def rotation(v, ax=-0.42, ay=0.28):
    x,y,z=v;y,z=math.cos(ax)*y-math.sin(ax)*z,math.sin(ax)*y+math.cos(ax)*z
    return (math.cos(ay)*x+math.sin(ay)*z,y,-math.sin(ay)*x+math.cos(ay)*z)


def geometry(data):
    positions=data['positions'];center=[sum(p[k] for p in positions)/30 for k in range(3)]
    xyz=[rotation([p[k]-center[k] for k in range(3)]) for p in positions]
    scale=min(490/(max(p[0] for p in xyz)-min(p[0] for p in xyz)),300/(max(p[1] for p in xyz)-min(p[1] for p in xyz)))
    points=[(320+p[0]*scale,225-p[1]*scale,p[2]) for p in xyz]
    body=text(28,34,'One sampled molecular geometry',22,weight='bold')
    body+=text(28,60,'30 atoms · C13H14N2O · isolated and neutral',16,color=MUTED)
    primitives=[]
    for i,j in bonds(data):
        p,q=points[i],points[j];primitives.append(((p[2]+q[2])/2,line(p[0],p[1],q[0],q[1],'#a8aaa9',4)))
    styles={1:('#ffffff','#8a8e91',6.5,''),6:('#555c60','#40484c',11,''),7:(BLUE,'#244b65',12,'N'),8:(WARM,'#86503a',12,'O')}
    for p,z in zip(points,data['atomic_numbers']):
        fill,stroke,r,label=styles[z]
        mark=f'<circle cx="{p[0]:.3f}" cy="{p[1]:.3f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        if label:mark+=text(p[0],p[1]+4,label,12,color='#ffffff',anchor='middle',weight='bold')
        primitives.append((p[2],mark))
    body+=''.join(mark for _,mark in sorted(primitives,key=lambda item:item[0]))
    for x,z,label in [(55,6,'C'),(140,1,'H'),(225,7,'N'),(310,8,'O')]:
        fill,stroke,_,_=styles[z];body+=f'<circle cx="{x}" cy="400" r="7" fill="{fill}" stroke="{stroke}"/>'+text(x+14,406,label,17)
    body+=text(28,432,'Distance guides show connectivity only; no bond order is assigned.',14,color=MUTED)
    save('motor-geometry.svg',body,'Actual sampled motor geometry','Orthographic ball-and-stick view of one completed reference geometry. Carbon is dark gray, hydrogen white, nitrogen blue, oxygen brown. Sizes are symbolic, and distance guides do not assign bond orders.',height=450)


def energy_ladder(data):
    e0,e1=data['energies'];gap=data['derived']['gap_eV'];mean=(e0+e1)/2
    xleft,xright=152,362
    y=lambda e:345-e/3.6*270
    body=text(28,34,'Two electronic states at one fixed geometry',21,weight='bold')
    body+=text(28,62,'Relative total energy, E − E0 (eV)',16,color=MUTED)
    body+=line(102,75,102,352,color=MUTED)
    for value in [0,1,2,3]:
        yy=y(value);body+=line(97,yy,108,yy,color=MUTED)+text(89,yy+5,str(value),16,anchor='end',color=MUTED)
    for label,value,absolute,color in [('S0',0,e0,BLUE),('S1',gap,e1,WARM)]:
        yy=y(value);body+=line(xleft,yy,xright,yy,color,4)+text(xleft+8,yy-13,label,20,color=color,weight='bold')
        body+=text(390,yy+5,f'{absolute:.7f} Ha',16,color=color)
    yy=y(gap/2);body+=line(xleft-4,yy,xright+4,yy,MUTED,2,'7 5')
    body+=text(390,yy-8,'Equal-weight mean',16,color=MUTED)+text(390,yy+15,f'{mean:.7f} Ha',15,color=MUTED)
    x=330;body+=line(x,y(gap)+8,x,y(0)-8,BLUE,1.5)
    body+=f'<path d="M {x-4} {y(gap)+14:.3f} L {x} {y(gap)+7:.3f} L {x+4} {y(gap)+14:.3f} M {x-4} {y(0)-14:.3f} L {x} {y(0)-7:.3f} L {x+4} {y(0)-14:.3f}" fill="none" stroke="{BLUE}" stroke-width="1.5"/>'
    body+=text(168,389,f'Excitation gap: {gap:.6f} eV',20,color=BLUE,weight='bold')
    body+=text(28,424,'The dashed line is an optimization objective, not a third electronic state.',14,color=MUTED)
    save('two-state-energy.svg',body,'Actual two-state total energies and their state-average objective','At one completed reference geometry S1 is 3.141262 eV above S0. The dashed equal-weight mean is the orbital-optimization objective. No orbital eigenvalues or potential-energy curve is shown.')


def active_space():
    body=text(26,32,'An occupation model for 114 electrons',22,weight='bold')
    body+=text(26,62,'Orbital shapes are optimized; these boxes specify allowed occupancies.',15,color=MUTED)
    body+=f'<rect x="26" y="84" width="588" height="68" rx="5" fill="#f1f2ef" stroke="{LINE}"/>'
    body+=text(43,112,'Inactive sector: 51 doubly occupied spatial orbitals',18,weight='bold')
    body+=text(43,136,'102 electrons remain paired in every included configuration.',15,color=MUTED)
    body+=text(26,184,'Active sector: 12 electrons in 9 spatial orbitals',19,color=BLUE,weight='bold')
    for row,(label,occupations) in enumerate([('A',[2,2,2,2,2,2,0,0,0]),('B',[2,2,2,2,2,1,1,0,0])]):
        yy=208+row*77;body+=text(30,yy+27,label,18,color=BLUE,weight='bold')
        for index,count in enumerate(occupations):
            x=72+index*59
            body+=f'<rect x="{x}" y="{yy}" width="44" height="40" rx="3" fill="#ffffff" stroke="{BLUE}" stroke-width="1.5"/>'
            for n in range(count):body+=f'<circle cx="{x+15+n*14}" cy="{yy+20}" r="4" fill="{BLUE}"/>'
            if row==0:body+=text(x+22,yy-5,str(index+1),12,color=MUTED,anchor='middle')
    body+=text(26,370,'External orbitals are empty in every included configuration.',16,color=MUTED)
    body+=text(26,405,'A and B are illustrative occupations, not measured CI weights.',15,color=MUTED)
    body+=text(26,427,'Spin-adapted singlet states combine determinants with the required spin.',14,color=MUTED)
    save('active-space.svg',body,'CASSCF active, inactive and external orbital occupations','There are 51 doubly occupied inactive spatial orbitals with 102 electrons and 9 active spatial orbitals containing 12 electrons. Two illustrative active occupation patterns contain twelve dots each. No actual orbital energies, CI weights or state assignment is implied.',height=445)


def validate(data):
    assert data['schema']=='molecular-excited-states-example-v1'
    assert collections.Counter(data['atomic_numbers'])=={1:14,6:13,7:2,8:1}
    assert sum(data['atomic_numbers'])==114==2*51+12
    assert data['units']=={'positions':'angstrom','energies':'hartree','forces':'hartree/bohr','nac_01':'1/bohr','G_01':'hartree/bohr'}
    assert len(data['positions'])==30 and len(data['forces'])==2 and all(len(s)==30 for s in data['forces'])
    assert all(len(p)==3 and all(math.isfinite(v) for v in p) for p in data['positions']+data['forces'][0]+data['forces'][1]+data['nac_01']+data['G_01'])
    gap=data['energies'][1]-data['energies'][0]
    assert abs(gap*data['constants']['hartree_to_eV']-data['derived']['gap_eV'])<1e-12
    assert abs(sum(data['energies'])/2-data['derived']['state_average_energy_hartree'])<1e-12
    residual=max(abs(gap*d-g) for drow,grow in zip(data['nac_01'],data['G_01']) for d,g in zip(drow,grow))
    assert residual<1e-11
    distance_error=max(abs(math.dist(a,b)-math.dist(rotation(a),rotation(b))) for a in data['positions'] for b in data['positions'])
    assert distance_error<1e-12
    return {'finite_and_shapes':True,'electron_count':114,'gap_eV':data['derived']['gap_eV'],
            'state_average_energy_hartree':data['derived']['state_average_energy_hartree'],
            'G_gap_NAC_residual_hartree_per_bohr':residual,'rotation_distance_error_angstrom':distance_error,
            'schematic_occupation_counts':[12,12],'evidence_scope':'One completed reference label; no basis or active-space convergence claim.'}


if __name__=='__main__':
    data=json.loads((HERE/'motor-example.json').read_text(encoding='utf-8'));metrics=validate(data)
    OUT.mkdir(parents=True,exist_ok=True);geometry(data);energy_ladder(data);active_space()
    metrics['source_example_sha256']=hashlib.sha256((HERE/'motor-example.json').read_bytes()).hexdigest()
    metrics['figures']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.svg')}
    (OUT/'figure-checks.json').write_bytes((json.dumps(metrics,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'status':'PASS','gap_eV':metrics['gap_eV'],'figures':list(metrics['figures'])}))
