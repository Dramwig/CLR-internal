"""Draw a vector overview of the existing CLR method; no experimental numbers."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--font-dir', type=Path, default=Path('C:/Windows/Fonts'))
args = parser.parse_args()
ROOT, fonts = args.project_root, args.font_dir
for name, file in [('CLRRegular', 'arial.ttf'), ('CLRBold', 'arialbd.ttf'), ('CLRItalic', 'ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
OUT = ROOT / 'outputs' / 'fig_arch.pdf'
OUT.parent.mkdir(exist_ok=True)
c = canvas.Canvas(str(OUT), pagesize=(252, 220), pageCompression=1)
c.setTitle('Calibrated Latent Reader: architecture')
c.setAuthor('')
ink, blue, green = map(HexColor, ['#203044', '#2671A8', '#247A65'])

def box(x,y,w,h,title,detail=None,fill='#F2F6FA',stroke='#B9CADA',size=8):
    c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke)); c.setLineWidth(.65)
    c.roundRect(x,y,w,h,4,fill=1,stroke=1)
    c.setFillColor(ink);c.setFont('CLRBold',size)
    assert pdfmetrics.stringWidth(title,'CLRBold',size) <= w-6, title
    c.drawCentredString(x+w/2,y+h/2+(3 if detail else -2.5),title)
    if detail:
        c.setFont('CLRRegular',6.8);c.drawCentredString(x+w/2,y+h/2-8,detail)

def arrow(x1,y1,x2,y2,color=blue):
    c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(.9)
    c.line(x1,y1,x2,y2)
    a=math.atan2(y2-y1,x2-x1);s=3.5
    p=c.beginPath();p.moveTo(x2,y2)
    p.lineTo(x2-s*math.cos(a-.45),y2-s*math.sin(a-.45))
    p.lineTo(x2-s*math.cos(a+.45),y2-s*math.sin(a+.45));p.close()
    c.drawPath(p,fill=1,stroke=0)

box(6,180,114,34,'Latent object z','Thought prefix / pooled KV')
box(132,180,114,34,'Typed question q','Template + candidate options',fill='#F0F8F5',stroke='#BED9CF')
arrow(63,180,63,170)
box(6,136,114,34,'Source adapter','Adapted tokens + step / source')
arrow(189,180,189,112)
arrow(63,136,63,112)
box(6,73,240,39,'Shared bidirectional reader','Block mask: options read latents + question',size=9)
c.setFillColor(ink);c.setFont('CLRItalic',6.6)
c.drawCentredString(126,77,'No attention between different options')
arrow(126,73,126,64)
box(6,35,240,29,'Typed output heads','Joint state x option (choice); yes/no; score',fill='#F0F8F5',stroke='#BED9CF',size=8)
arrow(126,35,126,26,color=green)
box(6,3,240,23,'Temperature calibration  >  Mondrian conformal sets',fill='#F6F3EA',stroke='#DDD0AD',size=7.1)
c.showPage();c.save()
print(OUT)
