"""Replace independent-refit poster panels with original-model GitHub evidence.

Run from repository root: python poster/scripts/build_original_poster.py
Requires: python-pptx, Pillow; optional LibreOffice for --export-pdf.
Never trains a model or edits research datasets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps
from pptx import Presentation
from pptx.util import Inches

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'poster'
FIG=P/'figures'
REPORT=P/'evidence/original_model_verification.json'
SOURCE=P/'source/GeoValue_TechXhibit8_Base.pptx'
OUTPUT=P/'final'
FILES={
 'observed':FIG/'01_observed_2025_sales.png',
 'gradient':FIG/'02_original_model_surface_and_gradient.png',
 'scatter':FIG/'03_original_2025_actual_vs_predicted.png',
 'bracket':FIG/'04_original_2025_price_bracket_errors.png',
 'support':FIG/'05_original_model_spatial_support_errors.png',
 'sensitivity':FIG/'07_original_model_gradient_sensitivity.png',
}
# Source poster has stable, verified shape locations (inches).
SLOTS={54:(1.20,26.88,11.35,2.67),58:(13.45,10.55,21.35,9.25),63:(13.45,22.06,12.60,6.42),
       64:(26.40,22.32,8.33,6.08),109:(35.60,24.70,9.85,4.50)}
# Prefix searches avoid reliance on indices after embedded pictures are removed.
REPLACE={
 'Our Spatial Evidence Indicator groups':
  'The Spatial Evidence Indicator uses tenth-nearest training-location distance. In the original 2025 audit, sparse-support areas had $69.4k MAE versus $54.1k for dense areas. Individual error association remains weak.',
 'A: observed 2025 sale records':
  'A: observed 2025 recorded sales  |  B: original saved XGBoost gradient',
 'A: actual sale medians':
  'A: recorded 2025 sale medians (minimum eight sales per hexagon). B: gradient of a smoothed fixed-property scenario (sigma 1.5). Not observed appreciation or a causal effect.',
 'Gradient check:':
  'Original model: mean gradient at sigma 1.0 / 1.5 / 2.5 = $2,092 / $1,922 / $1,690 per km. Magnitudes vary with smoothing.',
 'Independent refit:':
  'Original 2025 audit: sparse MAE $69.4k (n=2,463), dense $54.1k (n=3,559). 2024 validation Spearman rho=0.0969 (weak). Descriptive support, not a confidence interval.',
 'Linear baseline is distorted':
  'Original Linear Regression is distorted by an anomalous area-ratio record; it is not a definitive comparison with well-regularized linear regression.',
 'Current assessment attributes':
  'Assessment attributes may postdate historical sales. This is retrospective modeling, not verified forecasting. Numerical gradients depend on smoothing and fixed property/location categories.',
 'Original XGBoost results:':
  '2025 prediction, gradient and support audit use the verified original saved model. The 2024 support calibration uses a historical pre-2024 fit. Evidence and hashes on GitHub.',
}
BANNED=('independent refit','independent reproduction','independently reproduced',
        'model-refit','0.8378','bootstrap 95%','$69.9k','$53.8k')


def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def verify():
 for path in [REPORT,*FILES.values()]:
  if not path.is_file():raise RuntimeError(f'Missing committed original evidence: {path}')
 report=json.loads(REPORT.read_text())
 assert report['saved_model_verified_against_original_2025_report'] is True
 assert report['model_selected']=='xgboost' and report['test_rows']==8691
 assert report['spatial_evidence']['2025_rows']==8691
 got=report['recomputed_2025_test_metrics'];published=report['reported_2025_test_metrics']
 for key in ('mae','rmse','r2'):
  if abs(got[key]-published[key])>1e-7:raise RuntimeError(f'Original {key} is inconsistent')
 if abs(got['r2']-0.8411996698339903)>1e-8:raise RuntimeError('Unexpected model: R2 differs')
 return report


def fitted(image,size):
 canvas=Image.new('RGB',size,'white')
 im=ImageOps.contain(image,size,method=Image.Resampling.LANCZOS)
 canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2))
 return canvas


def hero(dest):
 with Image.open(FILES['observed']) as obj: a=obj.convert('RGB')
 with Image.open(FILES['gradient']) as obj:b=obj.convert('RGB')
 # Select the observed-price panel and the original-gradient panel respectively.
 a=a.crop((0,int(a.height*.045),int(a.width*.5),a.height))
 b=b.crop((int(b.width*.5),int(b.height*.045),b.width,b.height))
 width,height=2020,1700
 out=Image.new('RGB',(width*2+36,height),'white')
 out.paste(fitted(a,(width,height)),(0,0))
 out.paste(fitted(b,(width,height)),(width+36,0))
 path=dest/'original_model_hero.png';out.save(path,optimize=True)
 return path


def formatted(src,dest,name,w,h):
 with Image.open(src) as im: img=fitted(im.convert('RGB'),(int(w*105),int(h*105)))
 path=dest/(name+'.png');img.save(path,optimize=True);return path


def neutral_ui(picture,dest):
 # Retain the real browser interface while removing the old audit-model price.
 with Image.open(BytesIO(picture.image.blob)) as image:img=image.convert('RGB')
 w,h=img.size
 canvas=Image.new('RGB',(w,h),'white')
 cropped=img.crop((0,0,int(w*.80),h))
 canvas.paste(cropped,(0,0))
 path=dest/'application_inputs_only.png';canvas.save(path,optimize=True)
 return path


def put_text(shape,new):
 p=shape.text_frame.paragraphs[0]
 if p.runs:
  p.runs[0].text=new
  for extra in p.runs[1:]:extra.text=''
 else:p.text=new
 for extra in list(shape.text_frame.paragraphs)[1:]:
  extra._element.getparent().remove(extra._element)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--template',type=Path,default=SOURCE)
 ap.add_argument('--out-dir',type=Path,default=OUTPUT)
 ap.add_argument('--export-pdf',action='store_true')
 a=ap.parse_args();report=verify()
 if not a.template.is_file():raise SystemExit(f'Missing editable source poster: {a.template}')
 dest=a.out_dir;dest.mkdir(parents=True,exist_ok=True)
 prs=Presentation(a.template)
 if len(prs.slides)!=1 or abs(prs.slide_width/Inches(1)-46.81)>.03:
  raise RuntimeError('This script requires the supplied one-slide A0 template')
 slide=prs.slides[0]
 for i,coords in SLOTS.items():
  s=slide.shapes[i]
  actual=(s.left/Inches(1),s.top/Inches(1),s.width/Inches(1),s.height/Inches(1))
  if s.shape_type!=13 or any(abs(u-v)>.08 for u,v in zip(actual,coords)):
   raise RuntimeError(f'Poster layout changed at picture slot {i}')
 media={54:neutral_ui(slide.shapes[54],dest),58:hero(dest),63:formatted(FILES['support'],dest,'original_support_chart',12.6,6.42),
        64:formatted(FILES['scatter'],dest,'original_actual_predicted',8.33,6.08),
        109:formatted(FILES['bracket'],dest,'original_price_bracket',9.85,4.5)}
 for i in sorted(media,reverse=True):
  shape=slide.shapes[i]
  dimensions=(shape.left,shape.top,shape.width,shape.height)
  r_id=shape._element.blipFill.blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
  elm=shape._element;elm.getparent().remove(elm)
  if r_id:slide.part.drop_rel(r_id)
  slide.shapes.add_picture(str(media[i]),*dimensions)
 for prefix,new in REPLACE.items():
  matches=[s for s in slide.shapes if s.has_text_frame and s.text.startswith(prefix)]
  if len(matches)!=1:raise RuntimeError(f'Missing/ambiguous caption: {prefix}')
  put_text(matches[0],new)
 result=dest/'GeoValue_TechXhibit8_Original_Model_Final.pptx';prs.save(result)
 text=' '.join(s.text for s in Presentation(result).slides[0].shapes if s.has_text_frame).lower()
 for phrase in BANNED:
  if phrase in text:raise RuntimeError(f'Old reproduced-model phrase still present: {phrase}')
 manifest={'source_original_figures_commit':'5519c03a1adac44f7ad5338788739a3715a3a196',
  'original_model_verified':True,'2025_metrics':report['recomputed_2025_test_metrics'],
  '2025_support_audit':report['spatial_evidence']['held_out_2025_support_audit'],
  'support_validation_spearman':report['spatial_evidence']['spearman_validation'],
  'original_model_sha256':report['input_sha256']['artifacts/official/residential-sale-model.joblib'],
  'input_figure_sha256':{k:digest(v) for k,v in FILES.items()},
  'powerpoint_sha256':digest(result),
  'scope':'Retrospective sale-price prediction. The 2024 support calibration uses a pre-2024 fit. The original final model supplies 2025 predictions and the numerical gradient scenario, not causal price changes.'}
 (dest/'ORIGINAL_MODEL_RELEASE.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if a.export_pdf:
  cmd=shutil.which('libreoffice') or shutil.which('soffice')
  if cmd:
   subprocess.run([cmd,'-env:UserInstallation=file:///tmp/geovalue-poster-original',
     '--headless','--convert-to','pdf','--outdir',str(dest),str(result)],check=True)
  else:print('PDF not exported: use PowerPoint/Keynote Export to PDF on macOS.')
 print(f'Verified original-model poster: {result}')

if __name__=='__main__':main()
