"""Evaluate YOLO detections and image-level counts on the test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
from src.validate_dataset import load_dataset_config, validate_dataset

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
CONFIDENCE_CANDIDATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def class_target_ids(names):
    values = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(index) for index, name in values if 'lanternfly' in str(name).lower()}


def ground_truth_boxes(label_path: Path, width: int, height: int):
    boxes = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_id, cx, cy, bw, bh = map(float, fields)
        if int(class_id) != 0:
            continue
        boxes.append(((cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height))
    return boxes


def match_counts(gt, predictions, iou_threshold=0.5):
    used = set(); tp = 0
    for pred in sorted(predictions, key=lambda item: item[1], reverse=True):
        best_iou, best_index = 0.0, -1
        x1,y1,x2,y2=pred[0]
        for index,target in enumerate(gt):
            if index in used: continue
            a1,b1,a2,b2=target
            iw=max(0,min(x2,a2)-max(x1,a1)); ih=max(0,min(y2,b2)-max(y1,b1))
            inter=iw*ih; ap=max(0,x2-x1)*max(0,y2-y1); ag=max(0,a2-a1)*max(0,b2-b1)
            iou=inter/max(1e-9,ap+ag-inter)
            if iou>best_iou: best_iou,best_index=iou,index
        if best_iou>=iou_threshold: used.add(best_index); tp+=1
    return tp,len(predictions)-tp,len(gt)-tp


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data.yaml')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--conf',type=float,default=None,help='Override confidence; otherwise select using validation F1')
    parser.add_argument('--device',default=None,help='For example cpu or 0; omit for auto')
    args=parser.parse_args()
    if args.imgsz<32 or (args.conf is not None and not 0<=args.conf<=1):parser.error('Image size must be >=32 and confidence within [0,1].')
    weights,data=args.weights.resolve(),args.data.resolve()
    if not weights.is_file():parser.error(f'Weights not found: {weights}')
    report=validate_dataset(data);print(report.format())
    if not report.ok or not report.images['test']:parser.error('Dataset must pass validation and have a test split.')
    config,dataset_root=load_dataset_config(data)
    if report.names!=['lanternfly']:parser.error('Evaluation expects one class named lanternfly.')
    def split_files(split):
        image_dir=Path(config[split]); image_dir=image_dir if image_dir.is_absolute() else dataset_root/image_dir
        labels_cfg=config.get('labels',{})
        label_dir=Path(labels_cfg.get(split,f'labels/{split}')) if isinstance(labels_cfg,dict) else dataset_root/f'labels/{split}'
        if not label_dir.is_absolute():label_dir=dataset_root/label_dir
        images=sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        labels={p.stem.casefold():p for p in label_dir.glob('*.txt')}
        if any(p.stem.casefold() not in labels for p in images):raise ValueError(f'{split} image/label names do not match')
        return images,labels
    try:
        val_images,val_labels=split_files('val');test_images,test_labels=split_files('test')
    except (KeyError,ValueError) as exc:parser.error(str(exc))
    try:
        from ultralytics import YOLO
    except ImportError as exc:parser.error(f'Ultralytics is unavailable: {exc}')
    model=YOLO(str(weights)); target_ids=class_target_ids(model.names)
    if not target_ids:parser.error('Checkpoint has no class whose name contains lanternfly.')

    # Select an operating confidence on validation only. One prediction pass supplies
    # candidates for all thresholds; test photos never participate in selection.
    val_predictions=[]
    for path in val_images:
        with Image.open(path) as im:w,h=im.size
        gt=ground_truth_boxes(val_labels[path.stem.casefold()],w,h)
        result=model.predict(source=str(path),imgsz=args.imgsz,conf=min(CONFIDENCE_CANDIDATES),iou=0.45,max_det=1000,device=args.device,verbose=False)[0]
        predictions=[]
        if result.boxes is not None:
            for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                if int(cls) in target_ids:predictions.append((box,float(score)))
        val_predictions.append((path.name,gt,predictions))
    if not val_predictions:parser.error('Validation split is empty; cannot select a confidence threshold.')
    sweep=[]
    for threshold in CONFIDENCE_CANDIDATES:
        tp=fp=fn=abs_error=exact=0
        for _name,gt,preds in val_predictions:
            kept=[p for p in preds if p[1]>=threshold]
            a,b,c=match_counts(gt,kept);tp+=a;fp+=b;fn+=c
            abs_error+=abs(len(gt)-len(kept));exact+=len(gt)==len(kept)
        precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);f1=2*precision*recall/max(1e-12,precision+recall)
        sweep.append({'confidence':threshold,'precision_iou50':precision,'recall_iou50':recall,'f1_iou50':f1,'mean_absolute_count_error':abs_error/len(val_predictions),'exact_count_images':exact,'images':len(val_predictions)})
    selected=args.conf if args.conf is not None else max(sweep,key=lambda row:(row['f1_iou50'],row['precision_iou50'],row['confidence']))['confidence']

    metrics_obj=model.val(data=str(data),split='test',imgsz=args.imgsz,conf=0.001,device=args.device,plots=False,verbose=False)
    metrics={}
    for key,value in getattr(metrics_obj,'results_dict',{}).items():
        if hasattr(value,'item'):value=value.item()
        if isinstance(value,(int,float)) and math.isfinite(float(value)):metrics[str(key)]=float(value)
    per_image=[]; example_candidates=[]
    for path in test_images:
        with Image.open(path) as im:w,h=im.size
        gt=ground_truth_boxes(test_labels[path.stem.casefold()],w,h)
        result=model.predict(source=str(path),imgsz=args.imgsz,conf=selected,iou=0.45,max_det=1000,device=args.device,verbose=False)[0]
        preds=[]
        if result.boxes is not None:
            for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                if int(cls) in target_ids:preds.append((box,float(score)))
        tp,fp,fn=match_counts(gt,preds)
        count_error=abs(len(gt)-len(preds))
        per_image.append({'image':path.name,'ground_truth_count':len(gt),'predicted_count':len(preds),'absolute_count_error':count_error,'tp_iou50':tp,'fp_iou50':fp,'fn_iou50':fn})
        score=count_error*1000+min(len(gt),999)
        cutoff=min((item[0] for item in example_candidates),default=-1)
        if len(example_candidates)<8 or score>cutoff:
            import numpy as np
            plotted=result.plot()
            annotated=Image.fromarray(np.asarray(plotted)[:,:,::-1].copy())
            draw=ImageDraw.Draw(annotated)
            for box in gt:
                draw.rectangle(box,outline=(255,0,255),width=max(2,round(max(w,h)/500)))
            draw.rectangle((0,0,min(w,440),28),fill=(255,255,255))
            draw.text((6,5),f'GT={len(gt)}  prediction={len(preds)}  count error={count_error}',fill=(20,20,20))
            example_candidates.append((score,path.name,annotated))
            example_candidates=sorted(example_candidates,key=lambda item:item[0],reverse=True)[:8]
    count_mae=sum(row['absolute_count_error'] for row in per_image)/max(1,len(per_image))
    manifest_path=dataset_root/'dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    out=ROOT/'runs'/'evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    examples_dir=out/'example_images';examples_dir.mkdir()
    example_records=[]
    for _score,name,annotated in example_candidates:
        destination=examples_dir/f'{Path(name).stem}_gt_pred.jpg';annotated.save(destination,quality=90)
        example_records.append({'image':name,'annotated_example':str(destination)})
    summary={'evaluated_at':datetime.now().astimezone().isoformat(),'weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'dataset':str(data),'source_test_history':manifest.get('test_history','unknown'),'validation_images':len(val_images),'confidence_threshold':selected,'threshold_selection':'validation maximum F1 at IoU 0.50; ties prefer precision, then higher confidence' if args.conf is None else 'explicit override','validation_threshold_sweep':sweep,'test_images':len(test_images),'test_images_with_boxes':sum(r['ground_truth_count']>0 for r in per_image),'test_empty_images':sum(r['ground_truth_count']==0 for r in per_image),'test_ground_truth_adult_insects':sum(r['ground_truth_count'] for r in per_image),'imgsz':args.imgsz,'iou_nms':0.45,'detection_metrics_at_conf_0_001':metrics,'mean_absolute_count_error':count_mae,'exact_count_images':sum(r['absolute_count_error']==0 for r in per_image),'per_image_counts':per_image,'annotated_examples':example_records,'limitations':'Metrics are only as reliable as source annotations. Treat as exploratory when source test history is unknown or annotations are not manually verified; no verified negatives means false positives on empty scenes are not measured.'}
    destination=out/'evaluation.json';destination.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Validation threshold sweep:');[print(f"  conf={r['confidence']:.2f} P={r['precision_iou50']:.3f} R={r['recall_iou50']:.3f} F1={r['f1_iou50']:.3f} count_MAE={r['mean_absolute_count_error']:.3f}") for r in sweep]
    print(f'Selected confidence: {selected:.2f}')
    print('Test detection metrics:',json.dumps(metrics,indent=2));print(f'Test count MAE: {count_mae:.4f}; exact counts: {summary["exact_count_images"]}/{len(test_images)}')
    print(f'Evaluation report: {destination}')
    return 0

if __name__=='__main__':raise SystemExit(main())
