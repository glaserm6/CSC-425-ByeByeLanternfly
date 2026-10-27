# Bye-Bye Lanternfly

A small Flask and YOLOv8 application that finds and counts **visible adult spotted lanternflies in one photograph**. A photo count describes detections in that image; it does not establish the population or infestation level of an area. The project supports environmental monitoring by making photo-based observations easier to record and compare.

## Dataset and what the model recognizes

Training data came from Roboflow Universe, [Spotted Lanternfly Annotated Set All Boxes No Null](https://universe.roboflow.com/vasudevs-workspace-tjnq3/spotted-lanternfly-annotated-set-all-boxes-no-null-nbpge-70ujp), export **version 7**, object detection, **CC BY 4.0**. Attribution: Vasudevs Workspace, Roboflow Universe. The source export labels adults, early and late nymphs, and eggs. The model uses only the exact `Adult Spotted Lanternfly` class, mapped to one class named `lanternfly`; it does not claim to detect nymphs or eggs. See [source and screening notes](docs/roboflow_sources.md).

The final local training candidate contains 1,825/438/276 train/validation/test images and 2,969/788/765 adult boxes, respectively. Empty-adult-label images were excluded because they may contain unannotated adults, nymphs, or eggs; the dataset has **no verified empty-scene negatives**. Annotations are machine-screened, not individually verified. A small six-image visual sample was reviewed; the boxes generally aligned with visible adults, and two clearly off-scene graphics were excluded from training. This sample is not enough to certify the remaining annotations. Exact duplicate checks found no cross-split exact overlap; near-duplicate screening clustered related copies and retained one per cluster. The source split history is unknown, so test metrics below are exploratory, not an independent generalization claim.

## Install (Windows PowerShell)

Python 3.11 or 3.12 is recommended for a straightforward PyTorch setup. Install the PyTorch build appropriate for the computer first if needed. CPU inference and training are supported.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dataset, virtual environment, run logs, and checkpoints are local and ignored by Git. To recreate the local dataset, download the Roboflow version 7 COCO object-detection ZIP, then run the importer and screening/build steps described in `docs/roboflow_sources.md`. The ZIP inspected for this run had 3,449 images and CC BY 4.0 metadata; keep its original license and attribution with any redistribution.

```powershell
python scripts/import_roboflow_coco.py --zip "C:\path\to\dataset.v7i.coco.zip" --output data\external\roboflow\vasudevs_v7_review --version spotted-lanternfly-annotated-set-all-boxes-no-null-nbpge-70ujp/7 --source-url https://universe.roboflow.com/vasudevs-workspace-tjnq3/spotted-lanternfly-annotated-set-all-boxes-no-null-nbpge-70ujp --license "CC BY 4.0" --attribution "Vasudevs Workspace; Spotted Lanternfly Annotated Set All Boxes No Null, Roboflow Universe; CC BY 4.0" --adult-class "Adult Spotted Lanternfly"
python scripts/screen_roboflow_adults.py --review-csv data\external\roboflow\vasudevs_v7_review\review_manifest.csv --output data\external\roboflow\vasudevs_v7_review\machine_screened_manifest.csv
python scripts/build_roboflow_candidate.py --review-csv data\external\roboflow\vasudevs_v7_review\machine_screened_manifest.csv --name roboflow_v7_adult_reviewed_v1 --source-adult-screening --test-history unknown
python src/validate_dataset.py --data data\candidates\roboflow_v7_adult_reviewed_v1\dataset.yaml --require-test
```

The reviewed local candidate excludes two visually identified graphics (a meme and a wanted-poster image) in addition to the automated screening exclusions. `data.yaml` points at that ignored local candidate.

## Train and evaluate

The installed model is the best checkpoint from a completed **5-epoch CPU fine-tune** of pretrained `yolov8n.pt`, at 416-pixel training image size, batch 8, seed 42, deterministic mode, CPU workers 0. It took about 31 minutes on an Intel i7-13700H. A prior 100-epoch/640-pixel run stopped after 3 epochs. Its saved logs do not record a termination reason; the old CPU run took roughly 9 minutes per epoch, making the requested duration impractical on this host. The new run completed all 5 requested epochs and saved `best.pt`, `last.pt`, `results.csv`, and a training manifest under `runs/candidates/lanternfly_yolov8/adult_finetune_cpu_416_5e/`. It started from official generic YOLOv8n pretrained weights, not random initialization, and did not overwrite the prior checkpoint.

Reproduce with configurable epochs and image size:

```powershell
python train.py --data data.yaml --model yolov8n.pt --epochs 5 --imgsz 416 --batch 8 --patience 5 --device cpu --name adult_finetune_cpu_416_5e
# Resume an interrupted run from its own last.pt:
python train.py --data data.yaml --resume runs\candidates\lanternfly_yolov8\<run>\weights\last.pt
```

Evaluate a candidate (validation data chooses the confidence threshold; test data is used only for final scoring):

```powershell
python evaluate.py --weights runs\candidates\lanternfly_yolov8\adult_finetune_cpu_416_5e\weights\best.pt --data data.yaml --imgsz 640 --device cpu
```

### Measured evaluation (exploratory)

The candidate and prior checkpoint were compared at **640 pixels** on the same version 7 source test split. The operating confidence threshold was selected on the 438-image validation split by maximum F1 at IoU 0.50 (ties favor precision, then higher confidence). For the new checkpoint this was 0.20. At that threshold, test-set box matching gave precision **0.741**, recall **0.450**, and mean absolute count error **1.728 insects/image**; 190/276 images had the exact count. There were 276 test images and 765 annotated adults. Standard Ultralytics test metrics (confidence 0.001 for the precision/recall curve and mAP calculation) were precision **0.617**, recall **0.480**, mAP@0.50 **0.538**, and mAP@0.50:0.95 **0.294**. The precision/recall figures differ because they use different confidence settings.

Compared with the prior checkpoint at the same evaluation image size, the new checkpoint had better validation F1 (0.723 vs 0.627), test mAP@0.50 (0.538 vs 0.380), mAP@0.50:0.95 (0.294 vs 0.192), and threshold-selected test count MAE (1.728 vs 1.844). This supports using the new checkpoint for the demonstration, but the source test split's prior use is unknown, we consulted its results while comparing checkpoints, and its labels are not individually verified. These test metrics are exploratory, not an untouched final estimate of generalization. There are no verified empty scenes, so false positives on insect-free photos are not measured. The per-image counts and TP/FP/FN at IoU 0.50 are in [reports/test_counts.csv](reports/test_counts.csv); a compact metric summary is in [reports/evaluation_summary.json](reports/evaluation_summary.json). Visual prediction examples are generated under ignored `runs/evaluation/`.

Counting remains imperfect even at moderate density: one test photo labeled with 11 adults produced 6 detections. Dense scenes are a clear failure case: a photo labeled with 106 adults produced 10 boxes; for 101 and 84 adults it retained 13 and 8. Label crowding/quality and small overlapping insects need further review and additional verified annotations. These results are not suitable for estimating an area's infestation level.

## Run the website

The selected checkpoint is installed locally at `models/best.pt`. Model files are too large for normal source Git hosting and are ignored by Git; preserve the candidate at `runs/candidates/.../weights/best.pt` or provide a separately hosted model download before cloning elsewhere. The Flask app loads it once at startup. The default confidence threshold is 0.20, selected using validation data; set `LANTERNFLY_WEIGHTS` or `LANTERNFLY_CONFIDENCE` to override.

```powershell
$env:LANTERNFLY_WEIGHTS = "models\best.pt"
$env:LANTERNFLY_CONFIDENCE = "0.20"
python app.py
```

On a phone on the same Wi-Fi, open `http://<computer-LAN-address>:5000`, take a photo or choose a JPEG/PNG/WebP (up to 12 MB), preview it, and tap **Analyze image**. The response shows the annotated image and the count of retained adult-class boxes after YOLO non-maximum suppression. A zero is a successful count; upload and inference failures return separate errors. The Flask development server is intended for local demonstration, not public deployment. On iPhone/Safari, camera access may require HTTPS; photo selection remains available over the local network.

## Checks and limitations

All 11 automated unit tests pass. The app loaded `models/best.pt`; a held-out image returned an annotated response whose count equaled its detection list, a synthetic blank image produced a successful zero, an invalid upload returned HTTP 400, and `/` returned HTTP 200 through Flask's test client. An adult-free nymph photo produced a false adult detection, illustrating that adult-only scope does not eliminate false positives. No verified outdoor negative was available for a genuine zero-insect inference check. Flask starts, but sandbox network policy blocked connecting to its loopback socket, so a live browser request was not completed. The narrow mobile layout has not been visually verified in a browser or on a phone. Manual demo check: run the server, open its LAN URL on a phone, select/take a photo, verify preview and loading state, confirm result image and number of boxes agree, then repeat with an insect-free outdoor photo and an invalid file; check no horizontal scrolling. The current local host was not deployed.

The model may miss small, occluded, overlapping, distant, or cluttered insects, and may mistake nymphs or other objects for adults. More representative, individually verified adult boxes and verified zero-adult scenes are needed, with source/session groups kept together across splits. A photograph is not a population estimate. Review the [Ultralytics license](https://www.ultralytics.com/license) before redistribution or deployment.
