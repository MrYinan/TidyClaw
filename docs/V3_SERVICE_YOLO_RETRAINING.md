# V3 Service YOLO Retraining

V3 keeps the same online/offline boundary as V2, but expands the service
ontology with support structures and ignored kitchen objects so shelf/tabletop
objects are not mistaken for floor pickup targets.

Default route:

```powershell
python scripts\train_service_yolo_v3.py --collect-count 1200 --epochs 120 --imgsz 960 --batch 4 --device 0
```

This command uses:

- base model: `yolo11n.pt`
- ontology: `configs/service_task_ontology_v3.json`
- dataset root: `datasets/ai2thor_service_yolo_v3`
- offline labels: `/eval/state instance_detections2D`

The intended training line is:

```text
natural-image pretrained YOLO11n
  -> AI2-THOR RGB frames with offline instance labels
  -> service-task fine-tuned YOLO weights
  -> RGB-only online perception through yolo_service
```

After validation, deploy the trained model:

```powershell
copy runs\detect\service_v3_general_YYYYMMDD_HHMM\weights\best.pt skills\perceive-scene-yolo\weights\best.pt
```

Then restart the persistent YOLO service with the matching ontology:

```powershell
python scripts\yolo_service.py --weights skills\perceive-scene-yolo\weights\best.pt --ontology configs\service_task_ontology_v3.json
```

For a quick pipeline check without a long training run:

```powershell
python scripts\train_service_yolo_v3.py --collect-count 20 --skip-train
```

For a train-only continuation after collecting enough images:

```powershell
python scripts\train_service_yolo_v3.py --skip-collect --epochs 120 --imgsz 960 --batch 4 --device 0
```
