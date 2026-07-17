from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="datasets/merged/data.yaml",

        epochs=100,
        imgsz=640,

        batch=16,

        device=0,
        workers=8,

        cache=True,

        optimizer="AdamW",
        lr0=0.001,

        pretrained=True,

        cos_lr=True,
        amp=True,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        project="runs",
        name="road_hazard_v2",

        patience=30,
        save=True,
        verbose=True
    )

if __name__ == "__main__":
    main()