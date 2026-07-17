from ultralytics import YOLO

def main():
    model = YOLO("runs/road_hazard_v2/weights/best.pt")

    metrics = model.val(
        data="datasets/merged/data.yaml",
        split="test",
        plots=True,
        save_json=True
    )

    print(metrics)

if __name__ == "__main__":
    main()