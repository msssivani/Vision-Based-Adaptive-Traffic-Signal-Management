import cv2
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "traffic_video_10min.mp4"
ROI_PATH = PROJECT_ROOT / "configs" / "road_regions.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "module2"

MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40

# COCO vehicle class IDs:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

ROAD_NAMES = ["north", "east", "south", "west"]


# --------------------------------------------------
# TRAFFIC LIGHT EXCLUSION ZONES
# --------------------------------------------------

# Vehicles whose center point falls inside one of these
# zones are ignored. These zones are not drawn on the video.
EXCLUSION_ZONES = {
    "east_traffic_light": [
        (1080, 230),
        (1150, 230),
        (1150, 300),
        (1080, 300)
    ]
}


# --------------------------------------------------
# LOAD ROAD ROI CONFIGURATION
# --------------------------------------------------

def load_rois():

    if not ROI_PATH.exists():
        raise FileNotFoundError(f"ROI configuration not found: {ROI_PATH}")

    with open(ROI_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    roads = config["roads"]

    for road in ROAD_NAMES:

        if road not in roads:
            raise ValueError(f"Missing road ROI: {road}")

        if "road_roi" not in roads[road]:
            raise ValueError(f"Missing road_roi for {road}")

        if len(roads[road]["road_roi"]) < 3:
            raise ValueError(f"Invalid road_roi polygon for {road}")

        if "waiting_zone" not in roads[road]:
            raise ValueError(
                f"Missing waiting_zone for {road} in {ROI_PATH}"
            )

        if len(roads[road]["waiting_zone"]) < 3:
            raise ValueError(
                f"Invalid waiting_zone polygon for {road}"
            )

    return roads, config


# --------------------------------------------------
# CHECK TRAFFIC LIGHT EXCLUSION ZONES
# --------------------------------------------------

def is_excluded(x, y):

    for points in EXCLUSION_ZONES.values():

        polygon = np.array(points, dtype=np.int32)

        result = cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        )

        if result >= 0:
            return True

    return False


# --------------------------------------------------
# ASSIGN VEHICLE TO A ROAD
# --------------------------------------------------

def get_road(x, y, roads):

    for road_name in ROAD_NAMES:

        polygon = np.array(
            roads[road_name]["road_roi"],
            dtype=np.int32
        )

        result = cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False
        )

        if result >= 0:
            return road_name

    return None


# --------------------------------------------------
# CHECK WHETHER VEHICLE IS IN ITS ROAD'S WAITING ZONE
# --------------------------------------------------

def is_in_waiting_zone(x, y, road, roads):

    polygon = np.array(
        roads[road]["waiting_zone"],
        dtype=np.int32
    )

    result = cv2.pointPolygonTest(
        polygon,
        (float(x), float(y)),
        False
    )

    return result >= 0


# --------------------------------------------------
# DRAW ROAD ROIS
# --------------------------------------------------

def draw_rois(frame, roads):

    for road_name, region in roads.items():

        polygon = np.array(
            region["road_roi"],
            dtype=np.int32
        )

        cv2.polylines(
            frame,
            [polygon],
            True,
            (0, 255, 0),
            2
        )

        x, y = polygon[0]

        cv2.putText(
            frame,
            road_name.upper(),
            (int(x), max(25, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return frame


# --------------------------------------------------
# MAIN VEHICLE DETECTION AND TRACKING
# --------------------------------------------------

def process_video(video_path):

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roads, roi_config = load_rois()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open traffic video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS.")

    # Verify that ROI coordinates were defined for this video resolution.
    configured_width = roi_config.get("frame_width")
    configured_height = roi_config.get("frame_height")

    if (
        configured_width is not None
        and configured_height is not None
        and (
            configured_width != width
            or configured_height != height
        )
    ):
        cap.release()
        raise ValueError(
            "Video resolution does not match ROI configuration. "
            f"Video: {width}x{height}; "
            f"ROI config: {configured_width}x{configured_height}."
        )

    # Verify exclusion-zone coordinates are inside the frame.
    for zone_name, points in EXCLUSION_ZONES.items():

        for x, y in points:

            if not (0 <= x < width and 0 <= y < height):
                cap.release()
                raise ValueError(
                    f"Exclusion zone '{zone_name}' has coordinates "
                    "outside the video frame. Update EXCLUSION_ZONES "
                    "for this video's resolution."
                )

    output_video = OUTPUT_DIR / "vehicle_tracking_output.mp4"
    output_csv = OUTPUT_DIR / "vehicle_counts.csv"
    observations_csv = OUTPUT_DIR / "track_observations.csv"
    summary_json = OUTPUT_DIR / "tracking_summary.json"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    print("Loading YOLO11 model...")
    model = YOLO(MODEL_NAME)

    # Cumulative unique IDs observed on each road.
    road_track_ids = defaultdict(set)

    frame_index = 0
    observation_count = 0

    print("\nStarting vehicle detection and tracking...\n")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Resolution: {width}x{height}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Tracker: ByteTrack")
    print("Waiting-zone observations: enabled")
    print("Traffic-light exclusion zones: enabled (not displayed)")

    try:

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file, open(
            observations_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as observations_file:

            csv_writer = csv.writer(csv_file)
            observations_writer = csv.writer(observations_file)

            # One row per processed frame.
            csv_writer.writerow([
                "frame_index",
                "timestamp_seconds",
                "north_count",
                "east_count",
                "south_count",
                "west_count",
                "north_unique",
                "east_unique",
                "south_unique",
                "west_unique"
            ])

            # One row per tracked vehicle detection per frame.
            observations_writer.writerow([
                "frame_index",
                "timestamp_seconds",
                "track_id",
                "road",
                "vehicle_class",
                "center_x",
                "center_y",
                "in_waiting_zone"
            ])

            FRAME_SKIP = 3

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                if frame_index % FRAME_SKIP != 0:
                    frame_index += 1
                    continue

                timestamp = frame_index / fps

                # YOLO detection + persistent ByteTrack IDs.
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False
                )

                result = results[0]

                current_counts = {
                    road: 0 for road in ROAD_NAMES
                }

                annotated = draw_rois(frame.copy(), roads)

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    boxes = result.boxes

                    xyxy = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    confidences = boxes.conf.cpu().numpy()

                    # Only tracked detections with an assigned ID
                    # can be written to track_observations.csv.
                    if boxes.id is not None:

                        track_ids = boxes.id.cpu().numpy().astype(int)

                        for box, class_id, track_id, confidence in zip(
                            xyxy,
                            class_ids,
                            track_ids,
                            confidences
                        ):

                            x1, y1, x2, y2 = map(int, box)

                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            # Ignore detections inside exclusion zones.
                            if is_excluded(center_x, center_y):
                                continue

                            # Assign detection to one of the four roads.
                            road = get_road(
                                center_x,
                                center_y,
                                roads
                            )

                            if road is None:
                                continue

                            # Count this vehicle in the current frame.
                            current_counts[road] += 1

                            # Store its ID in the cumulative road set.
                            road_track_ids[road].add(int(track_id))

                            vehicle_name = CLASS_NAMES.get(
                                class_id,
                                "Vehicle"
                            )

                            # Check whether the vehicle center is inside
                            # the waiting zone belonging to this road.
                            waiting_status = is_in_waiting_zone(
                                center_x,
                                center_y,
                                road,
                                roads
                            )

                            # Save one observation for this vehicle/frame.
                            observations_writer.writerow([
                                frame_index,
                                round(timestamp, 3),
                                int(track_id),
                                road.upper(),
                                vehicle_name,
                                center_x,
                                center_y,
                                int(waiting_status)
                            ])

                            observation_count += 1

                            label = (
                                f"{vehicle_name} "
                                f"ID:{track_id} "
                                f"Conf:{confidence:.2f} "
                                f"{road.upper()}"
                            )

                            # Draw bounding box.
                            cv2.rectangle(
                                annotated,
                                (x1, y1),
                                (x2, y2),
                                (255, 0, 0),
                                2
                            )

                            # Draw vehicle label.
                            cv2.putText(
                                annotated,
                                label,
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2
                            )

                # Display frame-wise vehicle counts.
                y_position = 30

                for road in ROAD_NAMES:

                    text = (
                        f"{road.upper()}: "
                        f"{current_counts[road]} vehicles"
                    )

                    cv2.putText(
                        annotated,
                        text,
                        (15, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                    y_position += 30

                # Save annotated frame.
                writer.write(annotated)

                # Save frame-wise and cumulative unique counts.
                csv_writer.writerow([
                    frame_index,
                    round(timestamp, 3),
                    *[
                        current_counts[road]
                        for road in ROAD_NAMES
                    ],
                    *[
                        len(road_track_ids[road])
                        for road in ROAD_NAMES
                    ]
                ])

                frame_index += 1

                if frame_index % 100 == 0:
                    print(f"Processed {frame_index} frames...")

    finally:
        cap.release()
        writer.release()

    # --------------------------------------------------
    # SAVE TRACKING SUMMARY
    # --------------------------------------------------

    summary = {
        "video_filename": video_path.name,
        "model": MODEL_NAME,
        "tracker": "ByteTrack",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "total_frames_processed": frame_index,
        "duration_seconds": (
            round((frame_index - 1) / fps, 3)
            if frame_index > 0 else 0
        ),
        "vehicle_classes": CLASS_NAMES,
        "traffic_light_exclusion_zones": EXCLUSION_ZONES,
        "track_observations": {
            "filename": observations_csv.name,
            "total_observations": observation_count,
            "fields": [
                "frame_index",
                "timestamp_seconds",
                "track_id",
                "road",
                "vehicle_class",
                "center_x",
                "center_y",
                "in_waiting_zone"
            ]
        },
        "unique_vehicle_ids_per_road": {
            road.upper(): len(road_track_ids[road])
            for road in ROAD_NAMES
        }
    }

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # --------------------------------------------------
    # COMPLETION MESSAGE
    # --------------------------------------------------

    print("\n================================")
    print("VEHICLE TRACKING COMPLETED")
    print("================================")

    print(f"Frames processed: {frame_index}")
    print(f"Tracking observations: {observation_count}")

    for road in ROAD_NAMES:
        print(
            f"{road.upper()}: "
            f"{len(road_track_ids[road])} unique tracked IDs"
        )

    print(f"\nAnnotated video: {output_video}")
    print(f"Vehicle counts: {output_csv}")
    print(f"Tracking observations: {observations_csv}")
    print(f"Tracking summary: {summary_json}")


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "YOLO vehicle detection, ByteTrack tracking, "
            "and waiting-zone observation export"
        )
    )

    parser.add_argument(
        "--video",
        default=str(VIDEO_PATH),
        help="Path to traffic video"
    )

    args = parser.parse_args()

    process_video(Path(args.video))


if __name__ == "__main__":
    main()
