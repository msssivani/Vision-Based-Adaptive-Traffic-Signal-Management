# Pattern Recognition-Based Intelligent Adaptive Traffic Signal Control Using Computer Vision

## 1. Introduction

Traffic congestion is a major problem in urban areas. Traditional traffic signals use fixed timings without considering the actual traffic conditions on each road.

This project implements an intelligent adaptive traffic signal control approach using **Pattern Recognition and Computer Vision**. It analyzes traffic videos, identifies vehicles, extracts traffic-related features, recognizes congestion patterns, and allocates signal timings according to traffic demand.

## 2. Application of Pattern Recognition

Pattern Recognition plays an important role in identifying traffic conditions from visual information.

The proposed system follows these stages:

1. **Data Acquisition:** Traffic videos are used as input for analysis.
2. **Preprocessing:** OpenCV processes video frames and prepares them for vehicle detection.
3. **Vehicle Detection:** YOLO identifies vehicles such as cars, buses, trucks, and motorcycles.
4. **Vehicle Tracking:** Detected vehicles are tracked across consecutive frames using tracking IDs.
5. **Feature Extraction:** Important traffic features are obtained, including vehicle count, waiting time, queue length, and traffic density.
6. **Pattern Recognition:** Extracted features are analyzed to identify traffic conditions.
7. **Classification:** Traffic conditions are categorized into Low, Moderate, and High congestion.
8. **Decision Making:** Recognized traffic conditions are used to calculate road priority and allocate green signal duration.

## 3. Vehicle Detection and Tracking

Computer Vision techniques are used to analyze traffic video frames.

YOLO detects vehicles and generates bounding boxes around them. Object tracking maintains vehicle identities across consecutive frames.

This helps estimate the movement of vehicles and collect traffic information for each road.

## 4. Traffic Feature Extraction

The system extracts the following features for the four approaches: North, South, East, and West.

- **Vehicle Count:** Number of detected vehicles on each road.
- **Waiting Time:** Estimated time vehicles spend waiting.
- **Queue Length:** Measurement of vehicles accumulated in the waiting region.
- **Traffic Density:** Measurement representing the concentration of vehicles.

These features form the input for traffic-pattern analysis and signal allocation.

## 5. Traffic Pattern Classification

Based on the extracted traffic features, traffic conditions are identified as:

- **Low Congestion:** Few vehicles and shorter queues.
- **Moderate Congestion:** Increasing vehicle count and queue length.
- **High Congestion:** More vehicles, longer queues, and increased waiting time.

The recognized traffic conditions help the controller understand the traffic demand on each road.

## 6. Adaptive Traffic Signal Control

The adaptive controller calculates priority based on traffic demand and allocates green signal time accordingly.

The traffic priority can be represented as:

$$
P_i = w_c C_i + w_w W_i + w_q Q_i
$$

Where:

- $P_i$ = Traffic priority score.
- $C_i$ = Normalized vehicle count.
- $W_i$= Normalized waiting time.
- $Q_i$ = Normalized queue length.
- $w_c$, $w_w$, $w_q$ = Feature weights.

Green signal duration is calculated using:

$$
G_i=G_{min}+P_i(G_{max}-G_{min})
$$

The system considers two compatible signal phases:

- **North–South**
- **East–West**

The controller distributes signal time according to the calculated traffic demand while maintaining configured minimum and maximum green durations.

## 7. Adaptive Signal Allocation Output

The following results show an example of the calculated priority and green-time allocation for the four approaches.

| Road | Priority Score | Priority Rank | Green Duration |
|---|---:|---:|---:|
| North | 13.01 | 1 | 21.7 sec |
| West | 12.10 | 2 | 20.9 sec |
| East | 11.66 | 3 | 20.5 sec |
| South | 1.01 | 4 | 10.9 sec |

North receives the highest priority, while South receives the lowest priority based on the supplied traffic measurements.

## 8. Performance Evaluation

The adaptive signal control approach is compared with traditional fixed-time signal control using:

- Average waiting time.
- Total cumulative delay.
- Wasted green time.
- Vehicle throughput.

### Adaptive vs. Fixed-Time Signal Control Benchmark

<img width="945" height="641" alt="WhatsApp Image 2026-11-04 at 14 40 49" src="https://github.com/user-attachments/assets/75e94735-cc07-49b5-b0b1-f5995558b17a" />


The supplied benchmark graph reports:

- **59.5% delay reduction.**
- **17.5% wasted green eliminated.**

The comparison illustrates the potential benefits of allocating signal time according to traffic demand rather than using fixed durations.

*Note: These percentages are the values displayed in the supplied benchmark graph. They should be treated as benchmark-reported results until the underlying measurements and experimental procedure are independently validated.*

## 9. Conclusion

This project demonstrates the application of Pattern Recognition and Computer Vision in intelligent traffic management.

Vehicle detection and tracking provide important traffic information, while feature extraction helps identify traffic conditions. Pattern Recognition converts these extracted features into meaningful congestion categories.

The adaptive signal controller uses the recognized traffic demand to calculate road priorities and allocate green signal duration.

Thus, the proposed system demonstrates how visual traffic information can support intelligent decision-making and improve traffic signal management compared with traditional fixed-time control.
