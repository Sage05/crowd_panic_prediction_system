# Computer Vision Test Results

## Objective

Evaluate the performance of the computer vision pipeline for real-time crowd monitoring.

---

## Models Used

- YOLOv8 – Person Detection
- CSRNet – Crowd Density Estimation

---

## Dataset

- ShanghaiTech Crowd Counting Dataset (Part B)

---

## Evaluation Summary

### Person Detection

Model:
- YOLOv8

Status:
- Successfully detects individuals in crowded scenes.

Observation:
- Accurate for medium and high-density crowds.
- Occasional missed detections under severe occlusion.

---

### Density Estimation

Model:
- CSRNet

Output:
- 32 × 60 density map

Observation:
- Produces smooth density maps suitable for ConvLSTM forecasting.
- Density integrates to approximate crowd count.

---

## Integration Status

✓ Video stream acquisition

✓ Person detection

✓ Density estimation

✓ Density map generation

✓ ConvLSTM input generation

✓ Backend integration

---

## Known Limitations

- Tested primarily on recorded video.
- Performance depends on camera angle and crowd density.
- Limited evaluation under adverse lighting and weather conditions.

---

## Conclusion

The computer vision pipeline successfully generates density maps and crowd counts required for downstream forecasting and anomaly detection.
