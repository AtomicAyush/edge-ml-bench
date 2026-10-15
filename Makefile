PY := .venv/bin/python

setup:            ## create the virtualenv and install dependencies
	python3.12 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt

export:           ## export YOLOv8n to ONNX and Core ML (FP32, FP16, INT8)
	$(PY) -m bench.export

quantize:         ## ONNX Runtime INT8: dynamic, and static calibrated on coco128
	$(PY) -m bench.quantize_onnx dynamic
	$(PY) -m bench.quantize_onnx static
	$(PY) -m bench.quantize_onnx static --head-fp32

latency:          ## latency/throughput for every model on this machine -> results/latency.csv
	$(PY) -m bench.latency

accuracy:         ## mAP on coco128 for every model -> results/accuracy.csv
	$(PY) -m bench.accuracy

coco-val:         ## download COCO val2017 only (~1 GB) for final accuracy numbers
	$(PY) -m bench.coco_val

accuracy-full:    ## mAP on COCO val2017 (5,000 images) -> results/accuracy.csv
	$(PY) -m bench.accuracy --data data/coco-val2017.yaml

plot:             ## latency vs. accuracy chart -> results/tradeoff.png
	$(PY) -m bench.plot

.PHONY: setup export quantize latency accuracy coco-val accuracy-full plot
