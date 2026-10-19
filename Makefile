PY := .venv/bin/python

setup:            ## create the virtualenv and install dependencies
	python3.12 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt

export:           ## export YOLOv8n to ONNX and Core ML (FP32, FP16, INT8)
	$(PY) -m bench.export

quantize:         ## ONNX Runtime INT8: dynamic, and static calibrated on coco128
	$(PY) -m bench.quantize_onnx dynamic
	$(PY) -m bench.quantize_onnx static
	$(PY) -m bench.quantize_onnx static --head-fp32

int8:             ## calibration comparison, layer ranking, mixed-precision sweep, exports (~1 h)
	$(PY) -m bench.int8_sensitivity --stage calib
	$(PY) -m bench.int8_sensitivity --stage loo --calib percentile --per-channel
	$(PY) -m bench.int8_sensitivity --stage sweep --calib percentile --per-channel --ranking loo
	$(PY) -m bench.int8_sensitivity --export 0 --name onnx_int8_tuned --calib percentile --per-channel
	for k in 1 4 8; do $(PY) -m bench.int8_sensitivity --export $$k --name onnx_int8_mixed$$k \
		--calib percentile --per-channel || exit 1; done

W8A8 := coreml_w8a8 coreml_w8a8_head_fp16 coreml_w8a8_mixed1 coreml_w8a8_mixed4 coreml_w8a8_mixed8
COMPRESS := coreml_fp16_ios18 coreml_pal8 coreml_pal6 coreml_pal4 coreml_pal2 coreml_prune50 coreml_prune75 \
	coreml_prune50_pal4 coreml_pal4_tensor coreml_pal2_tensor

w8a8:             ## full INT8 Core ML: build, Core ML layer ranking, mixed exports, compute plans, mAP (slow)
	$(PY) -m bench.coreml_quant --build
	$(PY) -m bench.coreml_quant --stage loo
	for k in 1 4 8; do $(PY) -m bench.coreml_quant --export $$k --ranking loo || exit 1; done
	$(PY) -m bench.coreml_quant --plan all
	$(PY) -m bench.accuracy --data data/coco-val2017.yaml --no-baseline --only $(W8A8)

compress:         ## Core ML palettization and pruning -> results/coreml_compress.csv, then mAP
	$(PY) -m bench.coreml_compress --only $(COMPRESS)
	$(PY) -m bench.accuracy --data data/coco-val2017.yaml --no-baseline --only $(COMPRESS)

generalize:       ## INT8 recipe on YOLOv8s, YOLOv8m, YOLO11n, RT-DETR-L and YOLOv8n at 320/480 -> results/generalize.csv
	$(PY) -m bench.generalize --bases yolov8s.pt yolo11n.pt --sweep-sizes 320 480
	$(PY) -m bench.generalize --bases yolov8m.pt --sweep-sizes --calib-stride 8
	$(PY) -m bench.generalize --bases rtdetr-l.pt --sweep-sizes --calib-stride 4

qat:              ## quantization-aware training on 5,000 COCO train images (~800 MB download; slow) -> results/qat.csv
	$(PY) -m bench.coco_train_subset
	$(PY) -m bench.qat --eval data/coco-val2017.yaml --eval-ptq --eval-fp32
	$(PY) -m bench.accuracy --data data/coco-val2017.yaml --no-baseline --only onnx_int8_qat

UPSTREAM := onnx_int8_ultralytics onnx_int8_ultralytics_pc onnx_int8_ultralytics_pc_dfl

upstream:         ## Ultralytics' own ONNX INT8 export, as shipped and with two changes, then mAP
	$(PY) -m bench.upstream_onnx_int8
	$(PY) -m bench.accuracy --data data/coco-val2017.yaml --no-baseline --only $(UPSTREAM)

latency:          ## one latency pass (300 timed runs per configuration) -> results/latency.csv; run 3 times
	$(PY) -m bench.latency --runs 300

accuracy:         ## mAP on coco128 for every model -> results/accuracy.csv
	$(PY) -m bench.accuracy

coco-val:         ## download COCO val2017 only (~1 GB) for final accuracy numbers
	$(PY) -m bench.coco_val

accuracy-full:    ## mAP on COCO val2017 (5,000 images) -> results/accuracy.csv
	$(PY) -m bench.accuracy --data data/coco-val2017.yaml

ios-models:       ## copy the Core ML exports and a sample photo into the iOS app (run export first)
	mkdir -p ios/EdgeDetect/MLModels
	for p in fp32 fp16 int8; do P=$$(echo $$p | tr a-z A-Z); rm -rf ios/EdgeDetect/MLModels/YOLOv8n_$$P.mlpackage; \
		cp -R models/coreml_$$p/yolov8n.mlpackage ios/EdgeDetect/MLModels/YOLOv8n_$$P.mlpackage || exit 1; done
	cp $$($(PY) -c "import ultralytics, pathlib; print(pathlib.Path(ultralytics.__file__).parent / 'assets' / 'bus.jpg')") ios/EdgeDetect/MLModels/bus.jpg

plot:             ## latency vs. accuracy chart -> results/tradeoff.png
	$(PY) -m bench.plot

.PHONY: setup export quantize int8 w8a8 compress generalize qat upstream latency accuracy coco-val accuracy-full ios-models plot
