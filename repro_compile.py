import sys
import migraphx
m = migraphx.parse_onnx(sys.argv[1] if len(sys.argv) > 1 else "rfdetr-vr-v1-b2.onnx")
m.compile(migraphx.get_target("gpu"))  # aborts here with MLIR on
print("COMPILE_OK")
