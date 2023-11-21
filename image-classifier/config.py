from openvino.runtime import Core

# OpenVino Global Variables
# Model Configurations
ie = Core()

model_ac = ie.read_model(model="model/v3-small_224_1.0_float.xml")
compiled_model_ic = ie.compile_model(model=model_ac, device_name="CPU")

output_layer_ic = compiled_model_ic.output(0)

