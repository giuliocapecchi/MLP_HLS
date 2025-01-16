# 2025-01-16T11:01:37.074037800
import vitis

client = vitis.create_client()
client.set_workspace(path="MLP")

comp = client.get_component(name="hls_component")
comp.run(operation="C_SIMULATION")

comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

comp.run(operation="CO_SIMULATION")

