# 2025-01-16T10:03:46.560944700
import vitis

client = vitis.create_client()
client.set_workspace(path="MLP")

vitis.dispose()

