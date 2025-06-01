import mujoco
import numpy as np
import matplotlib.pyplot as plt


# Load the model
model = mujoco.MjModel.from_xml_path("envs/mujoco/model/walk/h1_walk.xml")
data = mujoco.MjData(model)

# Simulate a few steps
for _ in range(10):
    mujoco.mj_step(model, data)

# Set up offscreen rendering
width, height = 640, 480
viewport = mujoco.MjrRect(0, 0, width, height)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

# Create buffers for RGB and segmentation
rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
segmentation_buffer = np.zeros((height, width), dtype=np.uint8)

# Set up visualization options
option = mujoco.MjvOption()
mujoco.mjv_defaultOption(option)

# Set up scene and camera
scene = mujoco.MjvScene(model, maxgeom=1000)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)

# Update scene
mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

# Render the scene
mujoco.mjr_render(viewport, scene, context)

# Read the RGB image
mujoco.mjr_readPixels(rgb_buffer, None, viewport, context)

# Read the segmentation buffer
mujoco.mjr_readPixels(None, segmentation_buffer, viewport, context)

# Identify H1 robot geom IDs
h1_geom_ids = []
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and ("h1" in name.lower() or "left_hand" in name.lower() or "right_hand" in name.lower()):
        h1_geom_ids.append(i)

# Create binary mask for H1 robot
h1_mask = np.isin(segmentation_buffer, h1_geom_ids)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rgb_buffer)
plt.title("RGB Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(segmentation_buffer, cmap='nipy_spectral')
plt.title("Segmentation (Geom IDs)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(h1_mask, cmap='gray')
plt.title("H1 Robot Mask")
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the mask if needed
plt.imsave("h1_robot_mask.png", h1_mask.astype(np.uint8) * 255, cmap='gray')
