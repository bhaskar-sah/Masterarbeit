import mujoco
import mujoco.viewer
import os

# --- THIS IS THE CHANGE ---
# Point directly to the robot's scene file.
XML_PATH = "franka/scene.xml" 
# -------------------------

if not os.path.exists(XML_PATH):
    print(f"Error: Cannot find '{XML_PATH}'")
    print("Make sure you are in the 'franka_rl' directory and have")
    print("copied the robot files into the 'franka' subfolder.")
else:
    try:
        # 1. Load the model directly from the scene.xml
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        
        print(f"--- Successfully loaded '{XML_PATH}'! ---")
        print(f"Model has {model.nbody} bodies and {model.njnt} joints.")

        # 2. Launch the interactive viewer
        print("\nLaunching viewer... Close the window to exit.")
        mujoco.viewer.launch(model, data)

    except Exception as e:
        print(f"Error loading model: {e}")