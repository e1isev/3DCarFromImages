"""
Blender headless script — export GLB → FBX for Unreal Engine / Unity.

Settings tuned for Unreal Engine 5:
  • Forward axis: -Y → X  (UE5 convention)
  • Apply unit scale
  • Bake armature to root bone
  • Export textures embedded

Usage (headless):
    blender --background --python blender/export_fbx.py -- input.glb output.fbx
"""

import sys


def main(input_path: str, output_path: str) -> None:
    import bpy  # type: ignore[import]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=input_path)

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        bake_space_transform=True,
        axis_forward="-Z",
        axis_up="Y",
        mesh_smooth_type="FACE",
        use_mesh_modifiers=True,
        use_armature_deform_only=False,
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=True,
    )
    print(f"[export_fbx] Saved: {output_path}")


if __name__ == "__main__":
    argv = sys.argv
    try:
        sep = argv.index("--")
    except ValueError:
        print("Usage: blender --background --python export_fbx.py -- input.glb output.fbx")
        sys.exit(1)

    args = argv[sep + 1 :]
    if len(args) < 2:
        print("Error: need <input.glb> <output.fbx>")
        sys.exit(1)

    main(args[0], args[1])
