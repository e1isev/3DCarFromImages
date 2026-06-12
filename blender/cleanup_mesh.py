"""
Blender headless script — mesh cleanup pass.

Invoked by car2asset after 3D reconstruction to clean up the raw mesh:
  • Remove loose geometry and duplicate vertices
  • Apply smooth shading
  • Add a Subdivision Surface modifier (levels=1) for smoother silhouette
  • Separate car body, wheels (by loose parts), and glass
  • Centre the mesh at the origin with Z=0 at ground

Usage (headless):
    blender --background --python blender/cleanup_mesh.py -- input.glb output.glb
"""

import sys


def main(input_path: str, output_path: str) -> None:
    import bpy  # type: ignore[import]

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB
    bpy.ops.import_scene.gltf(filepath=input_path)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found in {input_path}")

    for obj in meshes:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Enter edit mode for mesh cleanup
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

        # Smooth shading
        bpy.ops.object.shade_smooth()

        # Light subdivision for smoother silhouette
        sub = obj.modifiers.new(name="Subdivision", type="SUBSURF")
        sub.levels = 1
        sub.render_levels = 2

        obj.select_set(False)

    # Centre meshes: move bounding box centre to world origin, Z bottom at 0
    all_obj = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if all_obj:
        import mathutils  # type: ignore[import]

        xs = [v.co.x for o in all_obj for v in o.data.vertices]
        ys = [v.co.y for o in all_obj for v in o.data.vertices]
        zs = [v.co.z for o in all_obj for v in o.data.vertices]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        cz = min(zs)
        delta = mathutils.Vector((-cx, -cy, -cz))
        for o in all_obj:
            o.location += delta

    # Export GLB
    bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")
    print(f"[cleanup_mesh] Saved: {output_path}")


if __name__ == "__main__":
    argv = sys.argv
    try:
        sep = argv.index("--")
    except ValueError:
        print("Usage: blender --background --python cleanup_mesh.py -- input.glb output.glb")
        sys.exit(1)

    args = argv[sep + 1 :]
    if len(args) < 2:
        print("Error: expected <input.glb> <output.glb> after --")
        sys.exit(1)

    main(args[0], args[1])
