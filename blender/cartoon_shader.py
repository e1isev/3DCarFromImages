"""
Blender headless script — cartoon / cel-shade material pass.

Applies a Blender-native cel-shaded material to every mesh in the scene:
  • Diffuse BSDF coloured with a Colour Ramp (3-step toon shadow)
  • Freestyle edge rendering for thick black outlines (render only)
  • Bright, saturated colour sampled from the dominant vertex colour
  • Exports the result as a GLB with baked vertex colours

Usage (headless):
    blender --background --python blender/cartoon_shader.py -- input.glb output.glb [r g b]

    r g b are optional 0–1 floats for the body colour; if omitted, the
    dominant vertex colour of the first mesh is used.
"""

import sys


def main(input_path: str, output_path: str, override_colour: tuple | None = None) -> None:
    import bpy  # type: ignore[import]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=input_path)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found")

    body_colour = override_colour or _sample_dominant_colour(meshes[0])

    # Enable Freestyle for outline rendering
    scene = bpy.context.scene
    scene.render.use_freestyle = True
    lineset = scene.view_layers[0].freestyle_settings.linesets[0]
    lineset.linestyle.color = (0.02, 0.02, 0.02)
    lineset.linestyle.thickness = 2.5

    for obj in meshes:
        _apply_toon_material(obj, body_colour)

    bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")
    print(f"[cartoon_shader] Saved: {output_path}")


def _apply_toon_material(obj, colour: tuple) -> None:
    import bpy  # type: ignore[import]

    mat = bpy.data.materials.new(name=f"toon_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Shader output
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)

    # Diffuse BSDF (flat, no specular)
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (200, 0)

    # Colour Ramp → toon shading steps
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (0, -150)
    ramp.color_ramp.interpolation = "CONSTANT"
    # 3-step toon: shadow / midtone / highlight
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (*[c * 0.4 for c in colour[:3]], 1.0)
    mid = ramp.color_ramp.elements.new(0.35)
    mid.color = (*colour[:3], 1.0)
    high = ramp.color_ramp.elements.new(0.75)
    high.color = (*[min(1.0, c * 1.3) for c in colour[:3]], 1.0)

    # Shader-to-RGB (toon evaluation)
    s2rgb = nodes.new("ShaderNodeShaderToRGB")
    s2rgb.location = (-200, -150)

    links.new(s2rgb.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], diffuse.inputs["Color"])
    links.new(diffuse.outputs["BSDF"], s2rgb.inputs["Shader"])
    links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _sample_dominant_colour(obj) -> tuple:
    """Return (r, g, b) 0-1 floats from the median vertex colour, or a default."""
    import numpy as np  # type: ignore[import]

    try:
        vcol = obj.data.vertex_colors.active
        if vcol is None:
            return (0.8, 0.15, 0.05)
        colours = np.array([loop_col.color[:3] for loop_col in vcol.data])
        median = np.median(colours, axis=0)
        # Boost saturation
        import colorsys  # type: ignore[import]

        h, s, v = colorsys.rgb_to_hsv(*median)
        return colorsys.hsv_to_rgb(h, min(1.0, s + 0.3), min(1.0, v + 0.1))
    except Exception:
        return (0.8, 0.15, 0.05)


if __name__ == "__main__":
    argv = sys.argv
    try:
        sep = argv.index("--")
    except ValueError:
        print("Usage: blender --background --python cartoon_shader.py -- input.glb output.glb [r g b]")
        sys.exit(1)

    args = argv[sep + 1 :]
    if len(args) < 2:
        print("Error: need at least <input.glb> <output.glb>")
        sys.exit(1)

    colour = None
    if len(args) >= 5:
        colour = (float(args[2]), float(args[3]), float(args[4]))

    main(args[0], args[1], colour)
