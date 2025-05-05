Features
----
- Calculate lighting in realtime at a limited view distance while moving VoxelViewer around, then save it to disk
- Run again at greater view distance using saved data, with calculation disabled
- Enable/disable sunlight, set sunlight y level
- Set color of emissive voxels by voxel ID
- RGB lighting
- Smooth or flat lighting
- This [occlusion trick](https://www.youtube.com/watch?v=rIX-RsFFF4Y)

Limitations
----
- Realtime lighting doesn't update when you place or remove blocks
- In calculating mode, struggles to calculate lighting with VoxelViewer view distance above 64, liable to drop blocks
- Not very optimised for performance because I wasn't able to profile or even debug Godot
- In particular, lighting threads run synchronously and attempts at multiprocessing with per-block locking cause crashes
- Saves thousands of light data files rather than bundling them (might address later to improve initial load times), light data could also be compressed after bundling
- Doesn't support transparent blocks other than air
- Keeps all light data in memory (~17KB per uncompressed block) and never unloads it, which is unsuitable for infinite worlds
- Hardcoded block size of 16x16x16
- X/Y/Z coordinates must be within -16000 to 16000 otherwise block data will overlap
- Code structured haphazardly, functionality glued on to existing classes rather than making separate ones

How to use
----
- Compile Godot as you would with the Voxel Tools module (see Voxel Tools documentation), but use this repo instead of Zylann's godot_voxel repo
- Use VoxelTerrain and VoxelMesherBlocky as you would normally, configure additional properties which were added to both classes
- Change `getEmissiveColor` emissive voxel types in the C++ code if you want (I've added many by default which you probably want to remove)
- Set VoxelViewer view distance to something small like 64
- Set sunlight y level to something appropriate
- Enable `Calculate Light`, run game and move around, calculating light data
- Write code which calls `VoxelTerrain.saveLightData()` on a keypress or something to save the calculated light data
- Enable `Use Baked Light` to load the saved light data
- `Calculate Light` won't overwrite existing data while `Use Baked Light` is set, to remove baked light data either overwrite it by disabling `Use Baked Light`, call `VoxelTerrain.deleteAllLightData()`, `VoxelTerrain.deleteLightData(x, y, z)` for specific blocks, or just delete the contents of your light data directory

Troubleshooting
----
- The VoxelViewer's initial position should be in range of the sunlight y level, otherwise light won't flow down
- Don't expect sunlight to flow down unless the air blocks at sunlight y level are loaded, this likely makes this unsuitable for typical Minecraft terrain with a lot of vertical variation
- You might need to build release templates when you want to export
- Don't put `res://` in the light data directory or it will break on export, just use a relative path like `light` verbatim
- If blocks get dropped or it crashes while calculating light data try reducing view distance, note that initial load is much more permissive than dynamically loading blocks while moving around
- Dropped air blocks may cause unexpected shadows on flat ground due to obstructed light flow
- Minimum light can't be (0, 0, 0) because it represents 'no light'
- An emissive block's light can't be (255, 255, 255) because it represents sunlight
- Light data is passed to the shader through COLOR along with baked AO and modulate color, which work as before
- You will probably need to apply a shader to the block material and reverse sRGB gamma correction with `pow(color, vec3(2.2))` to get appropriate light values if they come out scaled wrong
- Smooth lighting looks better with complex textures to obscure triangle interpolation
- Shadow sampling trick can be used with flat lighting but it's not recommended, it works better with smooth lighting, 8 light decay, and less bright colored emissive blocks as seen in Cubyz
