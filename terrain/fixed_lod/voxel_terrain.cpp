#include "voxel_terrain.h"
#include "../../constants/voxel_constants.h"
#include "../../constants/voxel_string_names.h"
#include "../../edition/voxel_tool_terrain.h"
#include "../../engine/buffered_task_scheduler.h"
#include "../../engine/voxel_engine.h"
#include "../../engine/voxel_engine_updater.h"
#include "../../generators/generate_block_task.h"
#include "../../meshers/blocky/voxel_mesher_blocky.h"
#include "../../meshers/mesh_block_task.h"
#include "../../storage/voxel_buffer_gd.h"
#include "../../storage/voxel_data.h"
#include "../../streams/load_block_data_task.h"
#include "../../streams/save_block_data_task.h"
#include "../../util/containers/container_funcs.h"
#include "../../util/godot/classes/base_material_3d.h" // For property hint in release mode in GDExtension...
#include "../../util/godot/classes/concave_polygon_shape_3d.h"
#include "../../util/godot/classes/engine.h"
#include "../../util/godot/classes/multiplayer_api.h"
#include "../../util/godot/classes/multiplayer_peer.h"
#include "../../util/godot/classes/scene_tree.h"
#include "../../util/godot/classes/script.h"
#include "../../util/godot/classes/shader_material.h"
#include "../../util/godot/core/array.h"
#include "../../util/godot/core/string.h"
#include "../../util/macros.h"
#include "../../util/math/conv.h"
#include "../../util/profiling.h"
#include "../../util/profiling_clock.h"
#include "../../util/string/format.h"
#include "../../util/tasks/async_dependency_tracker.h"
#include "../../util/tasks/threaded_task.h"
#include "../voxel_data_block_enter_info.h"
#include "../voxel_save_completion_tracker.h"
#include "voxel_terrain_multiplayer_synchronizer.h"
#include <mutex>
#include <array>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <optional>
#include <cstdio>
#include <cstdarg>
#include <memory>
#include <iostream>
#include <fstream>
#include "../../util/dstack.h"
#include "../../util/rgblight.h"
#include "../../util/godot/classes/file_access.h"
#include "../../util/godot/file_utils.h"

#ifdef TOOLS_ENABLED
#include "../../meshers/transvoxel/voxel_mesher_transvoxel.h"
#endif

#ifdef VOXEL_ENABLE_INSTANCER
#include "../instancing/voxel_instancer.h"
#endif

namespace zylann::voxel {

uint32_t vector3iKey(Vector3i v);

struct CubicAreaInfo {
	int edge_size; // In data blocks
	int mesh_block_size_factor;
	unsigned int anchor_buffer_index;

	inline bool is_valid() const {
		return edge_size != 0;
	}
};

CubicAreaInfo get_cubic_area_info_from_size(unsigned int size) {
	// Determine size of the cube of blocks
	int edge_size;
	int mesh_block_size_factor;
	switch (size) {
		case 3 * 3 * 3:
			edge_size = 3;
			mesh_block_size_factor = 1;
			break;
		case 4 * 4 * 4:
			edge_size = 4;
			mesh_block_size_factor = 2;
			break;
		default:
			ZN_PRINT_ERROR("Unsupported block count");
			return CubicAreaInfo{ 0, 0, 0 };
	}

	// Pick anchor block, usually within the central part of the cube (that block must be valid)
	const unsigned int anchor_buffer_index = edge_size * edge_size + edge_size + 1;

	return { edge_size, mesh_block_size_factor, anchor_buffer_index };
}

// copied from mesh_block_task.cpp
void copy_block_and_neighbors(
		Span<std::shared_ptr<VoxelBuffer>> blocks,
		VoxelBuffer &dst,
		int min_padding,
		int max_padding,
		int channels_mask,
		Ref<VoxelGenerator> generator,
		const VoxelData &voxel_data,
		uint8_t lod_index,
		Vector3i mesh_block_pos,
		StdVector<Box3i> *out_boxes_to_generate,
		Vector3i *out_origin_in_voxels
) {
	ZN_DSTACK();
	ZN_PROFILE_SCOPE();

	// Extract wanted channels in a list
	const SmallVector<uint8_t, VoxelBuffer::MAX_CHANNELS> channels = VoxelBuffer::mask_to_channels_list(channels_mask);

	// Determine size of the cube of blocks
	const CubicAreaInfo area_info = get_cubic_area_info_from_size(blocks.size());
	ERR_FAIL_COND(!area_info.is_valid());

	std::shared_ptr<VoxelBuffer> &central_buffer = blocks[area_info.anchor_buffer_index];
	ERR_FAIL_COND_MSG(central_buffer == nullptr && generator.is_null(), "Central buffer must be valid");
	if (central_buffer != nullptr) {
		ERR_FAIL_COND_MSG(
				Vector3iUtil::all_members_equal(central_buffer->get_size()) == false, "Central buffer must be cubic"
		);
	}
	const int data_block_size = voxel_data.get_block_size();
	const int mesh_block_size = data_block_size * area_info.mesh_block_size_factor;
	const int padded_mesh_block_size = mesh_block_size + min_padding + max_padding;

	dst.create(padded_mesh_block_size, padded_mesh_block_size, padded_mesh_block_size);

	// TODO Need to provide format differently, this won't work in full load mode where areas are generated on the fly
	// for (unsigned int ci = 0; ci < channels.size(); ++ci) {
	// 	dst.set_channel_depth(ci, central_buffer->get_channel_depth(ci));
	// }
	// This is a hack
	for (unsigned int i = 0; i < blocks.size(); ++i) {
		const std::shared_ptr<VoxelBuffer> &buffer = blocks[i];
		if (buffer != nullptr) {
			// Initialize channel depths from the first non-null block found
			dst.copy_format(*buffer);
			break;
		}
	}

	const Box3i bounds_in_voxels_lod0 = voxel_data.get_bounds();
	const Box3i bounds_in_voxels(bounds_in_voxels_lod0.position >> lod_index, bounds_in_voxels_lod0.size >> lod_index);

	// TODO In terrains that only work with caches, we should never consider generating voxels from here.
	// This is the case of VoxelTerrain, which is now doing unnecessary box subtraction calculations...

	const Vector3i min_pos = -Vector3iUtil::create(min_padding);
	const Vector3i max_pos = Vector3iUtil::create(mesh_block_size + max_padding);

	const Vector3i origin_in_voxels_without_padding =
			mesh_block_pos * (area_info.mesh_block_size_factor * data_block_size);
	const Vector3i origin_in_voxels = origin_in_voxels_without_padding - Vector3iUtil::create(min_padding);
	const Vector3i origin_in_voxels_lod0 = origin_in_voxels << lod_index;

	// These boxes are initially relative to the minimum corner of the minimum chunk.
	// TODO Candidate for temp allocator (or SmallVector?)
	StdVector<Box3i> boxes_to_generate;
	const Box3i mesh_data_box = Box3i::from_min_max(min_pos, max_pos);
	if (contains(blocks.to_const(), std::shared_ptr<VoxelBuffer>())) {
		const Box3i bounds_local(bounds_in_voxels.position - origin_in_voxels_without_padding, bounds_in_voxels.size);
		const Box3i box = mesh_data_box.clipped(bounds_local); // Prevent generation outside fixed bounds
		if (!box.is_empty()) {
			boxes_to_generate.push_back(box);
		}
	}

	{
		// TODO The following logic might as well be simplified and moved to VoxelData.
		// We are just sampling or generating data in a given area.

		const Vector3i data_block_pos0 = mesh_block_pos * area_info.mesh_block_size_factor;
		SpatialLock3D::Read srlock(
				voxel_data.get_spatial_lock(lod_index),
				BoxBounds3i(
						data_block_pos0 - Vector3i(1, 1, 1), data_block_pos0 + Vector3iUtil::create(area_info.edge_size)
				)
		);

		// Using ZXY as convention to reconstruct positions with thread locking consistency
		unsigned int block_index = 0;
		for (int z = -1; z < area_info.edge_size - 1; ++z) {
			for (int x = -1; x < area_info.edge_size - 1; ++x) {
				for (int y = -1; y < area_info.edge_size - 1; ++y) {
					const Vector3i offset = data_block_size * Vector3i(x, y, z);
					const std::shared_ptr<VoxelBuffer> &src = blocks[block_index];
					++block_index;

					if (src == nullptr) {
						continue;
					}

					const Vector3i src_min = min_pos - offset;
					const Vector3i src_max = max_pos - offset;

					for (const uint8_t channel_index : channels) {
						dst.copy_channel_from(*src, src_min, src_max, Vector3i(), channel_index);
					}

					if (boxes_to_generate.size() > 0) {
						// Subtract edited box from the area to generate
						// TODO This approach allows to batch boxes if necessary,
						// but is it just better to do it anyways for every clipped box?
						ZN_PROFILE_SCOPE_NAMED("Box subtract");
						const unsigned int input_count = boxes_to_generate.size();
						const Box3i block_box =
								Box3i(offset, Vector3iUtil::create(data_block_size)).clipped(mesh_data_box);

						for (unsigned int box_index = 0; box_index < input_count; ++box_index) {
							const Box3i box = boxes_to_generate[box_index];
							// Remainder boxes are added to the end of the list
							box.difference_to_vec(block_box, boxes_to_generate);
#ifdef DEBUG_ENABLED
							// Difference should add boxes to the vector, not remove any
							CRASH_COND(box_index >= boxes_to_generate.size());
#endif
						}

						// Remove input boxes
						boxes_to_generate.erase(boxes_to_generate.begin(), boxes_to_generate.begin() + input_count);
					}
				}
			}
		}
	}

	// Undo padding to go back to proper buffer coordinates
	for (Box3i &box : boxes_to_generate) {
		box.position += Vector3iUtil::create(min_padding);
	}

	if (out_origin_in_voxels != nullptr) {
		*out_origin_in_voxels = origin_in_voxels_lod0;
	}

	if (out_boxes_to_generate != nullptr) {
		// Delegate generation to the caller
		append_array(*out_boxes_to_generate, boxes_to_generate);

	} else {
		// Complete data with generated voxels on the CPU
		ZN_PROFILE_SCOPE_NAMED("Generate");
		VoxelBuffer generated_voxels(VoxelBuffer::ALLOCATOR_POOL);

#ifdef VOXEL_ENABLE_MODIFIERS
		const VoxelModifierStack &modifiers = voxel_data.get_modifiers();
#endif

		for (const Box3i &box : boxes_to_generate) {
			ZN_PROFILE_SCOPE_NAMED("Box");
			// print_line(String("size={0}").format(varray(box.size.to_vec3())));
			generated_voxels.create(box.size);
			// generated_voxels.set_voxel_f(2.0f, box.size.x / 2, box.size.y / 2, box.size.z / 2,
			// VoxelBuffer::CHANNEL_SDF);
			VoxelGenerator::VoxelQueryData q{ generated_voxels,
											  (box.position << lod_index) + origin_in_voxels_lod0,
											  lod_index };

			if (generator.is_valid()) {
				generator->generate_block(q);
			}
#ifdef VOXEL_ENABLE_MODIFIERS
			modifiers.apply(q.voxel_buffer, AABB(q.origin_in_voxels, q.voxel_buffer.get_size() << lod_index));
#endif

			for (const uint8_t channel_index : channels) {
				dst.copy_channel_from(
						generated_voxels, Vector3i(), generated_voxels.get_size(), box.position, channel_index
				);
			}
		}
	}
}

VoxelTerrain::VoxelTerrain() {
	// Note: don't do anything heavy in the constructor.
	// Godot may create and destroy dozens of instances of all node types on startup,
	// due to how ClassDB gets its default values.

	set_notify_transform(true);

	_data = make_shared_instance<VoxelData>();

	// TODO Should it actually be finite for better discovery?
	// Infinite by default
	_data->set_bounds(Box3i::from_center_extents(Vector3i(), Vector3iUtil::create(constants::MAX_VOLUME_EXTENT)));

	_streaming_dependency = make_shared_instance<StreamingDependency>();
	_meshing_dependency = make_shared_instance<MeshingDependency>();

	struct ApplyMeshUpdateTask : public ITimeSpreadTask {
		void run(TimeSpreadTaskContext &ctx) override {
			if (!VoxelEngine::get_singleton().is_volume_valid(volume_id)) {
				// The node can have been destroyed while this task was still pending
				ZN_PRINT_VERBOSE("Cancelling ApplyMeshUpdateTask, volume_id is invalid");
				return;
			}
			self->apply_mesh_update(data);
		}
		VolumeID volume_id;
		VoxelTerrain *self = nullptr;
		VoxelEngine::BlockMeshOutput data;
	};

	// Mesh updates are spread over frames by scheduling them in a task runner of VoxelEngine,
	// but instead of using a reception buffer we use a callback,
	// because this kind of task scheduling would otherwise delay the update by 1 frame
	VoxelEngine::VolumeCallbacks callbacks;
	callbacks.data = this;
	callbacks.mesh_output_callback = [](void *cb_data, VoxelEngine::BlockMeshOutput &ob) {
		VoxelTerrain *self = reinterpret_cast<VoxelTerrain *>(cb_data);
		ApplyMeshUpdateTask *task = ZN_NEW(ApplyMeshUpdateTask);
		task->volume_id = self->_volume_id;
		task->self = self;
		task->data = std::move(ob);
		VoxelEngine::get_singleton().push_main_thread_time_spread_task(task);
	};
	callbacks.data_output_callback = [](void *cb_data, VoxelEngine::BlockDataOutput &ob) {
		VoxelTerrain *self = reinterpret_cast<VoxelTerrain *>(cb_data);
		self->apply_data_block_response(ob);
	};

	_volume_id = VoxelEngine::get_singleton().add_volume(callbacks);

	// TODO Can't setup a default mesher anymore due to a Godot 4 warning...
	// For ease of use in editor
	// Ref<VoxelMesherBlocky> default_mesher;
	// default_mesher.instantiate();
	// _mesher = default_mesher;
}

VoxelTerrain::~VoxelTerrain() {
	ZN_PRINT_VERBOSE("Destroying VoxelTerrain");
	_streaming_dependency->valid = false;
	_meshing_dependency->valid = false;
	VoxelEngine::get_singleton().remove_volume(_volume_id);
}

void VoxelTerrain::set_material_override(Ref<Material> material) {
	if (_material_override == material) {
		return;
	}
	_material_override = material;
	_mesh_map.for_each_block([material](VoxelMeshBlockVT &block) { //
		block.set_material_override(material);
	});
}

Ref<Material> VoxelTerrain::get_material_override() const {
	return _material_override;
}

#ifdef VOXEL_ENABLE_GPU
void VoxelTerrain::set_generator_use_gpu(bool enabled) {
	_generator_use_gpu = enabled;
}

bool VoxelTerrain::get_generator_use_gpu() const {
	return _generator_use_gpu;
}
#endif

void VoxelTerrain::set_stream(Ref<VoxelStream> p_stream) {
	if (p_stream == get_stream()) {
		return;
	}

	_data->set_stream(p_stream);

	StreamingDependency::reset(_streaming_dependency, p_stream, get_generator());

#ifdef TOOLS_ENABLED
	if (p_stream.is_valid()) {
		if (Engine::get_singleton()->is_editor_hint()) {
			Ref<Script> stream_script = p_stream->get_script();
			if (stream_script.is_valid()) {
				// Safety check. It's too easy to break threads by making a script reload.
				// You can turn it back on, but be careful.
				_run_stream_in_editor = false;
				notify_property_list_changed();
			}
		}
	}
#endif

	_on_stream_params_changed();
}

Ref<VoxelStream> VoxelTerrain::get_stream() const {
	return _data->get_stream();
}

void VoxelTerrain::set_generator(Ref<VoxelGenerator> p_generator) {
	if (p_generator == get_generator()) {
		return;
	}

	Ref<VoxelGenerator> prev_generator = get_generator();
	if (prev_generator.is_valid()) {
		prev_generator->clear_cache();
		// TODO if we were to share this generator on multiple terrains, cache should not be entirely cleared. Instead,
		// we should just remove the area from all paired viewers.
	}

	_data->set_generator(p_generator);

	MeshingDependency::reset(_meshing_dependency, _mesher, p_generator);
	StreamingDependency::reset(_streaming_dependency, get_stream(), p_generator);

#ifdef TOOLS_ENABLED
	if (p_generator.is_valid()) {
		if (Engine::get_singleton()->is_editor_hint()) {
			Ref<Script> generator_script = p_generator->get_script();
			if (generator_script.is_valid()) {
				// Safety check. It's too easy to break threads by making a script reload.
				// You can turn it back on, but be careful.
				_run_stream_in_editor = false;
				notify_property_list_changed();
			}
		}
	}
#endif

	_on_stream_params_changed();
}

Ref<VoxelGenerator> VoxelTerrain::get_generator() const {
	return _data->get_generator();
}

// void VoxelTerrain::_set_block_size_po2(int p_block_size_po2) {
// 	_data_map.create(0);
// }

unsigned int VoxelTerrain::get_data_block_size_pow2() const {
	return _data->get_block_size_po2();
}

unsigned int VoxelTerrain::get_mesh_block_size_pow2() const {
	return _mesh_block_size_po2;
}

void VoxelTerrain::set_mesh_block_size(unsigned int mesh_block_size) {
	mesh_block_size = math::clamp(mesh_block_size, get_data_block_size(), constants::MAX_BLOCK_SIZE);

	unsigned int po2;
	switch (mesh_block_size) {
		case 16:
			po2 = 4;
			break;
		case 32:
			po2 = 5;
			break;
		default:
			mesh_block_size = 16;
			po2 = 4;
			break;
	}
	if (mesh_block_size == get_mesh_block_size()) {
		return;
	}

	_mesh_block_size_po2 = po2;

	// Unload all mesh blocks regardless of refcount
	clear_mesh_map();

	// Make paired viewers re-view the new meshable area
	for (unsigned int i = 0; i < _paired_viewers.size(); ++i) {
		PairedViewer &viewer = _paired_viewers[i];
		// Resetting both because it's a re-initialization.
		// We could also be doing that before or after their are shifted.
		viewer.state.mesh_box = Box3i();
		viewer.prev_state.mesh_box = Box3i();
	}

	// VoxelEngine::get_singleton().set_volume_render_block_size(_volume_id, mesh_block_size);

	// No update on bounds because we can support a mismatch, as long as it is a multiple of data block size
	// set_bounds(_bounds_in_voxels);
}

void VoxelTerrain::restart_stream() {
	_on_stream_params_changed();
}

void VoxelTerrain::_on_stream_params_changed() {
	stop_streamer();
	stop_updater();

	// if (_stream.is_valid()) {
	// 	const int stream_block_size_po2 = _stream->get_block_size_po2();
	// 	_set_block_size_po2(stream_block_size_po2);
	// }

	// The whole map might change, so regenerate it
	reset_map();

	_data->set_format(get_internal_format());

	if ((get_stream().is_valid() || get_generator().is_valid()) &&
		(Engine::get_singleton()->is_editor_hint() == false || _run_stream_in_editor)) {
		start_streamer();
		start_updater();
	}

	update_configuration_warnings();
}

void VoxelTerrain::_on_gi_mode_changed() {
	const GeometryInstance3D::GIMode gi_mode = get_gi_mode();
	_mesh_map.for_each_block([gi_mode](VoxelMeshBlockVT &block) { //
		block.set_gi_mode(gi_mode);
	});
}

void VoxelTerrain::_on_shadow_casting_changed() {
	const RenderingServer::ShadowCastingSetting mode = RenderingServer::ShadowCastingSetting(get_shadow_casting());
	_mesh_map.for_each_block([mode](VoxelMeshBlockVT &block) { //
		block.set_shadow_casting(mode);
	});
}

void VoxelTerrain::_on_render_layers_mask_changed() {
	const int mask = get_render_layers_mask();
	_mesh_map.for_each_block([mask](VoxelMeshBlockVT &block) { //
		block.set_render_layers_mask(mask);
	});
}

Ref<VoxelMesher> VoxelTerrain::get_mesher() const {
	return _mesher;
}

void VoxelTerrain::set_mesher(Ref<VoxelMesher> mesher) {
	if (mesher == _mesher) {
		return;
	}

	_mesher = mesher;

	MeshingDependency::reset(_meshing_dependency, _mesher, get_generator());

	stop_updater();

	if (_mesher.is_valid()) {
		start_updater();
		// Voxel appearance might completely change
		remesh_all_blocks();
	}

	update_configuration_warnings();
}

void VoxelTerrain::get_viewers_in_area(StdVector<ViewerID> &out_viewer_ids, Box3i voxel_box) const {
	const Box3i block_box = voxel_box.downscaled(get_data_block_size());

	for (auto it = _paired_viewers.begin(); it != _paired_viewers.end(); ++it) {
		const PairedViewer &viewer = *it;

		if (viewer.state.data_box.intersects(block_box)) {
			out_viewer_ids.push_back(viewer.id);
		}
	}
}

void VoxelTerrain::set_generate_collisions(bool enabled) {
	_generate_collisions = enabled;
}

void VoxelTerrain::set_collision_layer(int layer) {
	_collision_layer = layer;
	_mesh_map.for_each_block([layer](VoxelMeshBlockVT &block) { //
		block.set_collision_layer(layer);
	});
}

int VoxelTerrain::get_collision_layer() const {
	return _collision_layer;
}

void VoxelTerrain::set_collision_mask(int mask) {
	_collision_mask = mask;
	_mesh_map.for_each_block([mask](VoxelMeshBlockVT &block) { //
		block.set_collision_mask(mask);
	});
}

int VoxelTerrain::get_collision_mask() const {
	return _collision_mask;
}

void VoxelTerrain::set_collision_margin(float margin) {
	_collision_margin = margin;
	_mesh_map.for_each_block([margin](VoxelMeshBlockVT &block) { //
		block.set_collision_margin(margin);
	});
}

float VoxelTerrain::get_collision_margin() const {
	return _collision_margin;
}

int VoxelTerrain::get_max_view_distance() const {
	return _max_view_distance_voxels;
}

void VoxelTerrain::set_max_view_distance(int distance_in_voxels) {
	ERR_FAIL_COND(distance_in_voxels < 0);
	_max_view_distance_voxels = distance_in_voxels;

#ifdef VOXEL_ENABLE_INSTANCER
	if (_instancer != nullptr) {
		_instancer->update_mesh_lod_distances_from_parent();
	}
#endif
}

void VoxelTerrain::set_block_enter_notification_enabled(bool enable) {
	_block_enter_notification_enabled = enable;

	if (enable == false) {
		for (auto it = _loading_blocks.begin(); it != _loading_blocks.end(); ++it) {
			LoadingBlock &lb = it->second;
			lb.viewers_to_notify.clear();
		}
	}
}

bool VoxelTerrain::is_block_enter_notification_enabled() const {
	return _block_enter_notification_enabled;
}

void VoxelTerrain::set_area_edit_notification_enabled(bool enable) {
	_area_edit_notification_enabled = enable;
}

bool VoxelTerrain::is_area_edit_notification_enabled() const {
	return _area_edit_notification_enabled;
}

void VoxelTerrain::set_automatic_loading_enabled(bool enable) {
	_automatic_loading_enabled = enable;
}

bool VoxelTerrain::is_automatic_loading_enabled() const {
	return _automatic_loading_enabled;
}

void VoxelTerrain::try_schedule_mesh_update(VoxelMeshBlockVT &mesh_block) {
	ZN_PROFILE_SCOPE();
	if (mesh_block.is_in_update_list) {
		// Already in the list
		return;
	}
	if (mesh_block.mesh_viewers.get() == 0 && mesh_block.collision_viewers.get() == 0) {
		// No viewers want mesh on this block (why even call this function then?)
		return;
	}

	uint32_t key = vector3iKey(mesh_block.position);
	_lightMap.erase(key);
	_lightProcessed.erase(key);

	const int render_to_data_factor = get_mesh_block_size() / get_data_block_size();

	const Box3i data_box =
			Box3i(mesh_block.position * render_to_data_factor, Vector3iUtil::create(render_to_data_factor)).padded(1);

	// If we get an empty box at this point, something is wrong with the caller
	ZN_ASSERT_RETURN(!data_box.is_empty());

	const bool data_available = _data->has_all_blocks_in_area(data_box, 0);

	if (data_available) {
		// Regardless of if the updater is updating the block already,
		// the block could have been modified again so we schedule another update
		mesh_block.is_in_update_list = true;
		_blocks_pending_update.push_back(mesh_block.position);
	}
}

void VoxelTerrain::view_mesh_block(Vector3i bpos, bool mesh_flag, bool collision_flag) {
	if (mesh_flag == false && collision_flag == false) {
		// Why even call the function?
		return;
	}

	VoxelMeshBlockVT *block = _mesh_map.get_block(bpos);

	if (block == nullptr) {
		// Create if not found
		block = ZN_NEW(VoxelMeshBlockVT(bpos, get_mesh_block_size()));
		block->set_world(get_world_3d());
		_mesh_map.set_block(bpos, block);
	}
	CRASH_COND(block == nullptr);

	if (mesh_flag) {
		block->mesh_viewers.add();
	}
	if (collision_flag) {
		block->collision_viewers.add();
	}

	// This is needed in case a viewer wants to view meshes in places data blocks are already present.
	// Before that, meshes were updated only when a data block was loaded or modified,
	// so changing block size or viewer flags did not make meshes appear.
	try_schedule_mesh_update(*block);

	// TODO this logic schedules a mesh update even if there is a mesh already. It hides the fact that mixing up
	// viewers with collisions and viewers without will not actually create colliders/meshes individually.

	// TODO viewers with varying flags during the game is not supported at the moment.
	// They have to be re-created, which may cause world re-load...
}

void VoxelTerrain::unview_mesh_block(Vector3i bpos, bool mesh_flag, bool collision_flag) {
	VoxelMeshBlockVT *block = _mesh_map.get_block(bpos);
	// Mesh blocks are created on first view call,
	// so that would mean we unview one without viewing it in the first place
	ERR_FAIL_COND(block == nullptr);

	if (mesh_flag) {
		block->mesh_viewers.remove();
		if (block->mesh_viewers.get() == 0) {
			// Mesh no longer required
			block->drop_mesh();
			block->set_visible(false);
		}
	}

	if (collision_flag) {
		block->collision_viewers.remove();
		if (block->collision_viewers.get() == 0) {
			// Collision no longer required
			block->drop_collision();
			block->set_collision_enabled(false);
		}
	}

	if (block->mesh_viewers.get() == 0 && block->collision_viewers.get() == 0) {
		unload_mesh_block(bpos);
	}
}

void VoxelTerrain::unload_mesh_block(Vector3i bpos) {
	StdVector<Vector3i> &blocks_pending_update = _blocks_pending_update;

	bool was_loaded = false;
	_mesh_map.remove_block(bpos, [&blocks_pending_update, &was_loaded](const VoxelMeshBlockVT &block) {
		if (block.is_in_update_list) {
			// That block was in the list of blocks to update later in the process loop, we'll need to unregister
			// it. We expect that block to be in that list. If it isn't, something wrong happened with its state.
			ERR_FAIL_COND(!unordered_remove_value(blocks_pending_update, block.position));
		}
		was_loaded = block.is_loaded;
	});

#ifdef VOXEL_ENABLE_INSTANCER
	if (_instancer != nullptr) {
		_instancer->on_mesh_block_exit(bpos, 0);
	}
#endif

	// It's possible the block was added as the viewer moved, but did not have the time to receive its first mesh update
	if (was_loaded) {
		emit_mesh_block_exited(bpos);
	}
}

void VoxelTerrain::save_all_modified_blocks(bool with_copy, std::shared_ptr<AsyncDependencyTracker> tracker) {
	ZN_PROFILE_SCOPE();
	Ref<VoxelStream> stream = get_stream();
	ERR_FAIL_COND_MSG(stream.is_null(), "Attempting to save modified blocks, but there is no stream to save them to.");

	BufferedTaskScheduler &task_scheduler = BufferedTaskScheduler::get_for_current_thread();

	// That may cause a stutter, so should be used when the player won't notice
	_data->consume_all_modifications(_blocks_to_save, with_copy);

#ifdef VOXEL_ENABLE_INSTANCER
	if (stream.is_valid() && _instancer != nullptr && stream->supports_instance_blocks()) {
		_instancer->save_all_modified_blocks(task_scheduler, tracker, true);
	}
#endif

	consume_block_data_save_requests(
			task_scheduler,
			tracker,
			// Require all data we just gathered to be written to disk if the stream uses a cache. So if the
			// game crashes or gets killed after all tasks are done, data won't be lost.
			true
	);

	if (tracker != nullptr) {
		// Using buffered count instead of `_blocks_to_save` because it can also contain tasks from VoxelInstancer
		tracker->set_count(task_scheduler.get_io_count());
	}

	// Schedule all tasks
	task_scheduler.flush();
}

const VoxelTerrain::Stats &VoxelTerrain::get_stats() const {
	return _stats;
}

#ifdef VOXEL_ENABLE_INSTANCER
void VoxelTerrain::set_instancer(VoxelInstancer *instancer) {
	if (_instancer != nullptr && instancer != nullptr) {
		ERR_FAIL_COND_MSG(_instancer != nullptr, "No more than one VoxelInstancer per terrain");
	}
	_instancer = instancer;
}
#endif

void VoxelTerrain::get_meshed_block_positions(StdVector<Vector3i> &out_positions) const {
	_mesh_map.for_each_block([&out_positions](const VoxelMeshBlock &mesh_block) {
		if (mesh_block.has_mesh()) {
			out_positions.push_back(mesh_block.position);
		}
	});
}

// This function is primarily intended for editor use cases at the moment.
// It will be slower than using the instancing generation events,
// because it has to query VisualServer, which then allocates and decodes vertex buffers (assuming they are cached).
Array VoxelTerrain::get_mesh_block_surface(Vector3i block_pos) const {
	ZN_PROFILE_SCOPE();

	Ref<Mesh> mesh;
	{
		const VoxelMeshBlockVT *block = _mesh_map.get_block(block_pos);
		if (block != nullptr) {
			mesh = block->get_mesh();
		}
	}

	if (mesh.is_valid()) {
		return mesh->surface_get_arrays(0);
	}

	return Array();
}

Dictionary VoxelTerrain::_b_get_statistics() const {
	Dictionary d;

	// Breakdown of time spent in _process
	d["time_detect_required_blocks"] = _stats.time_detect_required_blocks;
	d["time_request_blocks_to_load"] = _stats.time_request_blocks_to_load;
	d["time_process_load_responses"] = _stats.time_process_load_responses;
	d["time_request_blocks_to_update"] = _stats.time_request_blocks_to_update;

	d["dropped_block_loads"] = _stats.dropped_block_loads;
	d["dropped_block_meshs"] = _stats.dropped_block_meshs;
	d["updated_blocks"] = _stats.updated_blocks;

	return d;
}

void VoxelTerrain::start_updater() {
	Ref<VoxelMesherBlocky> blocky_mesher = _mesher;
	if (blocky_mesher.is_valid()) {
		Ref<VoxelBlockyLibraryBase> library = blocky_mesher->get_library();
		if (library.is_valid()) {
			// TODO Any way to execute this function just after the TRES resource loader has finished to load?
			// VoxelBlockyLibrary should be baked ahead of time, like MeshLibrary
			library->bake();
		}
	}

	// VoxelEngine::get_singleton().set_volume_mesher(_volume_id, _mesher);
}

void VoxelTerrain::stop_updater() {
	// Invalidate pending tasks
	MeshingDependency::reset(_meshing_dependency, _mesher, get_generator());

	// VoxelEngine::get_singleton().set_volume_mesher(_volume_id, Ref<VoxelMesher>());

	// TODO We can still receive a few mesh delayed mesh updates after this. Is it a problem?
	//_reception_buffers.mesh_output.clear();

	for (const Vector3i bpos : _blocks_pending_update) {
		VoxelMeshBlockVT *block = _mesh_map.get_block(bpos);
		if (block != nullptr) {
			block->is_in_update_list = false;
		}
	}

	_blocks_pending_update.clear();
}

void VoxelTerrain::remesh_all_blocks() {
	_mesh_map.for_each_block([this](VoxelMeshBlockVT &block) { //
		try_schedule_mesh_update(block);
	});
}

// At the moment, this function is for client-side use case in multiplayer scenarios
void VoxelTerrain::generate_block_async(Vector3i block_position) {
	if (_data->has_block(block_position, 0)) {
		// Already exists
		return;
	}
	if (_loading_blocks.find(block_position) != _loading_blocks.end()) {
		// Already loading
		return;
	}

	// if (require_notification) {
	// 	new_loading_block.viewers_to_notify.push_back(viewer_id);
	// }

	LoadingBlock new_loading_block;
	const Box3i block_box(_data->block_to_voxel(block_position), Vector3iUtil::create(_data->get_block_size()));
	for (size_t i = 0; i < _paired_viewers.size(); ++i) {
		const PairedViewer &viewer = _paired_viewers[i];
		if (viewer.state.data_box.intersects(block_box)) {
			new_loading_block.viewers.add();
		}
	}

	if (new_loading_block.viewers.get() == 0) {
		return;
	}

	// Schedule a loading request
	// TODO This could also end up loading from stream
	_loading_blocks.insert({ block_position, new_loading_block });
	_blocks_pending_load.push_back(block_position);
}

void VoxelTerrain::start_streamer() {
	// VoxelEngine::get_singleton().set_volume_stream(_volume_id, _stream);
	// VoxelEngine::get_singleton().set_volume_generator(_volume_id, _generator);
}

void VoxelTerrain::stop_streamer() {
	// Invalidate pending tasks
	StreamingDependency::reset(_streaming_dependency, get_stream(), get_generator());
	// VoxelEngine::get_singleton().set_volume_stream(_volume_id, Ref<VoxelStream>());
	// VoxelEngine::get_singleton().set_volume_generator(_volume_id, Ref<VoxelGenerator>());
	_loading_blocks.clear();
	_blocks_pending_load.clear();
	_quick_reloading_blocks.clear();
	_unloaded_saving_blocks.clear();
}

void VoxelTerrain::clear_mesh_map() {
#ifdef VOXEL_ENABLE_INSTANCER
	if (_instancer != nullptr) {
		VoxelInstancer &instancer = *_instancer;
		_mesh_map.for_each_block([&instancer, this](VoxelMeshBlockVT &block) { //
			instancer.on_mesh_block_exit(block.position, 0);
			if (block.is_loaded) {
				emit_mesh_block_exited(block.position);
			}
		});
	} else
#endif
	{
		_mesh_map.for_each_block([this](VoxelMeshBlockVT &block) { //
			if (block.is_loaded) {
				emit_mesh_block_exited(block.position);
			}
		});
	}

	_mesh_map.clear();
}

void VoxelTerrain::reset_map() {
	// Discard everything, to reload it all

	_data->for_each_block_position([this](const Vector3i &bpos) { //
		emit_data_block_unloaded(bpos);
	});
	_data->reset_maps();

	clear_mesh_map();

	_loading_blocks.clear();
	_blocks_pending_load.clear();
	_blocks_pending_update.clear();
	_blocks_to_save.clear();

	// No need to care about refcounts, we drop everything anyways. Will pair it back on next process.
	_paired_viewers.clear();

	Ref<VoxelGenerator> generator = get_generator();
	if (generator.is_valid()) {
		generator->clear_cache();
	}
}

void VoxelTerrain::post_edit_voxel(Vector3i pos) {
	post_edit_area(Box3i(pos, Vector3i(1, 1, 1)), true);
}

void VoxelTerrain::try_schedule_mesh_update_from_data(const Box3i &box_in_voxels) {
	ZN_PROFILE_SCOPE();
	if (_mesher.is_null()) {
		// No mesher, can't do updates
		return;
	}
	// We pad by 1 because neighbor blocks might be affected visually (for example, baked ambient occlusion)
	const Box3i mesh_box = box_in_voxels.padded(1).downscaled(get_mesh_block_size());
	mesh_box.for_each_cell([this](Vector3i pos) {
		VoxelMeshBlockVT *block = _mesh_map.get_block(pos);
		// There isn't necessarily a mesh block, if the edit happens in a boundary,
		// or if it is done next to a viewer that doesn't need meshes
		if (block != nullptr) {
			try_schedule_mesh_update(*block);
		}
	});
}

void VoxelTerrain::post_edit_area(Box3i box_in_voxels, bool update_mesh) {
	_data->mark_area_modified(box_in_voxels, nullptr, false);

	box_in_voxels.clip(_data->get_bounds());

	// TODO Maybe remove this in preference for multiplayer synchronizer virtual functions?
	if (_area_edit_notification_enabled) {
		GDVIRTUAL_CALL(_on_area_edited, box_in_voxels.position, box_in_voxels.size);
	}

	if (_multiplayer_synchronizer != nullptr && _multiplayer_synchronizer->is_server()) {
		// TODO This is not efficient when the user does many individual modifications in a specific area.
		// We would either have to batch modified areas somehow, or expose a transactional API to the user
		// (begin(area), edit in area, end(area))
		_multiplayer_synchronizer->send_area(box_in_voxels);
	}

	if (update_mesh) {
		try_schedule_mesh_update_from_data(box_in_voxels);

#ifdef VOXEL_ENABLE_INSTANCER
		if (_instancer != nullptr) {
			_instancer->on_area_edited(box_in_voxels);
		}
#endif
	}
}

void VoxelTerrain::_notification(int p_what) {
	struct SetWorldAction {
		World3D *world;
		SetWorldAction(World3D *w) : world(w) {}
		void operator()(VoxelMeshBlockVT &block) {
			block.set_world(world);
		}
	};

	struct SetParentVisibilityAction {
		bool visible;
		SetParentVisibilityAction(bool v) : visible(v) {}
		void operator()(VoxelMeshBlockVT &block) {
			block.set_parent_visible(visible);
		}
	};

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
			set_process(true);
#ifdef TOOLS_ENABLED
#ifdef VOXEL_ENABLE_SMOOTH_MESHING
			// In the editor, auto-configure a default mesher, for convenience.
			// Because Godot has a property hint to automatically instantiate a resource, but if that resource is
			// abstract, it doesn't work... and it cannot be a default value because such practice was deprecated with a
			// warning in Godot 4.
			if (Engine::get_singleton()->is_editor_hint() && !get_mesher().is_valid()) {
				Ref<VoxelMesherTransvoxel> mesher;
				mesher.instantiate();
				set_mesher(mesher);
			}
#endif
#endif
			break;

		case NOTIFICATION_PROCESS:
			// Can't do that in enter tree because Godot is "still setting up children".
			// Can't do that in ready either because Godot says node state is locked.
			// This hack is quite miserable.
			VoxelEngineUpdater::ensure_existence(get_tree());

			process();
			break;

		case NOTIFICATION_EXIT_TREE:
			break;

		case NOTIFICATION_ENTER_WORLD: {
			World3D *world = *get_world_3d();
			_mesh_map.for_each_block(SetWorldAction(world));
#ifdef TOOLS_ENABLED
			if (debug_is_draw_enabled()) {
				_debug_renderer.set_world(is_visible_in_tree() ? world : nullptr);
			}
#endif
		} break;

		case NOTIFICATION_EXIT_WORLD:
			_mesh_map.for_each_block(SetWorldAction(nullptr));
#ifdef TOOLS_ENABLED
			_debug_renderer.set_world(nullptr);
#endif
			break;

		case NOTIFICATION_VISIBILITY_CHANGED:
			_mesh_map.for_each_block(SetParentVisibilityAction(is_visible()));
#ifdef TOOLS_ENABLED
			if (debug_is_draw_enabled()) {
				_debug_renderer.set_world(is_visible_in_tree() ? *get_world_3d() : nullptr);
			}
#endif
			break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			const Transform3D transform = get_global_transform();
			// VoxelEngine::get_singleton().set_volume_transform(_volume_id, transform);

			if (!is_inside_tree()) {
				// The transform and other properties can be set by the scene loader,
				// before we enter the tree
				return;
			}

			_mesh_map.for_each_block([&transform](VoxelMeshBlockVT &block) { //
				block.set_parent_transform(transform);
			});

		} break;

		default:
			break;
	}
}

namespace {

Vector3i get_block_center(Vector3i pos, int bs) {
	return pos * bs + Vector3iUtil::create(bs / 2);
}

void init_sparse_grid_priority_dependency(
		PriorityDependency &dep,
		Vector3i block_position,
		int block_size,
		std::shared_ptr<PriorityDependency::ViewersData> &shared_viewers_data,
		const Transform3D &volume_transform
) {
	const Vector3i voxel_pos = get_block_center(block_position, block_size);
	const float block_radius = block_size / 2;
	dep.shared = shared_viewers_data;
	dep.world_position = to_vec3f(volume_transform.xform(voxel_pos));
	const float transformed_block_radius =
			volume_transform.basis.xform(Vector3(block_radius, block_radius, block_radius)).length();

	// Distance beyond which no field of view can overlap the block.
	// Doubling block radius to account for an extra margin of blocks,
	// since they are used to provide neighbors when meshing
	dep.drop_distance_squared =
			math::squared(shared_viewers_data->highest_view_distance + 2.f * transformed_block_radius);
}

void request_block_load(
		VolumeID volume_id,
		std::shared_ptr<StreamingDependency> stream_dependency,
		Vector3i block_pos,
		std::shared_ptr<PriorityDependency::ViewersData> &shared_viewers_data,
		const Transform3D volume_transform,
		BufferedTaskScheduler &scheduler,
		bool use_gpu,
		const std::shared_ptr<VoxelData> &voxel_data
) {
	ZN_ASSERT(stream_dependency != nullptr);

#ifdef VOXEL_ENABLE_GPU
	if (use_gpu && (stream_dependency->generator.is_null() || !stream_dependency->generator->supports_shaders())) {
		use_gpu = false;
	}
#endif

	const unsigned int data_block_size = voxel_data->get_block_size();

	if (stream_dependency->stream.is_valid()) {
		PriorityDependency priority_dependency;
		init_sparse_grid_priority_dependency(
				priority_dependency, block_pos, data_block_size, shared_viewers_data, volume_transform
		);

		const bool request_instances = false;
		LoadBlockDataTask *task = ZN_NEW(LoadBlockDataTask(
				volume_id,
				block_pos,
				0,
				data_block_size,
				request_instances,
				stream_dependency,
				priority_dependency,
				true,
				use_gpu,
				voxel_data,
				TaskCancellationToken()
		));

		scheduler.push_io_task(task);

	} else {
		// Directly generate the block without checking the stream
		ERR_FAIL_COND(stream_dependency->generator.is_null());

		VoxelGenerator::BlockTaskParams params;
		params.format = voxel_data->get_format();
		params.volume_id = volume_id;
		params.block_position = block_pos;
		params.block_size = data_block_size;
		params.stream_dependency = stream_dependency;
#ifdef VOXEL_ENABLE_GPU
		params.use_gpu = use_gpu;
#endif
		params.data = voxel_data;

		init_sparse_grid_priority_dependency(
				params.priority_dependency, block_pos, data_block_size, shared_viewers_data, volume_transform
		);

		IThreadedTask *task = stream_dependency->generator->create_block_task(params);

		scheduler.push_main_task(task);
	}
}

} // namespace

void VoxelTerrain::send_data_load_requests() {
	ZN_PROFILE_SCOPE();

	if (_blocks_pending_load.size() > 0) {
		std::shared_ptr<PriorityDependency::ViewersData> shared_viewers_data =
				VoxelEngine::get_singleton().get_shared_viewers_data_from_default_world();

		const Transform3D volume_transform = get_global_transform();

		BufferedTaskScheduler &scheduler = BufferedTaskScheduler::get_for_current_thread();

		// Blocks to load
		for (size_t i = 0; i < _blocks_pending_load.size(); ++i) {
			const Vector3i block_pos = _blocks_pending_load[i];

			auto saving_block_it = _unloaded_saving_blocks.find(block_pos);
			const bool quick_reloading = saving_block_it != _unloaded_saving_blocks.end();

			if (quick_reloading) {
				ZN_PROFILE_SCOPE_NAMED("Quick reloading");
				// The block is unloaded and currently waiting to be saved but we already want it back. This simulates a
				// request and will complete on the next process.
				// Ideally this shouldn't happen often. This is a corner case that occurs if the player moves fast
				// back and forth or the task runner is overloaded.
				std::shared_ptr<VoxelBuffer> voxel_data =
						make_shared_instance<VoxelBuffer>(VoxelBuffer::ALLOCATOR_POOL);
				// Duplicating to make sure the saving version doesn't get altered by possible upcoming modifications.
				saving_block_it->second->copy_to(*voxel_data, true);
				_quick_reloading_blocks.push_back(QuickReloadingBlock{ voxel_data, block_pos });
				// Don't erase it just yet, we may only do this once we know it is saved
				// _unloaded_saving_blocks.erase(saving_block_it);

				// Notes:
				// Could we change the design so that saving tasks actually save a box of VoxelData?
				// To do that we would have to NOT remove data blocks of which refcount becomes 0. Instead, ownership
				// would sort of be given to a saving task. That task would make a copy of modified chunks and only then
				// remove them if they still have 0 viewers.
				// If a viewer moves back into the area, it would simply find the chunks again and no loading would be
				// needed. If those chunks get modified while saving is underway, it would still work fine as the saving
				// task would lock the saved regions for reading (which is currently a problem already, because no
				// locking actually occurs!).

			} else {
				request_block_load(
						_volume_id,
						_streaming_dependency,
						block_pos,
						shared_viewers_data,
						volume_transform,
						scheduler,
						_generator_use_gpu,
						_data
				);
			}
		}
		scheduler.flush();
		_blocks_pending_load.clear();
	}
}

void VoxelTerrain::consume_block_data_save_requests(
		BufferedTaskScheduler &task_scheduler,
		std::shared_ptr<AsyncDependencyTracker> saving_tracker,
		bool with_flush
) {
	ZN_PROFILE_SCOPE();

	// Blocks to save
	if (get_stream().is_valid()) {
		for (const VoxelData::BlockToSave &b : _blocks_to_save) {
			ZN_PRINT_VERBOSE(format("Requesting save of block {}", b.position));

			SaveBlockDataTask *task = ZN_NEW(SaveBlockDataTask(
					_volume_id, b.position, 0, b.voxels, _streaming_dependency, saving_tracker, with_flush
			));

			// No priority data, saving doesn't need sorting.
			task_scheduler.push_io_task(task);
		}
	} else {
		if (_blocks_to_save.size() > 0) {
			ZN_PRINT_VERBOSE(format("Not saving {} blocks because no stream is assigned", _blocks_to_save.size()));
		}
	}

	// print_line(String("Sending {0} block requests").format(varray(input.blocks_to_emerge.size())));
	_blocks_to_save.clear();
}

void VoxelTerrain::emit_data_block_loaded(Vector3i bpos) {
	// Not sure about exposing buffers directly... some stuff on them is useful to obtain directly,
	// but also it allows scripters to mess with voxels in a way they should not.
	// Example: modifying voxels without locking them first, while another thread may be reading them at the same
	// time. The same thing could happen the other way around (threaded task modifying voxels while you try to read
	// them). It isn't planned to expose VoxelBuffer locks because there are too many of them, it may likely shift
	// to another system in the future, and might even be changed to no longer inherit Reference. So unless this is
	// absolutely necessary, buffers aren't exposed. Workaround: use VoxelTool
	// const Variant vbuffer = block->voxels;
	// const Variant *args[2] = { &vpos, &vbuffer };
	emit_signal(VoxelStringNames::get_singleton().block_loaded, bpos);
}

void VoxelTerrain::emit_data_block_unloaded(Vector3i bpos) {
	emit_signal(VoxelStringNames::get_singleton().block_unloaded, bpos);
}

void VoxelTerrain::emit_mesh_block_entered(Vector3i bpos) {
	// Not sure about exposing buffers directly... some stuff on them is useful to obtain directly,
	// but also it allows scripters to mess with voxels in a way they should not.
	// Example: modifying voxels without locking them first, while another thread may be reading them at the same
	// time. The same thing could happen the other way around (threaded task modifying voxels while you try to read
	// them). It isn't planned to expose VoxelBuffer locks because there are too many of them, it may likely shift
	// to another system in the future, and might even be changed to no longer inherit Reference. So unless this is
	// absolutely necessary, buffers aren't exposed. Workaround: use VoxelTool
	// const Variant vbuffer = block->voxels;
	// const Variant *args[2] = { &vpos, &vbuffer };
	emit_signal(VoxelStringNames::get_singleton().mesh_block_entered, bpos);
}

void VoxelTerrain::emit_mesh_block_exited(Vector3i bpos) {
	emit_signal(VoxelStringNames::get_singleton().mesh_block_exited, bpos);
}

bool VoxelTerrain::try_get_paired_viewer_index(ViewerID id, size_t &out_i) const {
	for (size_t i = 0; i < _paired_viewers.size(); ++i) {
		const PairedViewer &p = _paired_viewers[i];
		if (p.id == id) {
			out_i = i;
			return true;
		}
	}
	return false;
}

// TODO It is unclear yet if this API will stay. I have a feeling it might consume a lot of CPU
void VoxelTerrain::notify_data_block_enter(const VoxelDataBlock &block, Vector3i bpos, ViewerID viewer_id) {
	if (!VoxelEngine::get_singleton().viewer_exists(viewer_id)) {
		// The viewer might have been removed between the moment we requested the block and the moment we finished
		// loading it
		return;
	}
	if (_data_block_enter_info_obj == nullptr) {
		_data_block_enter_info_obj = zylann::godot::make_unique<VoxelDataBlockEnterInfo>();
	}
	const int network_peer_id = VoxelEngine::get_singleton().get_viewer_network_peer_id(viewer_id);
	_data_block_enter_info_obj->network_peer_id = network_peer_id;
	_data_block_enter_info_obj->voxel_block = block;
	_data_block_enter_info_obj->block_position = bpos;

	if (!GDVIRTUAL_CALL(_on_data_block_entered, _data_block_enter_info_obj.get()) &&
		_multiplayer_synchronizer == nullptr) {
		WARN_PRINT_ONCE("VoxelTerrain::_on_data_block_entered is unimplemented!");
	}

	if (_multiplayer_synchronizer != nullptr && !Engine::get_singleton()->is_editor_hint() &&
		network_peer_id != MultiplayerPeer::TARGET_PEER_SERVER && _multiplayer_synchronizer->is_server()) {
		_multiplayer_synchronizer->send_block(network_peer_id, block, bpos);
	}
}

void VoxelTerrain::process() {
	ZN_PROFILE_SCOPE();

#ifdef VOXEL_ENABLE_GPU
	if (get_generator_use_gpu()) {
		Ref<VoxelGenerator> generator = get_generator();
		if (generator.is_valid() && generator->supports_shaders() &&
			generator->get_block_rendering_shader() == nullptr) {
			generator->compile_shaders();
		}
	}
#endif

	{
		for (const QuickReloadingBlock &qrb : _quick_reloading_blocks) {
			VoxelEngine::BlockDataOutput ob{
				VoxelEngine::BlockDataOutput::TYPE_LOADED, //
				qrb.voxels, //
#ifdef VOXEL_ENABLE_INSTANCER
				// TODO This doesn't work with VoxelInstancer because it unloads based on meshes...
				nullptr, //
#endif
				qrb.position, //
				0, // lod_index
				false, // dropped
				false, // max_lod_hint
				false, // initial_load
				false, // had_instances
				true // had_voxels
			};
			apply_data_block_response(ob);
		}
		_quick_reloading_blocks.clear();
	}

	process_viewers();
	// process_received_data_blocks();
	process_meshing();

#ifdef TOOLS_ENABLED
	if (debug_is_draw_enabled() && is_visible_in_tree()) {
		process_debug_draw();
	}
#endif
}

void VoxelTerrain::process_viewers() {
	ProfilingClock profiling_clock;

	// Ordered by ascending index in paired viewers list
	StdVector<size_t> unpaired_viewer_indexes;

	// Sync here to make sure tasks evaluate a more up-to-date distance. Otherwise, a viewer could spawn (or teleport
	// far away), trigger tasks, but if sync still hasn't run by the time a task priority gets evaluated, the task could
	// cancel itself because "too far from viewers".
	// Not ideal since VoxelEngine already calls this, but it should be quick enough.
	// An alternative is to use explicit cancellation tokens, which are used in VLT Clipbox.
	VoxelEngine::get_singleton().sync_viewers_task_priority_data();

	// Update viewers
	{
		// Our node doesn't have bounds yet, so for now viewers are always paired.
		// TODO Update: the node has bounds now, need to change this

		// Destroyed viewers
		for (size_t i = 0; i < _paired_viewers.size(); ++i) {
			PairedViewer &p = _paired_viewers[i];
			if (!VoxelEngine::get_singleton().viewer_exists(p.id)) {
				ZN_PRINT_VERBOSE(format("Detected destroyed viewer {} in VoxelTerrain", p.id));
				// Interpret removal as nullified view distance so the same code handling loading of blocks
				// will be used to unload those viewed by this viewer.
				// We'll actually remove unpaired viewers in a second pass.
				p.state.vertical_view_distance_voxels = 0;
				p.state.horizontal_view_distance_voxels = 0;
				// Also update boxes, they won't be updated since the viewer has been removed.
				// Assign prev state, otherwise in some cases resetting boxes would make them equal to prev state,
				// therefore causing no unload
				p.prev_state = p.state;
				p.state.data_box = Box3i();
				p.state.mesh_box = Box3i();
				unpaired_viewer_indexes.push_back(i);
			}
		}

		const Transform3D local_to_world_transform = get_global_transform();
		const Transform3D world_to_local_transform = local_to_world_transform.affine_inverse();

		// Note, this does not support non-uniform scaling
		// TODO There is probably a better way to do this
		const float view_distance_scale = world_to_local_transform.basis.xform(Vector3(1, 0, 0)).length();

		const Box3i bounds_in_voxels = _data->get_bounds();

		const Box3i bounds_in_data_blocks = bounds_in_voxels.downscaled(get_data_block_size());
		const Box3i bounds_in_mesh_blocks = bounds_in_voxels.downscaled(get_mesh_block_size());

		struct UpdatePairedViewer {
			VoxelTerrain &self;
			const Box3i bounds_in_data_blocks;
			const Box3i bounds_in_mesh_blocks;
			const Transform3D world_to_local_transform;
			const float view_distance_scale;

			inline void operator()(ViewerID viewer_id, const VoxelEngine::Viewer &viewer) {
				size_t paired_viewer_index;
				if (!self.try_get_paired_viewer_index(viewer_id, paired_viewer_index)) {
					// New viewer
					PairedViewer p;
					p.id = viewer_id;
					paired_viewer_index = self._paired_viewers.size();
					self._paired_viewers.push_back(p);
					ZN_PRINT_VERBOSE(format("Pairing viewer {} to VoxelTerrain", viewer_id));
				}

				PairedViewer &paired_viewer = self._paired_viewers[paired_viewer_index];
				paired_viewer.prev_state = paired_viewer.state;
				PairedViewer::State &state = paired_viewer.state;

				const unsigned int view_distance_voxels_h = static_cast<unsigned int>(
						static_cast<float>(viewer.view_distances.horizontal) * view_distance_scale
				);
				const unsigned int view_distance_voxels_v = static_cast<unsigned int>(
						static_cast<float>(viewer.view_distances.vertical) * view_distance_scale
				);

				const Vector3 local_position = world_to_local_transform.xform(viewer.world_position);

				state.horizontal_view_distance_voxels =
						math::min(view_distance_voxels_h, self._max_view_distance_voxels);
				state.vertical_view_distance_voxels = math::min(view_distance_voxels_v, self._max_view_distance_voxels);

				state.local_position_voxels = math::floor_to_int(local_position);
				state.requires_collisions = VoxelEngine::get_singleton().is_viewer_requiring_collisions(viewer_id);
				state.requires_meshes =
						VoxelEngine::get_singleton().is_viewer_requiring_visuals(viewer_id) && self._mesher.is_valid();

				// Update data and mesh view boxes

				const int data_block_size = self.get_data_block_size();
				const int mesh_block_size = self.get_mesh_block_size();

				int view_distance_data_blocks_h;
				int view_distance_data_blocks_v;
				Vector3i data_block_pos;

				if (state.requires_meshes || state.requires_collisions) {
					const int view_distance_mesh_blocks_h =
							math::ceildiv(state.horizontal_view_distance_voxels, mesh_block_size);
					const int view_distance_mesh_blocks_v =
							math::ceildiv(state.vertical_view_distance_voxels, mesh_block_size);

					const int render_to_data_factor = (mesh_block_size / data_block_size);
					const Vector3i mesh_block_pos = math::floordiv(state.local_position_voxels, mesh_block_size);

					// Adding one block of padding because meshing requires neighbors
					view_distance_data_blocks_h = view_distance_mesh_blocks_h * render_to_data_factor + 1;
					view_distance_data_blocks_v = view_distance_mesh_blocks_v * render_to_data_factor + 1;

					data_block_pos = mesh_block_pos * render_to_data_factor;
					state.mesh_box = Box3i::from_center_extents(
											 mesh_block_pos,
											 Vector3i(
													 view_distance_mesh_blocks_h,
													 view_distance_mesh_blocks_v,
													 view_distance_mesh_blocks_h
											 )
					)
											 .clipped(bounds_in_mesh_blocks);

				} else {
					view_distance_data_blocks_h = math::ceildiv(state.horizontal_view_distance_voxels, data_block_size);
					view_distance_data_blocks_v = math::ceildiv(state.vertical_view_distance_voxels, data_block_size);

					data_block_pos = math::floordiv(state.local_position_voxels, data_block_size);
					state.mesh_box = Box3i();
				}

				state.data_box = Box3i::from_center_extents(
										 data_block_pos,
										 Vector3i(
												 view_distance_data_blocks_h,
												 view_distance_data_blocks_v,
												 view_distance_data_blocks_h
										 )
				)
										 .clipped(bounds_in_data_blocks);
			}
		};

		// New viewers and updates. Removed viewers won't be iterated but are still paired until later.
		UpdatePairedViewer u{
			*this, bounds_in_data_blocks, bounds_in_mesh_blocks, world_to_local_transform, view_distance_scale
		};
		VoxelEngine::get_singleton().for_each_viewer(u);
	}

	const bool can_load_blocks = ((_automatic_loading_enabled &&
								   (_multiplayer_synchronizer == nullptr || _multiplayer_synchronizer->is_server())) &&
								  (get_stream().is_valid() || get_generator().is_valid())) &&
			(Engine::get_singleton()->is_editor_hint() == false || _run_stream_in_editor);

	// Find out which blocks need to appear and which need to be unloaded
	{
		ZN_PROFILE_SCOPE();

		for (size_t i = 0; i < _paired_viewers.size(); ++i) {
			const PairedViewer &viewer = _paired_viewers[i];

			{
				const Box3i &new_data_box = viewer.state.data_box;
				const Box3i &prev_data_box = viewer.prev_state.data_box;

				if (prev_data_box != new_data_box) {
					process_viewer_data_box_change(viewer.id, prev_data_box, new_data_box, can_load_blocks);
				}
			}

			{
				const Box3i &new_mesh_box = viewer.state.mesh_box;
				const Box3i &prev_mesh_box = viewer.prev_state.mesh_box;

				if (prev_mesh_box != new_mesh_box) {
					ZN_PROFILE_SCOPE();

					// TODO Any reason to unview old blocks before viewing new blocks?
					// Because if a viewer is removed and another is added, it will reload the whole area even if their
					// box is the same.

					// Unview blocks that just fell out of range
					prev_mesh_box.difference(new_mesh_box, [this, &viewer](Box3i out_of_range_box) {
						out_of_range_box.for_each_cell([this, &viewer](Vector3i bpos) {
							unview_mesh_block(
									bpos, viewer.prev_state.requires_meshes, viewer.prev_state.requires_collisions
							);
						});
					});

					// View blocks that just entered the range
					new_mesh_box.difference(prev_mesh_box, [this, &viewer](Box3i box_to_load) {
						box_to_load.for_each_cell([this, &viewer](Vector3i bpos) {
							// Load or update block
							view_mesh_block(bpos, viewer.state.requires_meshes, viewer.state.requires_collisions);
						});
					});
				}

				// Blocks that remained within range of the viewer may need some changes too if viewer flags were
				// modified. This operates on a DISTINCT set of blocks than the one above.

				if (viewer.state.requires_collisions != viewer.prev_state.requires_collisions) {
					const Box3i box = new_mesh_box.clipped(prev_mesh_box);
					if (viewer.state.requires_collisions) {
						box.for_each_cell([this](Vector3i bpos) { //
							view_mesh_block(bpos, false, true);
						});

					} else {
						box.for_each_cell([this](Vector3i bpos) { //
							unview_mesh_block(bpos, false, true);
						});
					}
				}

				if (viewer.state.requires_meshes != viewer.prev_state.requires_meshes) {
					const Box3i box = new_mesh_box.clipped(prev_mesh_box);
					if (viewer.state.requires_meshes) {
						box.for_each_cell([this](Vector3i bpos) { //
							view_mesh_block(bpos, true, false);
						});

					} else {
						box.for_each_cell([this](Vector3i bpos) { //
							unview_mesh_block(bpos, true, false);
						});
					}
				}
			}
		}
	}

	_stats.time_detect_required_blocks = profiling_clock.restart();

	// We no longer need unpaired viewers.
	for (size_t i = 0; i < unpaired_viewer_indexes.size(); ++i) {
		// Iterating backward so indexes of paired viewers that need removal will not change because of the removal
		// itself
		const size_t vi = unpaired_viewer_indexes[unpaired_viewer_indexes.size() - i - 1];
		ZN_PRINT_VERBOSE(format("Unpairing viewer {} from VoxelTerrain", _paired_viewers[vi].id));
		_paired_viewers[vi] = _paired_viewers.back();
		_paired_viewers.pop_back();
	}

	// It's possible the user didn't set a stream yet, or it is turned off
	if (can_load_blocks) {
		send_data_load_requests();
		BufferedTaskScheduler &task_scheduler = BufferedTaskScheduler::get_for_current_thread();
		consume_block_data_save_requests(task_scheduler, nullptr, false);
		task_scheduler.flush();
	}

	_stats.time_request_blocks_to_load = profiling_clock.restart();
}

void VoxelTerrain::process_viewer_data_box_change(
		const ViewerID viewer_id,
		const Box3i prev_data_box,
		const Box3i new_data_box,
		const bool can_load_blocks
) {
	ZN_PROFILE_SCOPE();
	ZN_ASSERT_RETURN(prev_data_box != new_data_box);

	static thread_local StdVector<Vector3i> tls_missing_blocks;
	static thread_local StdVector<Vector3i> tls_found_blocks_positions;

	Ref<VoxelGenerator> generator = get_generator();
	if (generator.is_valid()) {
		generator->process_viewer_diff(viewer_id, new_data_box, prev_data_box);
	}

	// Unview blocks that just fell out of range
	//
	// TODO Any reason to unview old blocks before viewing new blocks?
	// Because if a viewer is removed and another is added, it will reload the whole area even if their box is the same.
	{
		const bool may_save =
				get_stream().is_valid() && (!Engine::get_singleton()->is_editor_hint() || _run_stream_in_editor);

		tls_missing_blocks.clear();
		tls_found_blocks_positions.clear();

		const unsigned int to_save_index0 = _blocks_to_save.size();

		// Decrement refcounts from loaded blocks, and unload them
		prev_data_box.difference(new_data_box, [this, may_save](Box3i out_of_range_box) {
			// ZN_PRINT_VERBOSE(format("Unview data box {}", out_of_range_box));
			_data->unview_area(
					out_of_range_box,
					0,
					&tls_found_blocks_positions,
					&tls_missing_blocks,
					may_save ? &_blocks_to_save : nullptr
			);
		});

		// Temporarily store unloaded blocks in a map until saving completes
		for (unsigned int i = to_save_index0; i < _blocks_to_save.size(); ++i) {
			const VoxelData::BlockToSave &bts = _blocks_to_save[i];
			_unloaded_saving_blocks[bts.position] = bts.voxels;
		}

		{
			ZN_PROFILE_SCOPE_NAMED("Unload signals");
			// Remove loading blocks (those were loaded and had their refcount reach zero)
			for (const Vector3i bpos : tls_found_blocks_positions) {
				emit_data_block_unloaded(bpos);
				// TODO If they were loaded, why would they be in loading blocks?
				// Probably in case we move so fast that blocks haven't even finished loading
				_loading_blocks.erase(bpos);
			}
		}

		// Remove refcount from loading blocks, and cancel loading if it reaches zero
		{
			ZN_PROFILE_SCOPE_NAMED("Cancel missing blocks");
			for (const Vector3i bpos : tls_missing_blocks) {
				auto loading_block_it = _loading_blocks.find(bpos);
				if (loading_block_it == _loading_blocks.end()) {
					ZN_PRINT_VERBOSE("Request to unview a loading block that was never requested");
					// Not expected, but fine I guess
					return;
				}

				LoadingBlock &loading_block = loading_block_it->second;
				loading_block.viewers.remove();

				if (loading_block.viewers.get() == 0) {
					// No longer want to load it
					_loading_blocks.erase(loading_block_it);

					// TODO Do we really need that vector after all?
					for (size_t i = 0; i < _blocks_pending_load.size(); ++i) {
						if (_blocks_pending_load[i] == bpos) {
							_blocks_pending_load[i] = _blocks_pending_load.back();
							_blocks_pending_load.pop_back();
							break;
						}
					}
				}
			}
		}
	}

	// View blocks coming into range
	if (can_load_blocks) {
		const bool require_notifications =
				(_block_enter_notification_enabled ||
				 (_multiplayer_synchronizer != nullptr && _multiplayer_synchronizer->is_server())) &&
				VoxelEngine::get_singleton().viewer_exists(viewer_id) && // Could be a destroyed viewer
				VoxelEngine::get_singleton().is_viewer_requiring_data_block_notifications(viewer_id);

		static thread_local StdVector<VoxelDataBlock> tls_found_blocks;

		tls_missing_blocks.clear();
		tls_found_blocks.clear();
		tls_found_blocks_positions.clear();

		new_data_box.difference(prev_data_box, [this](Box3i box_to_load) {
			// ZN_PRINT_VERBOSE(format("View data box {}", box_to_load));
			_data->view_area(box_to_load, 0, &tls_missing_blocks, &tls_found_blocks_positions, &tls_found_blocks);
		});

		// Schedule loading of missing blocks
		{
			ZN_PROFILE_SCOPE_NAMED("Gather missing blocks");
			for (const Vector3i missing_bpos : tls_missing_blocks) {
				auto loading_block_it = _loading_blocks.find(missing_bpos);

				if (loading_block_it == _loading_blocks.end()) {
					// First viewer to request it
					LoadingBlock new_loading_block;
					new_loading_block.viewers.add();

					if (require_notifications) {
						new_loading_block.viewers_to_notify.push_back(viewer_id);
					}

					_loading_blocks.insert({ missing_bpos, new_loading_block });
					_blocks_pending_load.push_back(missing_bpos);

				} else {
					// More viewers
					LoadingBlock &loading_block = loading_block_it->second;
					loading_block.viewers.add();

					if (require_notifications) {
						loading_block.viewers_to_notify.push_back(viewer_id);
					}
				}
			}
		}

		if (require_notifications) {
			ZN_PROFILE_SCOPE_NAMED("Enter notifications");
			// Notifications for blocks that were already loaded
			for (unsigned int i = 0; i < tls_found_blocks.size(); ++i) {
				const Vector3i bpos = tls_found_blocks_positions[i];
				const VoxelDataBlock &block = tls_found_blocks[i];
				notify_data_block_enter(block, bpos, viewer_id);
			}
		}

		// Make sure to clear this because it holds refcounted stuff. If we don't, it could crash on exit because the
		// voxel engine deinitializes its stuff before thread_locals get destroyed
		tls_found_blocks.clear();

		// TODO viewers with varying flags during the game is not supported at the moment.
		// They have to be re-created, which may cause world re-load...
	}
}

void VoxelTerrain::apply_data_block_response(VoxelEngine::BlockDataOutput &ob) {
	ZN_PROFILE_SCOPE();

	// print_line(String("Receiving {0} blocks").format(varray(output.emerged_blocks.size())));

	if (ob.type == VoxelEngine::BlockDataOutput::TYPE_SAVED) {
		if (ob.dropped) {
			ERR_PRINT(String("Could not save block {0}").format(varray(ob.position)));

		} else if (ob.had_voxels) {
			// TODO What if the version that was saved is older than the one we cached here?
			// For that to be a problem, you'd have to edit a chunk, move away, move back in, edit it again, move away,
			// and have the first save complete before the second.
			// But we may consider adding version numbers, which requires adding block metadata
			_unloaded_saving_blocks.erase(ob.position);

		}
#ifdef VOXEL_ENABLE_INSTANCER
		else if (ob.had_instances && _instancer != nullptr) {
			_instancer->on_data_block_saved(ob.position, ob.lod_index);
		}
#endif
		return;
	}

	CRASH_COND(
			ob.type != VoxelEngine::BlockDataOutput::TYPE_LOADED &&
			ob.type != VoxelEngine::BlockDataOutput::TYPE_GENERATED
	);

	const Vector3i block_pos = ob.position;

	if (ob.dropped) {
		if (_loading_blocks.find(block_pos) == _loading_blocks.end()) {
			// We are no longer expecting this block, ignore
			return;
		}
		// That block was cancelled, but we are still expecting it.
		// We'll have to request it again.
		ZN_PRINT_VERBOSE(
				format("Received a block loading drop while we were still expecting it: "
					   "lod{} ({}, {}, {}), re-requesting it",
					   int(ob.lod_index),
					   ob.position.x,
					   ob.position.y,
					   ob.position.z)
		);

		++_stats.dropped_block_loads;

		_blocks_pending_load.push_back(ob.position);
		return;
	}

	LoadingBlock loading_block;
	{
		auto loading_block_it = _loading_blocks.find(block_pos);

		if (loading_block_it == _loading_blocks.end()) {
			// That block was not requested or is no longer needed, drop it.
			++_stats.dropped_block_loads;
			return;
		}

		// Using move semantics because it can contain an allocated vector
		loading_block = std::move(loading_block_it->second);

		// Now we got the block. If we still have to drop it, the cause will be an error.
		_loading_blocks.erase(loading_block_it);
	}

	ZN_ASSERT_RETURN(ob.voxels != nullptr);

	VoxelDataBlock block(ob.voxels, ob.lod_index);
	block.set_edited(ob.type == VoxelEngine::BlockDataOutput::TYPE_LOADED);
	// Viewers will be set only if the block doesn't already exist
	block.viewers = loading_block.viewers;

	if (block.has_voxels() && block.get_voxels_const().get_size() != Vector3iUtil::create(_data->get_block_size())) {
		// Voxel block size is incorrect, drop it
		ZN_PRINT_ERROR(
				format("Block is different from expected size. Expected {}, got {}",
					   Vector3iUtil::create(_data->get_block_size()),
					   block.get_voxels_const().get_size())
		);
		++_stats.dropped_block_loads;
		return;
	}

	_data->try_set_block(
			block_pos,
			block,
			[
#ifdef DEBUG_ENABLED
					block_pos
#endif
	](VoxelDataBlock &existing_block, const VoxelDataBlock &incoming_block) {
#ifdef DEBUG_ENABLED
				ZN_PRINT_VERBOSE(format("Replacing existing data block {}", block_pos));
#endif
				existing_block.set_voxels(incoming_block.get_voxels_shared());
				existing_block.set_edited(incoming_block.is_edited());
			}
	);

	emit_data_block_loaded(block_pos);

	for (unsigned int i = 0; i < loading_block.viewers_to_notify.size(); ++i) {
		const ViewerID viewer_id = loading_block.viewers_to_notify[i];
		notify_data_block_enter(block, block_pos, viewer_id);
	}

	// The block itself might not be suitable for meshing yet, but blocks surrounding it might be now
	// TODO Optimize: initial loading can hang for a while here.
	// Because lots of blocks are loaded at once, which leads to many block queries.
	try_schedule_mesh_update_from_data(
			Box3i(_data->block_to_voxel(block_pos), Vector3iUtil::create(get_data_block_size()))
	);

	// We might have requested some blocks again (if we got a dropped one while we still need them)
	// if (stream_enabled) {
	// 	send_block_data_requests();
	// }

	// if (_instancer != nullptr && ob.instances != nullptr) {
	// 	_instancer->on_data_block_loaded(ob.position, ob.lod_index, std::move(ob.instances));
	// }
}

// Sets voxel data of a block, discarding existing data if any.
// If the given block coordinates are not inside any viewer's area, this function won't do anything and return
// false. If a block is already loading or generating at this position, it will be cancelled.
bool VoxelTerrain::try_set_block_data(Vector3i position, std::shared_ptr<VoxelBuffer> &voxel_data) {
	ZN_PROFILE_SCOPE();
	ERR_FAIL_COND_V(voxel_data == nullptr, false);

	const Vector3i expected_block_size = Vector3iUtil::create(_data->get_block_size());
	ERR_FAIL_COND_V_MSG(
			voxel_data->get_size() != expected_block_size,
			false,
			String("Block size is different from expected size. "
				   "Expected {0}, got {1}")
					.format(varray(expected_block_size, voxel_data->get_size()))
	);

	// Setup viewers count intersecting with this block
	RefCount refcount;
	for (unsigned int i = 0; i < _paired_viewers.size(); ++i) {
		const PairedViewer &viewer = _paired_viewers[i];
		if (viewer.state.data_box.contains(position)) {
			refcount.add();
		}
	}

	if (refcount.get() == 0) {
		// Actually, this block is not even in range. So we may ignore it.
		// If we don't want this behavior, we could introduce a fake viewer that adds a reference to all blocks in
		// this volume as long as it is enabled?
		ZN_PRINT_VERBOSE("Trying to set a data block outside of any viewer range");
		return false;
	}

	// Cancel loading version if any
	_loading_blocks.erase(position);

	VoxelDataBlock block(voxel_data, 0);
	// TODO How to set the `edited` flag? Does it matter in use cases for this function?
	block.set_edited(true);
	block.viewers = refcount;

	// Create or update block data
	_data->try_set_block(position, block, [](VoxelDataBlock &existing_block, const VoxelDataBlock &incoming_block) {
		existing_block.set_voxels(incoming_block.get_voxels_shared());
		existing_block.set_edited(incoming_block.is_edited());
	});

	// The block itself might not be suitable for meshing yet, but blocks surrounding it might be now
	try_schedule_mesh_update_from_data(
			Box3i(_data->block_to_voxel(position), Vector3iUtil::create(get_data_block_size()))
	);

	return true;
}

bool VoxelTerrain::has_data_block(Vector3i position) const {
	return _data->has_block(position, 0);
}

struct LightQueueItem {
    RGBLight light;
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct CompareLight {
    bool operator()(const LightQueueItem& a, const LightQueueItem& b) const {
        return a.light.maxComponent() < b.light.maxComponent();
    }
};

uint32_t vector3iKey(Vector3i v) {
    // return 2147483647 + 1000 * 1000 * v.x + 1000 * v.y + v.z;
    // return 1000000000 + 1000 * 1000 * v.x + 1000 * v.y + v.z;
    return 1000 * 1000 * (v.x + 1000) + 1000 * (v.y + 1000) + (v.z + 1000);
}

std::array<RGBLight, 20*20*20> fixLightCubeSides(Vector3i blockPos, RGBLight *lightData, std::unordered_map<uint32_t, RGBLight*> *lightMap, std::unordered_set<uint32_t> *lightProcessed, bool sunlightEnabled) {
    std::array<RGBLight, 20*20*20> result; // needs to be 20x20 for 2-voxel margins on all sides

    for (int x = 0; x <= 19; ++x) {
        for (int y = 0; y <= 19; ++y) {
            for (int z = 0; z <= 19; ++z) {
                Vector3i oldVoxel{x, y, z};
                Vector3i newBlockPos{blockPos.x, blockPos.y, blockPos.z};
                int nx = x;
                int ny = y;
                int nz = z;
                if (nx <= 1) {
                    nx += 16;
                    newBlockPos.x--;
                } else if (nx >= 18) {
                    nx -= 16;
                    newBlockPos.x++;
                }
                if (ny <= 1) {
                    ny += 16;
                    newBlockPos.y--;
                } else if (ny >= 18) {
                    ny -= 16;
                    newBlockPos.y++;
                }
                if (nz <= 1) {
                    nz += 16;
                    newBlockPos.z--;
                } else if (nz >= 18) {
                    nz -= 16;
                    newBlockPos.z++;
                }

                uint32_t newBlockKey = vector3iKey(newBlockPos);
                RGBLight lightValue = sunlightEnabled ? RGBLight{255, 255, 255} : RGBLight{0, 0, 0};
				if (lightProcessed->count(newBlockKey)) {
                    RGBLight* lightArray = (*lightMap)[newBlockKey];
					lightValue = lightArray[index3D(nx - 1, ny - 1, nz - 1)]; // -1 accounts for 20x20x20 -> 19x19x19
                }

                uint32_t finalIndex = index3D(oldVoxel.x, oldVoxel.y, oldVoxel.z, 20);
                result[finalIndex] = lightValue;
            }
        }
    }

    return result;
}

class LightBlockTask : public IThreadedTask {
    public:

    std::vector<LightQueueItem> startingLight;
    bool compressedLight = false; // represents light coming down from everywhere from (0, 16, 0) to (16, 16, 16)
    Vector3i blockPos;
    RGBLight* lightData = nullptr;
    std::unordered_map<uint32_t, std::mutex> *pendingBlocks = nullptr;
    std::unordered_map<uint32_t, RGBLight*> *lightMap = nullptr;
	std::unordered_set<uint32_t> *lightProcessed = nullptr;
    std::optional<Vector3i> originBlock; // which block triggered this update (not set for initial updates)

    int mesh_to_data_factor = 0;
    std::shared_ptr<VoxelData> _data;
    Ref<VoxelMesher> _mesher;
    Ref<VoxelGenerator> _generator;

    std::mutex *tempLock = nullptr;
    std::vector<LightBlockTask*> *secondPassTasks = nullptr;
    std::unordered_set<Vector3i> *blocksToRemesh = nullptr;
    std::unordered_map<uint32_t, int16_t> *voxelCompressedCache = nullptr;
    std::unordered_map<uint32_t, std::shared_ptr<uint16_t>> *voxelDataCache = nullptr;
    int lightDecay = 0;
    int lightMinimum = 0;

    void run(ThreadedTaskContext &ctx) {
        uint32_t blockKey = vector3iKey(blockPos);

		Ref<VoxelMesherBlocky> mesher_blocky;
		ZN_ASSERT_MSG(zylann::godot::try_get_as(_mesher, mesher_blocky), "Expected to find VoxelMesherBlocky");

		Ref<VoxelBlockyLibraryBase> library_ref = mesher_blocky->get_library();
		ERR_FAIL_COND_MSG(library_ref.is_null(), "VoxelMesherBlocky has no library assigned");

		const blocky::BakedLibrary& baked_data = library_ref->get_baked_data();

        // only one task can operate on a block at once, this will make the thread sleep until the lock is free
        // TODO: currently crashes
        // std::lock_guard<std::mutex> lock((*pendingBlocks)[vector3iKey(blockPos)]);

        // completely synchronous execution
        std::lock_guard<std::mutex> lock(*tempLock);

        int LIGHT_FALLOFF = lightDecay;
        int LIGHT_MIN = lightMinimum;
        const int row_size = 18;
	    const int deck_size = 18 * 18;
        static const uint16_t AIR_ID = 0;

        // copy block data into a buffer if we don't already have it cached
        std::shared_ptr<uint16_t> voxelDataType;
        int16_t voxelCompressedType;
        bool voxelIsCompressed = false;
        if (voxelCompressedCache->find(blockKey) != voxelCompressedCache->end()) {
            voxelCompressedType = (*voxelCompressedCache)[blockKey];
            if (voxelCompressedType == -1) {
                voxelDataType = (*voxelDataCache)[blockKey];
            }
            voxelIsCompressed = voxelCompressedType != -1;
        } else {
            // get block data for the blocks we need
            FixedArray<std::shared_ptr<VoxelBuffer>, constants::MAX_BLOCK_COUNT_PER_REQUEST> blocks;
            const Box3i data_box = Box3i(blockPos * mesh_to_data_factor, Vector3iUtil::create(mesh_to_data_factor)).padded(1);
            bool dataSuccess = _data->get_blocks_with_voxel_data(data_box, 0, to_span(blocks));

            if (!dataSuccess) {
                return; // try to avoid crashing
            }

            uint64_t blocks_count = Vector3iUtil::get_volume_u64(data_box.size);

            VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_POOL);
            copy_block_and_neighbors(
                to_span(blocks, blocks_count),
                voxels, // output VoxelBuffer
                1, // min padding
                1, // max padding
                _mesher->get_used_channels_mask(),
                _generator,
                *_data, // all the voxel data stored in the terrain
                0, // lod_index
                blockPos,
                nullptr,
                nullptr
            );

            // get the type channel from the fetched voxels (copied from voxel_mesher_blocky.cpp)
            uint16_t compressedVoxelId;
            Span<const uint16_t> type_buffer;

            const VoxelBuffer::ChannelId channel = VoxelBuffer::CHANNEL_TYPE;
            if (voxels.get_channel_compression(channel) == VoxelBuffer::COMPRESSION_UNIFORM) {
                // All voxels have the same type.
                Span<const uint8_t> raw_channel;
                voxels.get_channel_as_bytes_read_only(channel, raw_channel); // this will return a span of 1 value
                Span<const uint16_t> value = raw_channel.reinterpret_cast_to<const uint16_t>(); // convert it

                voxelIsCompressed = true;
                compressedVoxelId = value[0];

                if (compressedVoxelId != AIR_ID) {
                    // nothing to do
					lightProcessed->insert(vector3iKey(blockPos));
                    return;
                }
            } else if (voxels.get_channel_compression(channel) != VoxelBuffer::COMPRESSION_NONE) {
                // No other form of compression is allowed
                ERR_PRINT("LightBlockTask received unsupported voxel compression");
                return;
            }

            if (voxelIsCompressed) {
                voxelCompressedType = static_cast<int16_t>(compressedVoxelId);
                (*voxelCompressedCache)[blockKey] = voxelCompressedType;
            } else {
                (*voxelCompressedCache)[blockKey] = -1;

                Span<const uint8_t> raw_channel;
                if (!voxels.get_channel_as_bytes_read_only(channel, raw_channel)) {
                    ERR_PRINT("Something wrong happened");
                    return;
                }

                // I don't know where the channel depth is specified, so assert it's uint16_t I guess...
                const VoxelBuffer::Depth channel_depth = voxels.get_channel_depth(channel);
                ZN_ASSERT(channel_depth == VoxelBuffer::DEPTH_16_BIT);

                type_buffer = raw_channel.reinterpret_cast_to<const uint16_t>();

                std::shared_ptr<uint16_t> voxelDataTypeNew(new uint16_t[18*18*18], std::default_delete<uint16_t[]>());

                for (unsigned int x = 0; x <= 17; ++x) {
                    for (unsigned int y = 0; y <= 17; ++y) {
                        for (unsigned int z = 0; z <= 17; ++z) {
                            const unsigned int voxel_index = y + x * row_size + z * deck_size;
                            const unsigned int voxel_id = type_buffer[voxel_index];
                            voxelDataTypeNew.get()[voxel_index] = voxel_id;
                        }
                    }
                }
                (*voxelDataCache)[blockKey] = voxelDataTypeNew;
                voxelDataType = voxelDataTypeNew;
            }
        }

        // make queue of subtasks for the flood fill algorithm
        std::priority_queue<LightQueueItem, std::vector<LightQueueItem>, CompareLight> tasks;
        if (compressedLight) { // entirely bright sunlight flowing down
            // continue the compression, pass the light down to one new task and stop
            if (voxelIsCompressed) {
				lightProcessed->insert(vector3iKey(blockPos));
                Vector3i newBlockPos = blockPos + Vector3i(0, -1, 0);

                RGBLight *newLightData = (*lightMap)[vector3iKey(newBlockPos)];
                LightBlockTask *task = ZN_NEW(LightBlockTask);
                // task->startingLight = nullptr;
                task->compressedLight = true;
                task->blockPos = newBlockPos;
                task->lightData = newLightData;
                task->pendingBlocks = pendingBlocks;
                task->lightMap = lightMap;
				task->lightProcessed = lightProcessed;
                task->originBlock = blockPos;
                task->mesh_to_data_factor = mesh_to_data_factor;
                task->_data = _data;
                task->_mesher = _mesher;
                task->_generator = _generator;
                task->tempLock = tempLock;
                task->blocksToRemesh = blocksToRemesh;
                task->lightDecay = lightDecay;
                task->lightMinimum = lightMinimum;
                task->voxelDataCache = voxelDataCache;
                task->voxelCompressedCache = voxelCompressedCache;

                if (originBlock) {
                    VoxelEngine::get_singleton().push_async_task(task);
                } else {
                    secondPassTasks->push_back(task);
                }
                return;
            } else {
            // expand the sunlight into tasks
                RGBLight sunlight{255, 255, 255};
                for (unsigned int i = 1; i <= 16; ++i) {
                    for (unsigned int j = 1; j <= 16; ++j) {
                        LightQueueItem item{sunlight, i, 16, j};
                        tasks.push(item);
                        lightData[index3D(item.x, item.y, item.z)] = item.light;
                    }
                }
            }
        } else { // use startingLight normally
            for (LightQueueItem item: startingLight) {
                RGBLight originalLight = lightData[index3D(item.x, item.y, item.z)];

                if (originalLight.r >= item.light.r ||
                    originalLight.b >= item.light.b ||
                    originalLight.g >= item.light.g) {
                    item.light.blendMax(originalLight);
                } else if (originalLight.r >= item.light.r &&
                    originalLight.b >= item.light.b &&
                    originalLight.g >= item.light.g) {
                    //continue; // no need for a task here
                    // TODO: disabled task culling for now
                }

                tasks.push(item);
                lightData[index3D(item.x, item.y, item.z)] = item.light;
            }
        }

        // add any other light sources (emissive blocks)
        if (!voxelIsCompressed) {
            for (unsigned int x = 1; x < 17; ++x) {
                for (unsigned int y = 1; y < 17; ++y) {
                    for (unsigned int z = 1; z < 17; ++z) {
                        const unsigned int voxel_index = y + x * row_size + z * deck_size;
                        const unsigned int voxel_id = voxelDataType.get()[voxel_index];

						ZN_ASSERT_MSG(baked_data.has_model(voxel_id), "No baked data available");
						const blocky::BakedModel& model = baked_data.models[voxel_id];

						RGBLight emissiveColor = model.light;

						if (emissiveColor.is_lucent()) {
                            RGBLight originalLight = lightData[index3D(x, y, z)];

                            if (originalLight.r >= emissiveColor.r ||
                                originalLight.b >= emissiveColor.b ||
                                originalLight.g >= emissiveColor.g) {
                                emissiveColor.blendMax(originalLight);
                            } else if (originalLight.r >= emissiveColor.r &&
                                originalLight.b >= emissiveColor.b &&
                                originalLight.g >= emissiveColor.g) {
                                //continue; // no need for a task here
                                // TODO: disabled task culling for now
                            }

                            LightQueueItem item{emissiveColor, x, y, z};
                            tasks.push(item);
                            lightData[index3D(item.x, item.y, item.z)] = item.light;
                        }
                    }
                }
            }
        }

		lightProcessed->insert(vector3iKey(blockPos));
        // we want to remesh this block
        (*blocksToRemesh).insert(blockPos);

        if (tasks.size() == 0) {
            return;
        }

        // keep track of in which directions other blocks need updating
        bool updatedDirections[6] = {false, false, false, false, false, false};

        // perform flood fill algorithm
        int iterations = 0;
        const int iterationsLimit = 999999;
        while (!tasks.empty()) {
            if (++iterations >= iterationsLimit) {
                break;
            }

            LightQueueItem task = tasks.top();
            tasks.pop();

            // check if the light here is worth continuing with or obsoleted due to better light coming from somewhere else
            RGBLight lightHere = lightData[index3D(task.x, task.y, task.z)];
            if (lightHere.r > task.light.r && lightHere.g > task.light.g && lightHere.b > task.light.b) {
                continue;
            }

            // try to propagate light to neighbours
            unsigned int testVoxels[6][3] = {
                {task.x - 1, task.y, task.z},
                {task.x + 1, task.y, task.z},
                {task.x, task.y - 1, task.z},
                {task.x, task.y + 1, task.z},
                {task.x, task.y, task.z - 1},
                {task.x, task.y, task.z + 1},
            };
            for (int dir = 0; dir < 6; ++dir) {
                unsigned int *testVoxel = testVoxels[dir];

                // check if we're totally outside of the array and continue if so
                bool outOfArray = false;
                for (int i = 0; i < 3; ++i) {
                    if (testVoxel[i] < 0 || testVoxel[i] > 17) {
                        outOfArray = true;
                        break;
                    }
                }
                if (outOfArray) {
					continue;
				}

                const unsigned int test_voxel_index = testVoxel[1] + testVoxel[0] * row_size + testVoxel[2] * deck_size;
                const unsigned int test_voxel_id = voxelIsCompressed ? voxelCompressedType : voxelDataType.get()[test_voxel_index];

                if (test_voxel_id != AIR_ID) {
                    // don't propagate light through a solid block, only through air
                    // TODO: include transparent blocks (if supported?)
                    continue;
                }

                RGBLight newLight = task.light;
                bool isMaximum = newLight.r == 255 && newLight.g == 255 && newLight.b == 255;
                if (dir == 2 && isMaximum) {
                    // special case, sunlight propagates straight down without attenuation
                } else if (dir == 3 && isMaximum) {
                    // sunlight should never propagate up, ever. important optimisation with no side effects
                    continue;
                } else {
                    newLight.dim(LIGHT_FALLOFF, LIGHT_MIN);
                }

                // any previously calculated light at the test voxel
                RGBLight existingLight = lightData[index3D(testVoxel[0], testVoxel[1], testVoxel[2])];
                bool updateNeeded = false; // if any of the components change, we need to add that voxel back to the queue
                if (newLight.r > existingLight.r) {
                    updateNeeded = true;
                }
                if (newLight.g > existingLight.g) {
                    updateNeeded = true;
                }
                if (newLight.b > existingLight.b) {
                    updateNeeded = true;
                }
                if (updateNeeded) {
                    // check if we're updating within the 1-voxel padding
                    bool outOfBounds = false;
                    for (int i = 0; i < 3; ++i) {
                        if (testVoxel[i] < 1 || testVoxel[i] > 16) {
                            outOfBounds = true;
                        }
                    }

                    newLight.blendMax(existingLight);
                    lightData[index3D(testVoxel[0], testVoxel[1], testVoxel[2])] = newLight;

                    if (outOfBounds) { // trigger a light update in the adjacent chunk
                        updatedDirections[dir] = true;

                        // keep propagating to get edges/corners
                        LightQueueItem newTask{newLight, testVoxel[0], testVoxel[1], testVoxel[2]};
                        tasks.push(newTask);
                    } else { // process this new voxel
                        LightQueueItem newTask{newLight, testVoxel[0], testVoxel[1], testVoxel[2]};
                        tasks.push(newTask);
                    }
                }
            }
        }

        Vector3i adjacentBlocks[6] = {
            Vector3i(blockPos.x - 1, blockPos.y, blockPos.z),
            Vector3i(blockPos.x + 1, blockPos.y, blockPos.z),
            Vector3i(blockPos.x, blockPos.y - 1, blockPos.z),
            Vector3i(blockPos.x, blockPos.y + 1, blockPos.z),
            Vector3i(blockPos.x, blockPos.y, blockPos.z - 1),
            Vector3i(blockPos.x, blockPos.y, blockPos.z + 1),
        };
        // create new LightBlockTasks to trigger updates for any blocks where data flowed out of bounds
        for (int i = 0; i < 6; ++i) {
            if (!updatedDirections[i]) {
				continue; // no light flowed out on this side
			}

            Vector3i newBlockPos = adjacentBlocks[i];
            uint32_t newBlockKey = vector3iKey(newBlockPos);

            if (pendingBlocks->find(newBlockKey) == pendingBlocks->end() && lightMap->find(newBlockKey) == lightMap->end()) {
				continue; // don't update this block if it's neither queued up, nor processed during a previous meshing batch
			}

            // populate startingLight for the new LightBlockTask, passing a plane of light across blocks
            std::vector<LightQueueItem> newStartingLight;
            Vector3i minNew{1, 1, 1};
            Vector3i maxNew{16, 16, 16}; // inclusive
            if (i == 0) {
                minNew.x = 1;
                maxNew.x = 1;
            } else if (i == 1) {
                minNew.x = 16;
                maxNew.x = 16;
            } else if (i == 2) {
                minNew.y = 1;
                maxNew.y = 1;
            } else if (i == 3) {
                minNew.y = 16;
                maxNew.y = 16;
            } else if (i == 4) {
                minNew.z = 1;
                maxNew.z = 1;
            } else if (i == 5) {
                minNew.z = 16;
                maxNew.z = 16;
            }
            Vector3i delta = (newBlockPos - blockPos) * 16;
            for (int x = minNew.x; x <= maxNew.x; ++x) {
                for (int y = minNew.y; y <= maxNew.y; ++y) {
                    for (int z = minNew.z; z <= maxNew.z; ++z) {
                        Vector3i oldVox = Vector3i(x, y, z);
                        Vector3i newVox = oldVox - delta; // translate to other block
                        RGBLight light = lightData[index3D(oldVox.x, oldVox.y, oldVox.z)];
                        if (light.r > 0 && light.g > 0 && light.b > 0) {
                            LightQueueItem item{
                                light,
                                static_cast<uint8_t>(newVox.x),
                                static_cast<uint8_t>(newVox.y),
                                static_cast<uint8_t>(newVox.z),
                            };
                            newStartingLight.push_back(item);
                        }
                    }
                }
            }

            RGBLight *newLightData = (*lightMap)[newBlockKey];
            LightBlockTask *task = ZN_NEW(LightBlockTask);
            task->startingLight = newStartingLight;
            task->compressedLight = false;
            task->blockPos = newBlockPos;
            task->lightData = newLightData;
            task->pendingBlocks = pendingBlocks;
            task->lightMap = lightMap;
			task->lightProcessed = lightProcessed;
            task->originBlock = blockPos;
            task->mesh_to_data_factor = mesh_to_data_factor;
            task->_data = _data;
            task->_mesher = _mesher;
            task->_generator = _generator;
            task->tempLock = tempLock;
            task->blocksToRemesh = blocksToRemesh;
            task->lightDecay = lightDecay;
            task->lightMinimum = lightMinimum;
            task->voxelDataCache = voxelDataCache;
            task->voxelCompressedCache = voxelCompressedCache;
            // task->firstPassMutex = firstPassMutex;
            // task->firstPassCV = firstPassCV;
            // task->firstPassBool = firstPassBool;

            if (originBlock) {
                VoxelEngine::get_singleton().push_async_task(task);
            } else {
                secondPassTasks->push_back(task);
            }
        }
    }
};

void VoxelTerrain::process_meshing() {
	ZN_PROFILE_SCOPE();
	ProfilingClock profiling_clock;

	_stats.dropped_block_meshs = 0;

	// Send mesh updates

	const Transform3D volume_transform = get_global_transform();
	std::shared_ptr<PriorityDependency::ViewersData> shared_viewers_data =
			VoxelEngine::get_singleton().get_shared_viewers_data_from_default_world();

	// const int used_channels_mask = get_used_channels_mask();
	const int mesh_to_data_factor = get_mesh_block_size() / get_data_block_size();

	BufferedTaskScheduler &scheduler = BufferedTaskScheduler::get_for_current_thread();

    bool performLighting = !Engine::get_singleton()->is_editor_hint() && _lighting_enabled;

    if (performLighting) {
		std::vector<Vector3i> firstPassBlocks;

		std::unordered_set<Vector3i> blocksToRemesh;

		// index these with vector3iKey(Vector3i)
		std::unordered_map<uint32_t, std::mutex> pendingBlocks;
		std::unordered_map<uint32_t, std::mutex> firstPassMutex;
		std::unordered_map<uint32_t, std::condition_variable> firstPassCV;
		std::unordered_map<uint32_t, bool> firstPassBool;
		std::unordered_map<uint32_t, std::shared_ptr<uint16_t>> voxelDataCache;
		std::unordered_map<uint32_t, int16_t> voxelCompressedCache;
		if (_blocks_pending_update.size() > 0) {
			// keep track of which blocks are pending
			for (size_t bi = 0; bi < _blocks_pending_update.size(); ++bi) {
				const Vector3i mesh_block_pos = _blocks_pending_update[bi];
				uint32_t key = vector3iKey(mesh_block_pos);
				if (_lightProcessed.count(key)) {
					// we already processed this block, don't do it again
					continue;
				}

				firstPassBlocks.push_back(mesh_block_pos);
				blocksToRemesh.insert(mesh_block_pos);
			}

			for (Vector3i blockPos: firstPassBlocks) {
				uint32_t key = vector3iKey(blockPos);
				pendingBlocks.try_emplace(key);
				firstPassMutex.try_emplace(key);
				firstPassCV.try_emplace(key);
				firstPassBool.try_emplace(false);
			}

			std::mutex tempLock; // make the threads entirely synchronous
			std::vector<LightBlockTask*> secondPassTasks;

			// process light for each block
			for (Vector3i blockPos: firstPassBlocks) {
				uint32_t key = vector3iKey(blockPos);

				bool compressedLight = false;
				std::vector<LightQueueItem> startingLight;
				if (_sunlight_enabled && blockPos.y >= _sunlight_y_level) {
					compressedLight = true; // top starts off with sunlight
				}

				if (_lightMap.find(key) == _lightMap.end()) {
					// initialise light data for that block, if it doesn't exist
					RGBLight *lightData = new RGBLight[18 * 18 * 18];
					for (int i = 0; i < 18; i++) {
						for (int j = 0; j < 18; j++) {
							for (int k = 0; k < 18; k++) {
								lightData[index3D(i, j, k)] = RGBLight{0, 0, 0};
							}
						}
					}
					_lightMap[key] = lightData;

					LightBlockTask *task = ZN_NEW(LightBlockTask);
					task->compressedLight = compressedLight;
					task->blockPos = blockPos;
					task->lightData = lightData;
					task->pendingBlocks = &pendingBlocks;
					task->lightMap = &_lightMap;
					task->lightProcessed = &_lightProcessed;
					task->mesh_to_data_factor = mesh_to_data_factor;
					task->_data = _data;
					task->_mesher = get_mesher();
					task->_generator = get_generator();
					task->tempLock = &tempLock;
					task->secondPassTasks = &secondPassTasks;
					task->blocksToRemesh = &blocksToRemesh;
					task->lightDecay = _light_decay;
					task->lightMinimum = _light_minimum;
					task->voxelDataCache = &voxelDataCache;
					task->voxelCompressedCache = &voxelCompressedCache;
					scheduler.push_main_task(task);
				}
			}

			scheduler.flush();
			VoxelEngine::get_singleton().wait_and_clear_all_tasks(false);

			// send updates from any adjacent (6-directional) blocks to those processed, using the precomputed light edge data
			Vector3i adjacentBlockOffset[6] = {
				Vector3i(-1, 0, 0),
				Vector3i(1, 0, 0),
				Vector3i(0, -1, 0),
				Vector3i(0, 1, 0),
				Vector3i(0, 0, -1),
				Vector3i(0, 0, 1),
			};
			Vector3i mins[6] = {
				Vector3i(16, 1, 1),
				Vector3i(1, 1, 1),
				Vector3i(1, 16, 1),
				Vector3i(1, 1, 1),
				Vector3i(1, 1, 16),
				Vector3i(1, 1, 1),
			};
			Vector3i maxs[6] = { // inclusive
				Vector3i(16, 16, 16),
				Vector3i(1, 16, 16),
				Vector3i(16, 16, 16),
				Vector3i(16, 1, 16),
				Vector3i(16, 16, 16),
				Vector3i(16, 16, 1),
			};
			for (Vector3i blockPos: firstPassBlocks) {
				uint32_t blockKey = vector3iKey(blockPos);
				RGBLight* lightData = _lightMap[blockKey];

				for (int d = 0; d < 6; ++d) {
					Vector3i newBlockPos = blockPos + adjacentBlockOffset[d];
					Vector3i min = mins[d];
					Vector3i max = maxs[d];
					Vector3i voxOffset = 16 * adjacentBlockOffset[d];

					uint32_t newBlockKey = vector3iKey(newBlockPos);

					if (pendingBlocks.find(newBlockKey) != pendingBlocks.end()) {
						continue; // don't trigger an update if we already processed this block in the first pass
					}

					if (_lightProcessed.count(newBlockKey) == 0) {
						continue; // don't trigger an update if the adjacent block has no light data
					}

					// initialise starting light values for the new task
					std::vector<LightQueueItem> newStartingLight;

					RGBLight* adjacentLightData; // may be unset if light is compressed
					adjacentLightData = _lightMap[newBlockKey];

					for (int x = min.x; x <= max.x; ++x) {
						for (int y = min.y; y <= max.y; ++y) {
							for (int z = min.z; z <= max.z; ++z) {
								Vector3i voxPos{x, y, z}; // position in the sampled new block
								Vector3i transformedVoxPos = voxPos + voxOffset; // voxPos transformed back to the original block

								RGBLight lightValueAdjacent = adjacentLightData[index3D(voxPos.x, voxPos.y, voxPos.z)];
								RGBLight lightValueHere = lightData[index3D(transformedVoxPos.x, transformedVoxPos.y, transformedVoxPos.z)];
								if (lightValueAdjacent.r > 0 && lightValueAdjacent.g > 0 && lightValueAdjacent.b > 0) {
									// make sure this blends rather than overwriting existing values
									if (lightValueAdjacent.r > lightValueHere.r || lightValueAdjacent.g > lightValueHere.g || lightValueAdjacent.b > lightValueHere.b) {
										lightValueHere.blendMax(lightValueAdjacent);
										LightQueueItem item{lightValueHere,
											static_cast<unsigned>(transformedVoxPos.x),
											static_cast<unsigned>(transformedVoxPos.y),
											static_cast<unsigned>(transformedVoxPos.z)
										};
										newStartingLight.push_back(item);

										uint32_t finalIndex = index3D(transformedVoxPos.x, transformedVoxPos.y, transformedVoxPos.z);
										lightData[finalIndex] = lightValueHere;
									}
								}
							}
						}
					}

					// only spawn a task if there would be more than 0 updates
					if (newStartingLight.size() > 0) {
						LightBlockTask *task = ZN_NEW(LightBlockTask);
						task->startingLight = newStartingLight;
						task->compressedLight = false;
						task->blockPos = blockPos;
						task->lightData = lightData;
						task->pendingBlocks = &pendingBlocks;
						task->lightMap = &_lightMap;
						task->lightProcessed = &_lightProcessed;
						task->originBlock = blockPos;
						task->mesh_to_data_factor = mesh_to_data_factor;
						task->_data = _data;
						task->_mesher = get_mesher();
						task->_generator = get_generator();
						task->tempLock = &tempLock;
						task->secondPassTasks = &secondPassTasks;
						task->blocksToRemesh = &blocksToRemesh;
						task->lightDecay = _light_decay;
						task->lightMinimum = _light_minimum;
						task->voxelDataCache = &voxelDataCache;
						task->voxelCompressedCache = &voxelCompressedCache;

						secondPassTasks.push_back(task);
					}
				}
			}

			for (LightBlockTask *task: secondPassTasks) {
				scheduler.push_main_task(task);
			}
			scheduler.flush();
			VoxelEngine::get_singleton().wait_and_clear_all_tasks(false);
		}

		for (const Vector3i &blockPos: blocksToRemesh) {
			// add them to _blocks_pending_update if they aren't already in the list, to stop everything from falling apart
			VoxelMeshBlockVT *mesh_block = _mesh_map.get_block(blockPos);

			// some of these blocks will already be added to the pending list, so ignore those
			if (mesh_block != nullptr && !mesh_block->is_in_update_list) {
				// from try_schedule_mesh_update
				const int render_to_data_factor = get_mesh_block_size() / get_data_block_size();

				const Box3i data_box =
						Box3i(mesh_block->position * render_to_data_factor, Vector3iUtil::create(render_to_data_factor)).padded(1);

				const bool data_available = _data->has_all_blocks_in_area(data_box, 0);

				if (data_available) { // prevents crashes
					mesh_block->is_in_update_list = true;
					_blocks_pending_update.push_back(mesh_block->position);
				}
			}
        }
    }

	for (size_t bi = 0; bi < _blocks_pending_update.size(); ++bi) {
		ZN_PROFILE_SCOPE_NAMED("Block");
		const Vector3i mesh_block_pos = _blocks_pending_update[bi];

		VoxelMeshBlockVT *mesh_block = _mesh_map.get_block(mesh_block_pos);

		// If we got here, it must have been because of scheduling an update
		ZN_ASSERT_CONTINUE(mesh_block != nullptr);
		ZN_ASSERT_CONTINUE(mesh_block->is_in_update_list);

		// Pad by 1 because meshing requires neighbors
		const Box3i data_box =
				Box3i(mesh_block_pos * mesh_to_data_factor, Vector3iUtil::create(mesh_to_data_factor)).padded(1);

#ifdef DEBUG_ENABLED
		// We must have picked up a valid data block
		{
			const Vector3i anchor_pos = data_box.position + Vector3i(1, 1, 1);
			ZN_ASSERT_CONTINUE(_data->has_block(anchor_pos, 0));
		}
#endif

		// print_line(String("DDD request {0}").format(varray(mesh_request.render_block_position.to_vec3())));
		// We'll allocate this quite often. If it becomes a problem, it should be easy to pool.
		MeshBlockTask *task = ZN_NEW(MeshBlockTask);
		task->volume_id = _volume_id;
		task->mesh_block_position = mesh_block_pos;
		task->lod_index = 0;
		task->meshing_dependency = _meshing_dependency;
		task->require_visual = mesh_block->mesh_viewers.get() > 0;
		task->collision_hint = _generate_collisions && mesh_block->collision_viewers.get() > 0;
		task->data = _data;
        task->lightingEnabled = performLighting;

        if (performLighting) {
            task->lightMinimum = _light_minimum;

            uint32_t key = vector3iKey(mesh_block_pos);
			ZN_ASSERT(_lightMap.find(key) != _lightMap.end());

			std::array<RGBLight, 20*20*20> lightDataArray = fixLightCubeSides(mesh_block_pos, _lightMap[key], &_lightMap, &_lightProcessed, _sunlight_enabled); // add extra data for adjacent blocks
			task->lightData = lightDataArray;
        }

		// This iteration order is specifically chosen to match VoxelEngine and threaded access
		_data->get_blocks_with_voxel_data(data_box, 0, to_span(task->blocks));
		task->blocks_count = Vector3iUtil::get_volume_u64(data_box.size);

#ifdef DEBUG_ENABLED
		{
			unsigned int count = 0;
			for (unsigned int i = 0; i < task->blocks_count; ++i) {
				if (task->blocks[i] != nullptr) {
					++count;
				}
			}
			// Blocks that were in the list must have been scheduled because we have data for them!
			if (count == 0) {
				ZN_PRINT_ERROR("Unexpected empty block list in meshing block task");
				ZN_DELETE(task);
				continue;
			}
		}
#endif

		init_sparse_grid_priority_dependency(
				task->priority_dependency,
				task->mesh_block_position,
				get_mesh_block_size(),
				shared_viewers_data,
				volume_transform
		);

		scheduler.push_main_task(task);

		mesh_block->is_in_update_list = false;
	}

	scheduler.flush();

	_blocks_pending_update.clear();

	_stats.time_request_blocks_to_update = profiling_clock.restart();

	// print_line(String("d:") + String::num(_dirty_blocks.size()) + String(", q:") +
	// String::num(_block_update_queue.size()));
}


void VoxelTerrain::set_lighting_enabled(bool enabled) {
	_lighting_enabled = enabled;
}

void VoxelTerrain::set_sunlight_enabled(bool enabled) {
	_sunlight_enabled = enabled;
}

void VoxelTerrain::set_sunlight_y_level(int value) {
	_sunlight_y_level = value;
}

void VoxelTerrain::set_light_decay(int decay) {
	_light_decay = std::clamp(decay, 2, 128);
}

void VoxelTerrain::set_light_minimum(int mimimum) {
	_light_minimum = std::clamp(mimimum, 1, 127);
}

void VoxelTerrain::apply_mesh_update(const VoxelEngine::BlockMeshOutput &ob) {
	ZN_PROFILE_SCOPE();
	// print_line(String("DDD receive {0}").format(varray(ob.position.to_vec3())));

	VoxelMeshBlockVT *block = _mesh_map.get_block(ob.position);
	if (block == nullptr) {
		// print_line("- no longer loaded");
		// That block is no longer loaded, drop the result
		++_stats.dropped_block_meshs;
		return;
	}

	if (ob.type == VoxelEngine::BlockMeshOutput::TYPE_DROPPED) {
		// That block is loaded, but its meshing request was dropped.
		// TODO Not sure what to do in this case, the code sending update queries has to be tweaked
		ZN_PRINT_VERBOSE("Received a block mesh drop while we were still expecting it");
		++_stats.dropped_block_meshs;
		return;
	}

	// There is a slim chance for some updates to come up just after setting the mesher to null. Avoids a crash.
	if (_mesher.is_null()) {
		++_stats.dropped_block_meshs;
		return;
	}

	Ref<ArrayMesh> mesh;
	Ref<Mesh> shadow_occluder_mesh;
	StdVector<uint16_t> material_indices;
	if (ob.visual_was_required) {
		if (ob.has_mesh_resource) {
			// The mesh was already built as part of the threaded task
			mesh = ob.mesh;
			shadow_occluder_mesh = ob.shadow_occluder_mesh;
			// It can be empty
			material_indices = std::move(ob.mesh_material_indices);
		} else {
			// Can't build meshes in threads, do it here
			material_indices.clear();
			mesh = build_mesh(
					to_span_const(ob.surfaces.surfaces),
					ob.surfaces.primitive_type,
					ob.surfaces.mesh_flags,
					material_indices
			);
			shadow_occluder_mesh = build_mesh(ob.surfaces.shadow_occluder);
		}
	}
	if (mesh.is_valid()) {
		const unsigned int surface_count = mesh->get_surface_count();
		for (unsigned int surface_index = 0; surface_index < surface_count; ++surface_index) {
			const unsigned int material_index = material_indices[surface_index];
			Ref<Material> material = _mesher->get_material_by_index(material_index);
			mesh->surface_set_material(surface_index, material);
		}
	}

	if (mesh.is_null() && block->has_mesh()) {
		// No surface anymore in this block
#ifdef VOXEL_ENABLE_INSTANCER
		if (_instancer != nullptr) {
			_instancer->on_mesh_block_exit(ob.position, ob.lod);
		}
#endif
	}
	if (ob.surfaces.surfaces.size() > 0 && mesh.is_valid() && !block->has_mesh()) {
		// TODO The mesh could come from an edited region!
		// We would have to know if specific voxels got edited, or different from the generator
		// TODO Support multi-surfaces in VoxelInstancer
#ifdef VOXEL_ENABLE_INSTANCER
		if (_instancer != nullptr) {
			_instancer->on_mesh_block_enter(
					ob.position,
					ob.lod,
					ob.surfaces.surfaces[0].arrays,
					ob.surfaces.collision_surface.submesh_vertex_end,
					ob.surfaces.collision_surface.submesh_index_end
			);
		}
#endif
	}

#ifdef TOOLS_ENABLED
	const RenderingServer::ShadowCastingSetting shadow_occluder_mode = _debug_draw_shadow_occluders
			? RenderingServer::SHADOW_CASTING_SETTING_ON
			: RenderingServer::SHADOW_CASTING_SETTING_SHADOWS_ONLY;
#endif

	block->set_mesh(
			mesh,
			get_gi_mode(),
			static_cast<RenderingServer::ShadowCastingSetting>(get_shadow_casting()),
			get_render_layers_mask(),
			shadow_occluder_mesh
#ifdef TOOLS_ENABLED
			,
			shadow_occluder_mode
#endif
	);

	if (_material_override.is_valid()) {
		block->set_material_override(_material_override);
	}

	const bool gen_collisions = _generate_collisions && block->collision_viewers.get() > 0;
	if (gen_collisions) {
		Ref<Shape3D> collision_shape = make_collision_shape_from_mesher_output(ob.surfaces, **_mesher);

		bool debug_collisions = false;
		if (is_inside_tree()) {
			const SceneTree *scene_tree = get_tree();
#if DEBUG_ENABLED
			if (collision_shape.is_valid()) {
				const Color debug_color = zylann::godot::get_shape_3d_default_color(*scene_tree);
				zylann::godot::set_shape_3d_debug_color(**collision_shape, debug_color);
			}
#endif
			debug_collisions = scene_tree->is_debugging_collisions_hint();
		}

		block->set_collision_shape(collision_shape, debug_collisions, this, _collision_margin);

		block->set_collision_layer(_collision_layer);
		block->set_collision_mask(_collision_mask);
	}

	block->set_visible(block->mesh_viewers.get() > 0);
	block->set_collision_enabled(gen_collisions);
	block->set_parent_visible(is_visible());
	block->set_parent_transform(get_global_transform());
	// TODO We don't set MESH_UP_TO_DATE anywhere, but it seems to work?
	// Can't set the state because there could be more than one update in progress. Perhaps it needs refactoring.
	// block->set_mesh_state(VoxelMeshBlockVT::MESH_UP_TO_DATE);

	if (block->is_loaded == false) {
		block->is_loaded = true;
		emit_mesh_block_entered(ob.position);
	}
}

Ref<VoxelTool> VoxelTerrain::get_voxel_tool() {
	Ref<VoxelTool> vt = memnew(VoxelToolTerrain(this));
	const int used_channels_mask = get_used_channels_mask();
	// Auto-pick first used channel
	for (int channel = 0; channel < VoxelBuffer::MAX_CHANNELS; ++channel) {
		if ((used_channels_mask & (1 << channel)) != 0) {
			vt->set_channel(VoxelBuffer::ChannelId(channel));
			break;
		}
	}
	return vt;
}

void VoxelTerrain::set_run_stream_in_editor(bool enable) {
	if (enable == _run_stream_in_editor) {
		return;
	}

	_run_stream_in_editor = enable;

	if (Engine::get_singleton()->is_editor_hint()) {
		if (_run_stream_in_editor) {
			_on_stream_params_changed();

		} else {
			// This is expected to block the main thread until the streaming thread is done.
			stop_streamer();
		}
	}
}

bool VoxelTerrain::is_stream_running_in_editor() const {
	return _run_stream_in_editor;
}

void VoxelTerrain::set_bounds(Box3i box) {
	Box3i bounds_in_voxels =
			box.clipped(Box3i::from_center_extents(Vector3i(), Vector3iUtil::create(constants::MAX_VOLUME_EXTENT)));

	const int smallest_dimension = get_data_block_size();
	bounds_in_voxels.size = math::max(bounds_in_voxels.size, Vector3iUtil::create(smallest_dimension));

	// Round to block size
	bounds_in_voxels = bounds_in_voxels.snapped(get_data_block_size());

	_data->set_bounds(bounds_in_voxels);

	const unsigned int largest_dimension =
			static_cast<unsigned int>(math::max(math::max(box.size.x, box.size.y), box.size.z));
	if (largest_dimension > MAX_VIEW_DISTANCE_FOR_LARGE_VOLUME) {
		// Cap view distance to make sure you don't accidentally blow up memory when changing parameters
		if (_max_view_distance_voxels > MAX_VIEW_DISTANCE_FOR_LARGE_VOLUME) {
			_max_view_distance_voxels = math::min(_max_view_distance_voxels, MAX_VIEW_DISTANCE_FOR_LARGE_VOLUME);
			notify_property_list_changed();
		}
	}
	// TODO Editor gizmo bounds

	update_configuration_warnings();
}

Box3i VoxelTerrain::get_bounds() const {
	return _data->get_bounds();
}

void VoxelTerrain::set_multiplayer_synchronizer(VoxelTerrainMultiplayerSynchronizer *synchronizer) {
	_multiplayer_synchronizer = synchronizer;
}

const VoxelTerrainMultiplayerSynchronizer *VoxelTerrain::get_multiplayer_synchronizer() const {
	return _multiplayer_synchronizer;
}

bool VoxelTerrain::is_area_meshed(const Box3i &box_in_voxels) const {
	// This assumes we store mesh blocks even when there is no mesh
	const Box3i mesh_box = box_in_voxels.downscaled(get_mesh_block_size());
	return mesh_box.all_cells_match([this](Vector3i bpos) {
		const VoxelMeshBlockVT *block = _mesh_map.get_block(bpos);
		return block != nullptr && block->is_loaded;
	});
}

#ifdef TOOLS_ENABLED

void VoxelTerrain::get_configuration_warnings(PackedStringArray &warnings) const {
	VoxelNode::get_configuration_warnings(warnings);

#ifdef VOXEL_ENABLE_GPU
	if (get_generator_use_gpu()) {
		Ref<VoxelGenerator> generator = get_generator();
		if (generator.is_valid() && !generator->supports_shaders()) {
			warnings.append(String("`use_gpu_generation` is enabled, but {0} does not support running on the GPU.")
									.format(varray(generator->get_class())));
		}
	}
#endif

	if (get_bounds().is_empty()) {
		warnings.append(String("Terrain bounds have an empty size."));
	}
}

#endif

void VoxelTerrain::on_format_changed() {
	_on_stream_params_changed();
}

// DEBUG LAND

void VoxelTerrain::debug_set_draw_enabled(bool enabled) {
#ifdef TOOLS_ENABLED
	_debug_draw_enabled = enabled;
	if (_debug_draw_enabled) {
		if (is_inside_tree()) {
			_debug_renderer.set_world(is_visible_in_tree() ? *get_world_3d() : nullptr);
		}
	} else {
		_debug_renderer.clear();
		// _debug_mesh_update_items.clear();
		// _debug_edit_items.clear();
	}
#endif
}

bool VoxelTerrain::debug_is_draw_enabled() const {
#ifdef TOOLS_ENABLED
	return _debug_draw_enabled;
#else
	return false;
#endif
}

void VoxelTerrain::debug_set_draw_flag(DebugDrawFlag flag_index, bool enabled) {
#ifdef TOOLS_ENABLED
	ERR_FAIL_INDEX(flag_index, DEBUG_DRAW_FLAGS_COUNT);
	if (enabled) {
		_debug_draw_flags |= (1 << flag_index);
	} else {
		_debug_draw_flags &= ~(1 << flag_index);
	}
#endif
}

bool VoxelTerrain::debug_get_draw_flag(DebugDrawFlag flag_index) const {
#ifdef TOOLS_ENABLED
	ERR_FAIL_INDEX_V(flag_index, DEBUG_DRAW_FLAGS_COUNT, false);
	return (_debug_draw_flags & (1 << flag_index)) != 0;
#else
	return false;
#endif
}

void VoxelTerrain::debug_set_draw_shadow_occluders(bool enable) {
#ifdef TOOLS_ENABLED
	if (enable == _debug_draw_shadow_occluders) {
		return;
	}
	_debug_draw_shadow_occluders = enable;
	const RenderingServer::ShadowCastingSetting mode =
			enable ? RenderingServer::SHADOW_CASTING_SETTING_ON : RenderingServer::SHADOW_CASTING_SETTING_SHADOWS_ONLY;
	_mesh_map.for_each_block([mode](VoxelMeshBlockVT &block) {
		if (block.shadow_occluder.is_valid()) {
			block.shadow_occluder.set_cast_shadows_setting(mode);
		}
	});
#endif
}

bool VoxelTerrain::debug_get_draw_shadow_occluders() const {
#ifdef TOOLS_ENABLED
	return _debug_draw_shadow_occluders;
#else
	return false;
#endif
}

#ifdef TOOLS_ENABLED

void VoxelTerrain::process_debug_draw() {
	ZN_PROFILE_SCOPE();

	zylann::godot::DebugRenderer &dr = _debug_renderer;
	dr.begin();

	const Transform3D parent_transform = get_global_transform();

	// Volume bounds
	if (debug_get_draw_flag(DEBUG_DRAW_VOLUME_BOUNDS)) {
		const Box3i bounds_in_voxels = get_bounds();
		const float bounds_in_voxels_len = Vector3(bounds_in_voxels.size).length();

		if (bounds_in_voxels_len < 10000) {
			const Vector3 margin = Vector3(1, 1, 1) * bounds_in_voxels_len * 0.0025f;
			const Vector3 size = bounds_in_voxels.size;
			const Transform3D local_transform(
					Basis().scaled(size + margin * 2.f), Vector3(bounds_in_voxels.position) - margin
			);
			dr.draw_box(parent_transform * local_transform, Color(1, 1, 1));
		}
	}

	if (debug_get_draw_flag(DEBUG_DRAW_VISUAL_AND_COLLISION_BLOCKS)) {
		const int mesh_block_size = get_mesh_block_size();
		_mesh_map.for_each_block([&parent_transform, &dr, mesh_block_size](const VoxelMeshBlockVT &block) {
			Color8 color;
			const bool visual = block.is_visible();
			const bool collision = block.is_collision_enabled();
			if (visual && collision) {
				color = Color8(255, 255, 0, 255);
			} else if (visual) {
				color = Color8(0, 255, 0, 255);
			} else if (collision) {
				color = Color8(255, 0, 0, 255);
			} else {
				return;
			}
			const Vector3i voxel_pos = block.position * mesh_block_size;
			const Transform3D local_transform(
					Basis().scaled(Vector3(mesh_block_size, mesh_block_size, mesh_block_size)), voxel_pos
			);
			const Transform3D t = parent_transform * local_transform;
			dr.draw_box(t, color);
		});
	}

	dr.end();
}

#endif

// BINDING LAND

Vector3i VoxelTerrain::_b_voxel_to_data_block(Vector3 pos) const {
	return _data->voxel_to_block(math::floor_to_int(pos));
}

Vector3i VoxelTerrain::_b_data_block_to_voxel(Vector3i pos) const {
	return _data->block_to_voxel(pos);
}

Ref<VoxelSaveCompletionTracker> VoxelTerrain::_b_save_modified_blocks() {
	std::shared_ptr<AsyncDependencyTracker> tracker = make_shared_instance<AsyncDependencyTracker>();
	save_all_modified_blocks(true, tracker);
	ZN_ASSERT_RETURN_V(tracker != nullptr, Ref<VoxelSaveCompletionTracker>());
	return VoxelSaveCompletionTracker::create(tracker);
}

// Explicitly ask to save a block if it was modified
void VoxelTerrain::_b_save_block(Vector3i p_block_pos) {
	VoxelData::BlockToSave to_save;
	if (_data->consume_block_modifications(p_block_pos, to_save)) {
		_blocks_to_save.push_back(to_save);
	}
}

void VoxelTerrain::_b_set_bounds(AABB aabb) {
	ERR_FAIL_COND(!math::is_valid_size(aabb.size));
	set_bounds(Box3i(math::round_to_int(aabb.position), math::round_to_int(aabb.size)));
}

AABB VoxelTerrain::_b_get_bounds() const {
	const Box3i b = get_bounds();
	return AABB(b.position, b.size);
}

bool VoxelTerrain::_b_try_set_block_data(Vector3i position, Ref<godot::VoxelBuffer> voxel_data) {
	ERR_FAIL_COND_V(voxel_data.is_null(), false);
	std::shared_ptr<VoxelBuffer> buffer = voxel_data->get_buffer_shared();

#ifdef DEBUG_ENABLED
	// It is not allowed to call this function at two different positions with the same voxel buffer
	const StringName &key = VoxelStringNames::get_singleton()._voxel_debug_vt_position;
	if (voxel_data->has_meta(key)) {
		const Vector3i meta_pos = voxel_data->get_meta(key);
		ERR_FAIL_COND_V_MSG(
				meta_pos != position,
				false,
				String("Setting the same {0} at different positions is not supported")
						.format(varray(godot::VoxelBuffer::get_class_static()))
		);
	} else {
		voxel_data->set_meta(key, position);
	}
#endif

	return try_set_block_data(position, buffer);
}

PackedInt32Array VoxelTerrain::_b_get_viewer_network_peer_ids_in_area(Vector3i area_origin, Vector3i area_size) const {
	static thread_local StdVector<ViewerID> s_ids;
	StdVector<ViewerID> &viewer_ids = s_ids;
	viewer_ids.clear();
	get_viewers_in_area(viewer_ids, Box3i(area_origin, area_size));

	PackedInt32Array peer_ids;
	peer_ids.resize(viewer_ids.size());
	// Using direct access because when compiling with GodotCpp the array access syntax is different, also it is a bit
	// faster
	int32_t *peer_ids_data = peer_ids.ptrw();
	ZN_ASSERT_RETURN_V(peer_ids_data != nullptr, peer_ids);
	for (size_t i = 0; i < viewer_ids.size(); ++i) {
		const int peer_id = VoxelEngine::get_singleton().get_viewer_network_peer_id(viewer_ids[i]);
		peer_ids_data[i] = peer_id;
	}

	return peer_ids;
}

bool VoxelTerrain::_b_is_area_meshed(AABB aabb) const {
	return is_area_meshed(Box3i(aabb.position, aabb.size));
}

void VoxelTerrain::_bind_methods() {
	using Self = VoxelTerrain;

	ClassDB::bind_method(D_METHOD("set_lighting_enabled", "enabled"), &Self::set_lighting_enabled);
	ClassDB::bind_method(D_METHOD("get_lighting_enabled"), &Self::get_lighting_enabled);

	ClassDB::bind_method(D_METHOD("set_sunlight_enabled", "enabled"), &Self::set_sunlight_enabled);
	ClassDB::bind_method(D_METHOD("get_sunlight_enabled"), &Self::get_sunlight_enabled);

	ClassDB::bind_method(D_METHOD("set_sunlight_y_level", "value"), &Self::set_sunlight_y_level);
	ClassDB::bind_method(D_METHOD("get_sunlight_y_level"), &Self::get_sunlight_y_level);

	ClassDB::bind_method(D_METHOD("set_light_decay", "decay"), &Self::set_light_decay);
	ClassDB::bind_method(D_METHOD("get_light_decay"), &Self::get_light_decay);

	ClassDB::bind_method(D_METHOD("set_light_minimum", "minimum"), &Self::set_light_minimum);
	ClassDB::bind_method(D_METHOD("get_light_minimum"), &Self::get_light_minimum);

	ClassDB::bind_method(D_METHOD("set_material_override", "material"), &Self::set_material_override);
	ClassDB::bind_method(D_METHOD("get_material_override"), &Self::get_material_override);

	ClassDB::bind_method(D_METHOD("set_max_view_distance", "distance_in_voxels"), &Self::set_max_view_distance);
	ClassDB::bind_method(D_METHOD("get_max_view_distance"), &Self::get_max_view_distance);

	ClassDB::bind_method(
			D_METHOD("set_block_enter_notification_enabled", "enabled"), &Self::set_block_enter_notification_enabled
	);
	ClassDB::bind_method(D_METHOD("is_block_enter_notification_enabled"), &Self::is_block_enter_notification_enabled);

	ClassDB::bind_method(
			D_METHOD("set_area_edit_notification_enabled", "enabled"), &Self::set_area_edit_notification_enabled
	);
	ClassDB::bind_method(D_METHOD("is_area_edit_notification_enabled"), &Self::is_area_edit_notification_enabled);

	ClassDB::bind_method(D_METHOD("get_generate_collisions"), &Self::get_generate_collisions);
	ClassDB::bind_method(D_METHOD("set_generate_collisions", "enabled"), &Self::set_generate_collisions);

	ClassDB::bind_method(D_METHOD("get_collision_layer"), &Self::get_collision_layer);
	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &Self::set_collision_layer);

	ClassDB::bind_method(D_METHOD("get_collision_mask"), &Self::get_collision_mask);
	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &Self::set_collision_mask);

	ClassDB::bind_method(D_METHOD("get_collision_margin"), &Self::get_collision_margin);
	ClassDB::bind_method(D_METHOD("set_collision_margin", "margin"), &Self::set_collision_margin);

	ClassDB::bind_method(D_METHOD("voxel_to_data_block", "voxel_pos"), &Self::_b_voxel_to_data_block);
	ClassDB::bind_method(D_METHOD("data_block_to_voxel", "block_pos"), &Self::_b_data_block_to_voxel);

	ClassDB::bind_method(D_METHOD("get_data_block_size"), &Self::get_data_block_size);

	ClassDB::bind_method(D_METHOD("get_mesh_block_size"), &Self::get_mesh_block_size);
	ClassDB::bind_method(D_METHOD("set_mesh_block_size", "size"), &Self::set_mesh_block_size);

	ClassDB::bind_method(D_METHOD("get_statistics"), &Self::_b_get_statistics);
	ClassDB::bind_method(D_METHOD("get_voxel_tool"), &Self::get_voxel_tool);

	ClassDB::bind_method(D_METHOD("save_modified_blocks"), &Self::_b_save_modified_blocks);
	ClassDB::bind_method(D_METHOD("save_block", "position"), &Self::_b_save_block);

	ClassDB::bind_method(D_METHOD("set_run_stream_in_editor", "enable"), &Self::set_run_stream_in_editor);
	ClassDB::bind_method(D_METHOD("is_stream_running_in_editor"), &Self::is_stream_running_in_editor);

	ClassDB::bind_method(D_METHOD("set_automatic_loading_enabled", "enable"), &Self::set_automatic_loading_enabled);
	ClassDB::bind_method(D_METHOD("is_automatic_loading_enabled"), &Self::is_automatic_loading_enabled);

#ifdef VOXEL_ENABLE_GPU
	ClassDB::bind_method(D_METHOD("set_generator_use_gpu", "enable"), &Self::set_generator_use_gpu);
	ClassDB::bind_method(D_METHOD("get_generator_use_gpu"), &Self::get_generator_use_gpu);
#endif

	// TODO Rename `_voxel_bounds`
	ClassDB::bind_method(D_METHOD("set_bounds", "bounds"), &Self::_b_set_bounds);
	ClassDB::bind_method(D_METHOD("get_bounds"), &Self::_b_get_bounds);

	ClassDB::bind_method(D_METHOD("try_set_block_data", "position", "voxels"), &Self::_b_try_set_block_data);

	ClassDB::bind_method(
			D_METHOD("get_viewer_network_peer_ids_in_area", "area_origin", "area_size"),
			&Self::_b_get_viewer_network_peer_ids_in_area
	);

	ClassDB::bind_method(D_METHOD("has_data_block", "block_position"), &Self::has_data_block);
	ClassDB::bind_method(D_METHOD("is_area_meshed", "area_in_voxels"), &Self::_b_is_area_meshed);

	ClassDB::bind_method(D_METHOD("debug_set_draw_enabled", "enabled"), &Self::debug_set_draw_enabled);
	ClassDB::bind_method(D_METHOD("debug_is_draw_enabled"), &Self::debug_is_draw_enabled);
	ClassDB::bind_method(D_METHOD("debug_set_draw_flag", "flag_index", "enabled"), &Self::debug_set_draw_flag);
	ClassDB::bind_method(D_METHOD("debug_get_draw_flag", "flag_index"), &Self::debug_get_draw_flag);

	ClassDB::bind_method(
			D_METHOD("debug_set_draw_shadow_occluders", "enabled"), &Self::debug_set_draw_shadow_occluders
	);
	ClassDB::bind_method(D_METHOD("debug_get_draw_shadow_occluders"), &Self::debug_get_draw_shadow_occluders);

#ifdef ZN_GODOT
	GDVIRTUAL_BIND(_on_data_block_entered, "info");
	GDVIRTUAL_BIND(_on_area_edited, "area_origin", "area_size");
#endif

	ADD_GROUP("Flood Fill Lighting", "");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lighting_enabled"), "set_lighting_enabled", "get_lighting_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sunlight_enabled"), "set_sunlight_enabled", "get_sunlight_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sunlight_y_level"), "set_sunlight_y_level", "get_sunlight_y_level");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_decay"), "set_light_decay", "get_light_decay");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "light_minimum"), "set_light_minimum", "get_light_minimum");

	ADD_GROUP("Bounds", "");

	ADD_PROPERTY(PropertyInfo(Variant::AABB, "bounds"), "set_bounds", "get_bounds");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_view_distance"), "set_max_view_distance", "get_max_view_distance");

	ADD_GROUP("Collisions", "");

	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "generate_collisions"), "set_generate_collisions", "get_generate_collisions"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS),
			"set_collision_layer",
			"get_collision_layer"
	);
	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS),
			"set_collision_mask",
			"get_collision_mask"
	);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_margin"), "set_collision_margin", "get_collision_margin");

	ADD_GROUP("Materials", "");

	ADD_PROPERTY(
			PropertyInfo(
					Variant::OBJECT,
					"material_override",
					PROPERTY_HINT_RESOURCE_TYPE,
					zylann::godot::MATERIAL_3D_PROPERTY_HINT_STRING
			),
			"set_material_override",
			"get_material_override"
	);

	ADD_GROUP("Networking", "");

	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "block_enter_notification_enabled"),
			"set_block_enter_notification_enabled",
			"is_block_enter_notification_enabled"
	);

	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "area_edit_notification_enabled"),
			"set_area_edit_notification_enabled",
			"is_area_edit_notification_enabled"
	);

	// This may be set to false in multiplayer designs where the server is the one sending the blocks
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "automatic_loading_enabled"),
			"set_automatic_loading_enabled",
			"is_automatic_loading_enabled"
	);

	ADD_GROUP("Advanced", "");

	// TODO Should probably be in the parent class?
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "run_stream_in_editor"),
			"set_run_stream_in_editor",
			"is_stream_running_in_editor"
	);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_block_size"), "set_mesh_block_size", "get_mesh_block_size");
#ifdef VOXEL_ENABLE_GPU
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_gpu_generation"), "set_generator_use_gpu", "get_generator_use_gpu");
#endif

	ADD_GROUP("Debug", "debug_");

	// Debug drawing is not persistent

	BIND_ENUM_CONSTANT(DEBUG_DRAW_VOLUME_BOUNDS);
	BIND_ENUM_CONSTANT(DEBUG_DRAW_FLAGS_COUNT);

	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "debug_draw_enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR),
			"debug_set_draw_enabled",
			"debug_is_draw_enabled"
	);

#define ADD_DEBUG_DRAW_FLAG(m_name, m_flag)                                                                            \
	ADD_PROPERTYI(                                                                                                     \
			PropertyInfo(Variant::BOOL, m_name, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR),                        \
			"debug_set_draw_flag",                                                                                     \
			"debug_get_draw_flag",                                                                                     \
			m_flag                                                                                                     \
	);

	ADD_DEBUG_DRAW_FLAG("debug_draw_volume_bounds", DEBUG_DRAW_VOLUME_BOUNDS);
	ADD_DEBUG_DRAW_FLAG("debug_draw_visual_and_collision_blocks", DEBUG_DRAW_VISUAL_AND_COLLISION_BLOCKS);

	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "debug_draw_shadow_occluders", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR),
			"debug_set_draw_shadow_occluders",
			"debug_get_draw_shadow_occluders"
	);

	// TODO Add back access to block, but with an API securing multithreaded access
	ADD_SIGNAL(MethodInfo("block_loaded", PropertyInfo(Variant::VECTOR3I, "position")));
	ADD_SIGNAL(MethodInfo("block_unloaded", PropertyInfo(Variant::VECTOR3I, "position")));

	ADD_SIGNAL(MethodInfo("mesh_block_entered", PropertyInfo(Variant::VECTOR3I, "position")));
	ADD_SIGNAL(MethodInfo("mesh_block_exited", PropertyInfo(Variant::VECTOR3I, "position")));
}

} // namespace zylann::voxel
