#include "voxel_graph_node_dialog.h"
#include "../../constants/voxel_string_names.h"
#include "../../generators/graph/node_type_db.h"
#include "../../util/containers/std_unordered_set.h"
#include "../../util/godot/classes/button.h"
#include "../../util/godot/classes/display_server.h"
#include "../../util/godot/classes/editor_file_dialog.h"
#include "../../util/godot/classes/editor_quick_open.h"
#include "../../util/godot/classes/font.h"
#include "../../util/godot/classes/input_event_key.h"
#include "../../util/godot/classes/label.h"
#include "../../util/godot/classes/line_edit.h"
#include "../../util/godot/classes/object.h"
#include "../../util/godot/classes/rich_text_label.h"
#include "../../util/godot/classes/tree.h"
#include "../../util/godot/classes/v_box_container.h"
#include "../../util/godot/classes/v_split_container.h"
#include "../../util/godot/core/array.h"
#include "../../util/godot/core/keyboard.h"
#include "../../util/godot/core/string.h"
#include "../../util/godot/editor_scale.h"
#include "graph_nodes_doc_data.h"

#ifdef ZN_GODOT
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR >= 4
#include <editor/editor_node.h>
#endif
#endif

namespace zylann::voxel {

namespace {

const GraphNodesDocData::Node *get_graph_node_documentation(String name) {
	for (unsigned int i = 0; i < GraphNodesDocData::COUNT; ++i) {
		const GraphNodesDocData::Node &node = GraphNodesDocData::g_data[i];
		if (node.name == name) {
			return &node;
		}
	}
	return nullptr;
}

void get_graph_node_documentation_category_names(StdVector<String> &out_category_names) {
	StdUnorderedSet<String> categories;
	for (unsigned int i = 0; i < GraphNodesDocData::COUNT; ++i) {
		const GraphNodesDocData::Node &node = GraphNodesDocData::g_data[i];
		if (categories.insert(node.category).second) {
			out_category_names.push_back(node.category);
		}
	}
}

// This is a dumbed down re-implementation of `Tree::_up` because this stuff is not exposed...
void select_up(Tree &tree) {
	TreeItem *selected_item = tree.get_selected();

	if (selected_item == nullptr) {
		ZN_PRINT_VERBOSE("No item selected in tree, can't select down");
		return;
	}

	TreeItem *prev = selected_item->get_prev_visible();

	const int col = 0;
	while (prev != nullptr && !prev->is_selectable(col)) {
		prev = prev->get_prev_visible();
	}
	if (prev == nullptr) {
		return;
	}

	prev->select(col);

	tree.ensure_cursor_is_visible();
	// tree.accept_event();
}

// This is a dumbed down re-implementation of `Tree::_down` because this stuff is not exposed...
void select_down(Tree &tree) {
	TreeItem *selected_item = tree.get_selected();

	if (selected_item == nullptr) {
		ZN_PRINT_VERBOSE("No item selected in tree, can't select down");
		return;
	}

	TreeItem *next = selected_item->get_next_visible();

	const int col = 0;

	while (next != nullptr && !next->is_selectable(col)) {
		next = next->get_next_visible();
	}
	if (next == nullptr) {
		return;
	}

	next->select(col);

	tree.ensure_cursor_is_visible();
	// tree.accept_event();
}

} // namespace

const char *VoxelGraphNodeDialog::SIGNAL_NODE_SELECTED = "node_selected";
const char *VoxelGraphNodeDialog::SIGNAL_FILE_SELECTED = "file_selected";

VoxelGraphNodeDialog::VoxelGraphNodeDialog() {
	set_title(ZN_TTR("Create Graph Node"));
	set_exclusive(false);

	set_ok_button_text(ZN_TTR("Create"));
	get_ok_button()->connect("pressed", callable_mp(this, &VoxelGraphNodeDialog::on_ok_pressed));
	get_ok_button()->set_disabled(true);
	// connect("canceled", callable_mp(this, &VisualShaderEditor::_member_cancel));

	VBoxContainer *vb_container = memnew(VBoxContainer);
	vb_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	LineEdit *filter_line_edit = memnew(LineEdit);
	filter_line_edit->connect("text_changed", callable_mp(this, &VoxelGraphNodeDialog::on_filter_text_changed));
	filter_line_edit->connect("gui_input", callable_mp(this, &VoxelGraphNodeDialog::on_filter_gui_input));
	filter_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	filter_line_edit->set_placeholder(ZN_TTR("Search"));
	vb_container->add_child(filter_line_edit);
	_filter_line_edit = filter_line_edit;

	const float editor_scale = EDSCALE;

	VSplitContainer *vsplit_container = memnew(VSplitContainer);
	vsplit_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	Tree *tree = memnew(Tree);
	tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tree->set_hide_root(true);
	tree->set_allow_reselect(true);
	tree->set_hide_folding(false);
	tree->set_custom_minimum_size(Size2(180 * editor_scale, 200 * editor_scale));
	tree->connect("item_activated", callable_mp(this, &VoxelGraphNodeDialog::on_tree_item_activated));
	tree->connect("item_selected", callable_mp(this, &VoxelGraphNodeDialog::on_tree_item_selected));
	tree->connect("nothing_selected", callable_mp(this, &VoxelGraphNodeDialog::on_tree_nothing_selected));
	vsplit_container->add_child(tree);
	_tree = tree;

	RichTextLabel *description_label = memnew(RichTextLabel);
	description_label->set_v_size_flags(Control::SIZE_FILL);
	description_label->set_custom_minimum_size(Size2(0, 70 * editor_scale));
	description_label->set_use_bbcode(true);
	description_label->connect(
			"meta_clicked", callable_mp(this, &VoxelGraphNodeDialog::on_description_label_meta_clicked)
	);
	vsplit_container->add_child(description_label);
	_description_label = description_label;

	vb_container->add_child(vsplit_container);

	add_child(vb_container);

	_function_file_dialog = memnew(EditorFileDialog);
	_function_file_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
	_function_file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	// TODO Usability: there is no way to limit a file dialog to a specific TYPE of resource, only file extensions. So
	// it's not useful because text resources are almost all using `.tres`...
	_function_file_dialog->add_filter("*.tres", ZN_TTR("Text Resource"));
	_function_file_dialog->add_filter("*.res", ZN_TTR("Binary Resource"));
	_function_file_dialog->connect(
			"file_selected", callable_mp(this, &VoxelGraphNodeDialog::on_function_file_dialog_file_selected)
	);
	add_child(_function_file_dialog);

	// TODO Replace QuickOpen with listing of project functions directly in the dialog
	// TODO GDX: EditorQuickOpen is not exposed to extensions
#ifdef ZN_GODOT
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
	_function_quick_open_dialog = memnew(EditorQuickOpen);
	_function_quick_open_dialog->connect(
			"quick_open", callable_mp(this, &VoxelGraphNodeDialog::on_function_quick_open_dialog_quick_open)
	);
	add_child(_function_quick_open_dialog);
#endif
#endif

	// In this editor, categories come from the documentation and may be unrelated to internal node categories.
	// They serve different purposes.
	get_graph_node_documentation_category_names(_category_names);
	{
		SortArray<String> sorter;
		sorter.sort(_category_names.data(), _category_names.size());
	}

	// TODO Usability: have CustomInput and CustomOutput subcategories based on I/O definitions, + a "new" option for
	// unbound
	const pg::NodeTypeDB &type_db = pg::NodeTypeDB::get_singleton();
	for (int type_index = 0; type_index < type_db.get_type_count(); ++type_index) {
		const pg::NodeType &type = type_db.get_type(type_index);

		int category_index = -1;
		String description;

		const GraphNodesDocData::Node *doc = get_graph_node_documentation(type.name);
		if (doc != nullptr) {
			for (unsigned int i = 0; i < _category_names.size(); ++i) {
				if (_category_names[i] == doc->category) {
					category_index = i;
					break;
				}
			}
			description = doc->description;
		}

		if (type_index == pg::VoxelGraphFunction::NODE_FUNCTION) {
			{
				Item item;
				item.name = ZN_TTR("Browse Custom Function...");
				item.description = description;
				item.category = category_index;
				item.id = ID_FUNCTION_BROWSE;
				_items.push_back(item);
			}
			// TODO GDX: EditorQuickOpen is not exposed to extensions
#ifdef ZN_GODOT
			{
				Item item;
				item.name = ZN_TTR("Quick Open Custom Function...");
				item.description = description;
				item.category = category_index;
				item.id = ID_FUNCTION_QUICK_OPEN;
				_items.push_back(item);
			}
#endif
			continue;
		}

		const pg::NodeType &node_type = pg::NodeTypeDB::get_singleton().get_type(type_index);

		Item item;
		item.name = ZN_TTR(node_type.name);
		item.category = category_index;
		item.description = description;
		item.id = type_index;
		_items.push_back(item);
	}

	update_tree(false);
}

void VoxelGraphNodeDialog::popup_at_screen_position(Vector2 screen_pos) {
	// Similar popup as in VisualShaderEditor::_show_members_dialog

	VoxelGraphNodeDialog &dialog = *this;

	dialog.set_position(screen_pos);

	dialog.popup();

	_filter_line_edit->call_deferred(VoxelStringNames::get_singleton().grab_focus); // Still not visible.
	_filter_line_edit->select_all();

	// Keep within screen bounds.
	// Seems we also have to do this after showing the window because Godot is unable to update its size
	// without making it visible first...
	// TODO Shouldn't we check for screen size instead of window?
	const Rect2 window_rect =
			Rect2(DisplayServer::get_singleton()->window_get_position(),
				  DisplayServer::get_singleton()->window_get_size());
	const Rect2 dialog_rect = Rect2(dialog.get_position(), get_size());
	const Vector2 difference = (dialog_rect.get_end() - window_rect.get_end()).max(Vector2());
	dialog.set_position(dialog.get_position() - difference);
}

void VoxelGraphNodeDialog::update_tree(bool autoselect) {
	_tree->clear();
	TreeItem *root = _tree->create_item();

	// Filter items

	const String filter = _filter_line_edit->get_text().strip_edges();
	const bool use_filter = !filter.is_empty();

	StdVector<unsigned int> filtered_items;

	for (unsigned int i = 0; i < _items.size(); ++i) {
		const Item &item = _items[i];
		if (!use_filter || item.name.findn(filter) != -1) {
			filtered_items.push_back(i);
		}
	}

	// Populate tree

	StdVector<TreeItem *> category_tree_items;
	category_tree_items.resize(_category_names.size(), nullptr);

	bool autoselected = false;

	for (const unsigned int item_index : filtered_items) {
		const Item &item = _items[item_index];

		TreeItem *parent_tree_item = root;

		if (item.category != -1) {
			parent_tree_item = category_tree_items[item.category];

			if (parent_tree_item == nullptr) {
				parent_tree_item = _tree->create_item(root);
				parent_tree_item->set_text(0, _category_names[item.category]);
				parent_tree_item->set_selectable(0, false);
				parent_tree_item->set_collapsed(!use_filter);

				category_tree_items[item.category] = parent_tree_item;
			}
		}

		TreeItem *tree_item = _tree->create_item(parent_tree_item);
		tree_item->set_text(0, item.name);
		if (autoselected == false && autoselect) {
			tree_item->select(0);
			autoselected = true;
		}

		tree_item->set_metadata(0, item.id);
	}
}

void VoxelGraphNodeDialog::on_filter_text_changed(String new_text) {
	update_tree(true);
}

void VoxelGraphNodeDialog::on_filter_gui_input(Ref<InputEvent> event) {
	Ref<InputEventKey> key_event = event;
	if (key_event.is_valid()) {
		// TODO GDX: `Control::gui_input()` is not exposed to GDExtension, can't forward events.
		// This is what `VisualShaderEditor::_sbox_input` does.
		// `_gui_input` is exposed, but that's only for user implementation, not for calling directly...
		// Also `gui_input` is a name already taken in GDScript, so if it gets exposed some day, it will need a
		// different name.
		//
		// _tree->gui_input(key_event);

		if (key_event->is_pressed()) {
			switch (key_event->get_keycode()) {
				case ::godot::KEY_UP:
					select_up(*_tree);
					_filter_line_edit->accept_event();
					break;

				case ::godot::KEY_DOWN:
					select_down(*_tree);
					_filter_line_edit->accept_event();
					break;

				case ::godot::KEY_ENTER:
					on_tree_item_activated();
					break;

				default:
					break;
			}
		}
	}
}

void VoxelGraphNodeDialog::on_tree_item_activated() {
	const TreeItem *item = _tree->get_selected();
	if (item == nullptr) {
		return;
	}
	const int id = item->get_metadata(0);
	ZN_ASSERT_RETURN(id >= 0);

	if (id < pg::VoxelGraphFunction::NODE_TYPE_COUNT) {
		// Node selected
		emit_signal(SIGNAL_NODE_SELECTED, id);
		hide();

	} else if (id == ID_FUNCTION_BROWSE) {
		// Browse function nodes
		zylann::godot::popup_file_dialog(*_function_file_dialog);

	} else if (id == ID_FUNCTION_QUICK_OPEN) {
#ifdef ZN_GODOT
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
		// Quick open function nodes
		_function_quick_open_dialog->popup_dialog(pg::VoxelGraphFunction::get_class_static());
#else
		Vector<StringName> base_types;
		base_types.append(pg::VoxelGraphFunction::get_class_static());
		EditorQuickOpenDialog *quick_open_dialog = EditorNode::get_singleton()->get_quick_open_dialog();
		quick_open_dialog->popup_dialog(
				base_types, callable_mp(this, &VoxelGraphNodeDialog::on_function_quick_open_dialog_item_selected)
		);
#endif
#endif

	} else {
		WARN_PRINT(String("Unknown ID {} picked in {}").format(varray(id, get_class())));
	}
}

void VoxelGraphNodeDialog::on_tree_item_selected() {
	const TreeItem *tree_item = _tree->get_selected();
	if (tree_item == nullptr) {
		get_ok_button()->set_disabled(true);
		return;
	}
	const int id = tree_item->get_metadata(0);
	ZN_ASSERT_RETURN(id >= 0);

	const Item *item = nullptr;
	for (const Item &i : _items) {
		if (i.id == id) {
			item = &i;
			break;
		}
	}
	ZN_ASSERT_RETURN(item != nullptr);

	_description_label->set_text(item->description);

	if (id < pg::VoxelGraphFunction::NODE_TYPE_COUNT) {
		get_ok_button()->set_disabled(false);

	} else {
		get_ok_button()->set_disabled(true);
	}
}

void VoxelGraphNodeDialog::on_tree_nothing_selected() {}

void VoxelGraphNodeDialog::on_ok_pressed() {
	on_tree_item_activated();
	// hide();
}

void VoxelGraphNodeDialog::on_function_file_dialog_file_selected(String fpath) {
	emit_signal(SIGNAL_FILE_SELECTED, fpath);
	hide();
}

#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
void VoxelGraphNodeDialog::on_function_quick_open_dialog_quick_open() {
#ifdef ZN_GODOT
	String fpath = _function_quick_open_dialog->get_selected();
	on_function_quick_open_dialog_item_selected(fpath);
#endif
}

#endif

void VoxelGraphNodeDialog::on_function_quick_open_dialog_item_selected(String fpath) {
#ifdef ZN_GODOT
	if (fpath.is_empty()) {
		return;
	}
	emit_signal(SIGNAL_FILE_SELECTED, fpath);
	hide();
#endif
}

void VoxelGraphNodeDialog::on_description_label_meta_clicked(Variant meta) {
	// TODO Open docs if a class name is clicked
}

void VoxelGraphNodeDialog::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		_filter_line_edit->set_clear_button_enabled(true);

	} else if (p_what == NOTIFICATION_THEME_CHANGED) {
		const VoxelStringNames &sn = VoxelStringNames::get_singleton();
		_filter_line_edit->set_right_icon(get_theme_icon(sn.Search, sn.EditorIcons));

		const Ref<Font> mono_font = get_theme_font(sn.source, sn.EditorFonts);
		if (mono_font.is_valid()) {
			_description_label->add_theme_font_override("mono_font", mono_font);
		}
	}
}

void VoxelGraphNodeDialog::_bind_methods() {
	ADD_SIGNAL(MethodInfo(SIGNAL_NODE_SELECTED, PropertyInfo(Variant::INT, "node_type_id")));
	ADD_SIGNAL(MethodInfo(SIGNAL_FILE_SELECTED, PropertyInfo(Variant::STRING, "file_path")));
}

} // namespace zylann::voxel
