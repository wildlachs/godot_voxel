#ifndef VOXEL_GRAPH_SHADER_GENERATOR_H
#define VOXEL_GRAPH_SHADER_GENERATOR_H

#include "../../util/containers/span.h"
#include "../../util/containers/std_vector.h"
#include "../../util/errors.h"
#include "../../util/godot/core/variant.h"
#include "../../util/string/std_string.h"
#include "code_gen_helper.h"
#include "voxel_graph_function.h"
#include "voxel_graph_runtime.h"

namespace zylann::voxel::pg {

// Generates GLSL code from the given graph.
CompilationResult generate_shader(
		const ProgramGraph &p_graph,
		Span<const VoxelGraphFunction::Port> input_defs,
		FwdMutableStdString source_code,
		StdVector<ShaderParameter> &uniforms,
		StdVector<ShaderOutput> &outputs,
		Span<const VoxelGraphFunction::NodeTypeID> restricted_outputs
);

// Sent as argument to functions implementing generator nodes, in order to generate shader code.
class ShaderGenContext {
public:
	ShaderGenContext(
			const StdVector<Variant> &params,
			Span<const char *> input_names,
			Span<const char *> output_names,
			CodeGenHelper &code_gen,
			StdVector<ShaderParameter> &uniforms
	) :
			_params(params),
			_input_names(input_names),
			_output_names(output_names),
			_code_gen(code_gen),
			_uniforms(uniforms) {}

	Variant get_param(size_t i) const {
		ZN_ASSERT(i < _params.size());
		return _params[i];
	}

	const char *get_input_name(unsigned int i) const {
		return _input_names[i];
	}

	const char *get_output_name(unsigned int i) const {
		return _output_names[i];
	}

	void make_error(String message) {
		_error_message = message;
		_has_error = true;
	}

	bool has_error() const {
		return _has_error;
	}

	const String &get_error_message() const {
		return _error_message;
	}

	template <typename... TN>
	void add_format(const char *fmt, const TN &...an) {
		_code_gen.add_format(fmt, an...);
	}

	void require_lib_code(const char *lib_name, const char *code);
	// If the code is too long for a string constant, it can be provided as a list of strings
	void require_lib_code(const char *lib_name, const char **code);

	StdString add_uniform(std::shared_ptr<ComputeShaderResource> res);

private:
	const StdVector<Variant> &_params;
	Span<const char *> _input_names;
	Span<const char *> _output_names;
	CodeGenHelper &_code_gen;
	String _error_message;
	bool _has_error = false;
	StdVector<ShaderParameter> &_uniforms;
};

typedef void (*ShaderGenFunc)(ShaderGenContext &);

} // namespace zylann::voxel::pg

#endif // VOXEL_GRAPH_SHADER_GENERATOR_H
