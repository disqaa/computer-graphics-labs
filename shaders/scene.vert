#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec4 camera_position;
	vec4 directional_light_direction;
	vec4 directional_light_color;
	vec4 ambient_light;
	uint point_light_count;
	uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
	mat4 model;
	vec4 albedo;
	vec4 specular;
	float alpha;
} model_uniforms;

void main() {
	vec4 world_position = model_uniforms.model * vec4(in_position, 1.0);
	mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));

	out_world_position = world_position.xyz;
	out_world_normal = normal_matrix * in_normal;

	gl_Position = scene_uniforms.view_projection * world_position;
}
