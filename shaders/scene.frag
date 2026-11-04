#version 450

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_color;

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

vec3 blinnPhong(vec3 normal, vec3 view_direction, vec3 light_direction, vec3 radiance) {
	float n_dot_l = max(dot(normal, light_direction), 0.0);

	vec3 half_vector = normalize(light_direction + view_direction);
	float n_dot_h = max(dot(normal, half_vector), 0.0);

	float specular_factor = 0.0;
	if (n_dot_l > 0.0) {
		specular_factor = pow(n_dot_h, model_uniforms.alpha);
	}

	vec3 diffuse = model_uniforms.albedo.rgb * n_dot_l;
	vec3 specular = model_uniforms.specular.rgb * specular_factor;

	return radiance * (diffuse + specular);
}

void main() {
	vec3 normal = normalize(in_world_normal);
	vec3 view_direction = normalize(scene_uniforms.camera_position.xyz - in_world_position);

	vec3 color = scene_uniforms.ambient_light.rgb * model_uniforms.albedo.rgb;

	vec3 to_light = normalize(-scene_uniforms.directional_light_direction.xyz);
	color += blinnPhong(normal, view_direction, to_light, scene_uniforms.directional_light_color.rgb);

	out_color = vec4(color, 1.0);
}
