#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

#include "geometry.hpp"

namespace application {

	namespace {

		struct SceneUniforms {
			glm::mat4 view_projection;
			glm::vec4 camera_position;
			glm::vec4 directional_light_direction;
			glm::vec4 directional_light_color;
			glm::vec4 ambient_light;
			uint32_t point_light_count;
			uint32_t spot_light_count;
			uint32_t padding[2];
		};
		static_assert(sizeof(SceneUniforms) == 144, "SceneUniforms must match std140 layout");

		struct ModelUniform {
			glm::mat4 model;
			glm::vec4 albedo;
			glm::vec4 specular;
			float alpha;
			float padding[3];
		};
		static_assert(sizeof(ModelUniform) == 112, "ModelUniform must match std140 layout");

		struct PointLightData {
			glm::vec4 position;
			glm::vec4 color;
		};
		static_assert(sizeof(PointLightData) == 32, "PointLightData must match std430 layout");

		struct SpotLightData {
			glm::vec4 position;
			glm::vec4 color;
			glm::vec4 direction;
			glm::vec4 angles;
		};
		static_assert(sizeof(SpotLightData) == 64, "SpotLightData must match std430 layout");

		struct PointLight {
			glm::vec3 position;
			glm::vec3 color;
			float intensity;
		};

		struct SpotLight {
			glm::vec3 position;
			glm::vec3 direction;
			glm::vec3 color;
			float intensity;
			float inner_angle;
			float outer_angle;
		};

		struct Buffer {
			VkBuffer handle = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;
		};

		struct Object {
			Buffer uniform_buffer;
			VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
		};

		struct LightBuffer {
			Buffer buffer;
			size_t capacity = 0;
		};

		VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
		VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		Buffer vertex_buffer;
		Buffer index_buffer;
		uint32_t index_count = 0;

		Buffer scene_buffer;
		VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

		Object figure;

		LightBuffer point_light_buffer;
		LightBuffer spot_light_buffer;

		std::vector<PointLight> point_lights = {
			{ glm::vec3(2.0f, 1.5f, 2.0f), glm::vec3(1.0f, 0.8f, 0.5f), 10.0f },
		};

		std::vector<SpotLight> spot_lights = {
			{ glm::vec3(-2.0f, 3.0f, 2.0f), glm::vec3(2.0f, -3.0f, -2.0f), glm::vec3(0.5f, 0.7f, 1.0f),
			  30.0f, 15.0f, 25.0f },
		};

		glm::vec3 camera_position{ 0.0f, 0.0f, 5.0f };
		float camera_yaw = 0.0f;
		float camera_pitch = 0.0f;
		float fov_degrees = 60.0f;
		float near_plane = 0.1f;
		float far_plane = 100.0f;

		glm::vec3 directional_direction{ -0.5f, -1.0f, -0.3f };
		glm::vec3 directional_color{ 1.0f, 1.0f, 1.0f };
		float directional_intensity = 1.0f;

		glm::vec3 ambient_color{ 1.0f, 1.0f, 1.0f };
		float ambient_intensity = 0.1f;

		glm::vec3 position{ 0.0f, 0.0f, 0.0f };
		glm::vec3 rotation_degrees{ 20.0f, 30.0f, 0.0f };
		glm::vec3 scale{ 1.0f, 1.0f, 1.0f };

		glm::vec3 albedo{ 0.8f, 0.8f, 0.8f };
		glm::vec3 specular{ 1.0f, 1.0f, 1.0f };
		float alpha = 32.0f;

		bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
			auto& context = graphics::internal::context;

			const VkBufferCreateInfo buffer_info = {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			const VmaAllocationCreateInfo allocation_info = {
				.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO,
			};

			if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
				&buffer.handle, &buffer.allocation, nullptr) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan buffer\n";
				return false;
			}

			if (data != nullptr) {
				if (vmaCopyMemoryToAllocation(context.allocator, data, buffer.allocation, 0, size) != VK_SUCCESS) {
					std::cerr << "Failed to upload data to Vulkan buffer\n";
					return false;
				}
			}

			return true;
		}

		void uploadBuffer(const Buffer& buffer, const void* data, VkDeviceSize size) {
			auto& context = graphics::internal::context;
			vmaCopyMemoryToAllocation(context.allocator, data, buffer.allocation, 0, size);
		}

		void destroyBuffer(Buffer& buffer) {
			auto& context = graphics::internal::context;
			if (buffer.handle != VK_NULL_HANDLE) {
				vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
			}
			buffer = {};
		}

		VkShaderModule loadShader(const char* path) {
			auto& context = graphics::internal::context;

			std::ifstream file(path, std::ios::binary | std::ios::ate);
			if (!file) {
				std::cerr << "Failed to open shader file " << path
					<< " (run the app from the project root and build the shaders)\n";
				return VK_NULL_HANDLE;
			}

			const size_t size = size_t(file.tellg());
			std::vector<uint32_t> code((size + 3) / 4);
			file.seekg(0);
			file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

			const VkShaderModuleCreateInfo module_info = {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = size,
				.pCode = code.data(),
			};

			VkShaderModule module = VK_NULL_HANDLE;
			if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
				std::cerr << "Failed to create shader module from " << path << '\n';
				return VK_NULL_HANDLE;
			}

			return module;
		}

		void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
			const Buffer& buffer, VkDeviceSize range) {
			auto& context = graphics::internal::context;

			const VkDescriptorBufferInfo buffer_info = {
				.buffer = buffer.handle,
				.offset = 0,
				.range = range,
			};

			const VkWriteDescriptorSet write = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = set,
				.dstBinding = binding,
				.descriptorCount = 1,
				.descriptorType = type,
				.pBufferInfo = &buffer_info,
			};

			vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
		}

		bool createDescriptorSetLayout(const VkDescriptorSetLayoutBinding* bindings, uint32_t binding_count,
			VkDescriptorSetLayout& layout) {
			auto& context = graphics::internal::context;

			const VkDescriptorSetLayoutCreateInfo layout_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = binding_count,
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				return false;
			}

			return true;
		}

		bool ensureLightBuffer(LightBuffer& light_buffer, uint32_t binding, size_t element_size, size_t required) {
			if (light_buffer.buffer.handle != VK_NULL_HANDLE && required <= light_buffer.capacity) {
				return true;
			}

			const size_t new_capacity = std::max<size_t>({ required, light_buffer.capacity * 2, 4 });
			const VkDeviceSize new_size = VkDeviceSize(new_capacity * element_size);

			Buffer new_buffer;
			if (!createBuffer(nullptr, new_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, new_buffer)) {
				return false;
			}

			destroyBuffer(light_buffer.buffer);
			light_buffer.buffer = new_buffer;
			light_buffer.capacity = new_capacity;

			writeBufferDescriptor(scene_descriptor_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				light_buffer.buffer, new_size);

			return true;
		}

		bool createDescriptors() {
			auto& context = graphics::internal::context;

			const VkDescriptorSetLayoutBinding scene_bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			const VkDescriptorSetLayoutBinding model_bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			if (!createDescriptorSetLayout(scene_bindings, sizeof(scene_bindings) / sizeof(scene_bindings[0]),
				scene_set_layout)) {
				return false;
			}

			if (!createDescriptorSetLayout(model_bindings, sizeof(model_bindings) / sizeof(model_bindings[0]),
				model_set_layout)) {
				return false;
			}

			const VkDescriptorPoolSize pool_sizes[] = {
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 8 },
				{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 },
			};

			const VkDescriptorPoolCreateInfo pool_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 8,
				.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
				.pPoolSizes = pool_sizes,
			};

			if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				return false;
			}

			if (!createBuffer(nullptr, sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer)) {
				return false;
			}

			if (!createBuffer(nullptr, sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				figure.uniform_buffer)) {
				return false;
			}

			const VkDescriptorSetAllocateInfo scene_set_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &scene_set_layout,
			};

			if (vkAllocateDescriptorSets(context.device, &scene_set_info, &scene_descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to allocate Vulkan descriptor set for the scene\n";
				return false;
			}

			const VkDescriptorSetAllocateInfo model_set_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &model_set_layout,
			};

			if (vkAllocateDescriptorSets(context.device, &model_set_info, &figure.descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to allocate Vulkan descriptor set for the model\n";
				return false;
			}

			writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				scene_buffer, sizeof(SceneUniforms));
			writeBufferDescriptor(figure.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				figure.uniform_buffer, sizeof(ModelUniform));

			if (!ensureLightBuffer(point_light_buffer, 1, sizeof(PointLightData), 1)) return false;
			if (!ensureLightBuffer(spot_light_buffer, 2, sizeof(SpotLightData), 1)) return false;

			return true;
		}

		bool createPipeline() {
			auto& context = graphics::internal::context;

			VkShaderModule vert = loadShader("shaders/scene.vert.spv");
			VkShaderModule frag = loadShader("shaders/scene.frag.spv");
			if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
				if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vert, nullptr);
				if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, frag, nullptr);
				return false;
			}

			const VkPipelineShaderStageCreateInfo stages[] = {
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = vert,
					.pName = "main",
				},
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
					.module = frag,
					.pName = "main",
				},
			};

			const VkVertexInputBindingDescription binding = {
				.binding = 0,
				.stride = sizeof(geometry::Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};

			const VkVertexInputAttributeDescription attributes[] = {
				{
					.location = 0,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = uint32_t(offsetof(geometry::Vertex, position)),
				},
				{
					.location = 1,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = uint32_t(offsetof(geometry::Vertex, normal)),
				},
			};

			const VkPipelineVertexInputStateCreateInfo vertex_input = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &binding,
				.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
				.pVertexAttributeDescriptions = attributes,
			};

			const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};

			const VkPipelineViewportStateCreateInfo viewport = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.scissorCount = 1,
			};

			const VkPipelineRasterizationStateCreateInfo rasterization = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_NONE,
				.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
				.lineWidth = 1.0f,
			};

			const VkPipelineMultisampleStateCreateInfo multisample = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			};

			const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = VK_TRUE,
				.depthWriteEnable = VK_TRUE,
				.depthCompareOp = VK_COMPARE_OP_LESS,
			};

			const VkPipelineColorBlendAttachmentState blend_attachment = {
				.blendEnable = VK_FALSE,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
								  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
			};

			const VkPipelineColorBlendStateCreateInfo color_blend = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 1,
				.pAttachments = &blend_attachment,
			};

			const VkDynamicState dynamic_states[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR,
			};

			const VkPipelineDynamicStateCreateInfo dynamic = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
				.pDynamicStates = dynamic_states,
			};

			const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };

			const VkPipelineLayoutCreateInfo layout_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
				.pSetLayouts = set_layouts,
			};

			bool ok = true;

			if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan pipeline layout\n";
				ok = false;
			}

			if (ok) {
				const VkGraphicsPipelineCreateInfo pipeline_info = {
					.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
					.stageCount = sizeof(stages) / sizeof(stages[0]),
					.pStages = stages,
					.pVertexInputState = &vertex_input,
					.pInputAssemblyState = &input_assembly,
					.pViewportState = &viewport,
					.pRasterizationState = &rasterization,
					.pMultisampleState = &multisample,
					.pDepthStencilState = &depth_stencil,
					.pColorBlendState = &color_blend,
					.pDynamicState = &dynamic,
					.layout = pipeline_layout,
					.renderPass = context.render_pass,
					.subpass = 0,
				};

				if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
					nullptr, &pipeline) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan graphics pipeline\n";
					ok = false;
				}
			}

			vkDestroyShaderModule(context.device, vert, nullptr);
			vkDestroyShaderModule(context.device, frag, nullptr);

			return ok;
		}

		glm::mat4 buildViewProjection(VkExtent2D extent) {
			const float aspect = float(extent.width) / float(extent.height);

			glm::mat4 view(1.0f);
			view = glm::rotate(view, glm::radians(-camera_pitch), glm::vec3(1.0f, 0.0f, 0.0f));
			view = glm::rotate(view, glm::radians(-camera_yaw), glm::vec3(0.0f, 1.0f, 0.0f));
			view = glm::translate(view, -camera_position);

			glm::mat4 projection = glm::perspective(glm::radians(fov_degrees), aspect, near_plane, far_plane);
			projection[1][1] *= -1.0f;

			return projection * view;
		}

		glm::mat4 buildModelMatrix() {
			glm::mat4 model(1.0f);
			model = glm::translate(model, position);
			model = glm::rotate(model, glm::radians(rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
			model = glm::rotate(model, glm::radians(rotation_degrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
			model = glm::rotate(model, glm::radians(rotation_degrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
			model = glm::scale(model, scale);
			return model;
		}

	} // namespace

	bool initialize() {
		const geometry::Mesh mesh = geometry::make_truncated_tetrahedron();
		index_count = uint32_t(mesh.indices.size());

		if (!createBuffer(mesh.vertices.data(),
			mesh.vertices.size() * sizeof(geometry::Vertex),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) {
			return false;
		}

		if (!createBuffer(mesh.indices.data(),
			mesh.indices.size() * sizeof(uint16_t),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
			return false;
		}

		if (!createDescriptors()) {
			return false;
		}

		return createPipeline();
	}

	void shutdown() {
		auto& context = graphics::internal::context;
		vkQueueWaitIdle(context.graphics_queue);

		vkDestroyPipeline(context.device, pipeline, nullptr);
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
		vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

		destroyBuffer(spot_light_buffer.buffer);
		destroyBuffer(point_light_buffer.buffer);
		destroyBuffer(figure.uniform_buffer);
		destroyBuffer(scene_buffer);
		destroyBuffer(index_buffer);
		destroyBuffer(vertex_buffer);
	}

	void update([[maybe_unused]] double time) {
		ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
		ImGui::Begin("Lab 2: camera and lighting");

		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Camera position", &camera_position.x, 0.05f, -50.0f, 50.0f);
			ImGui::SliderFloat("Yaw (deg)", &camera_yaw, -180.0f, 180.0f);
			ImGui::SliderFloat("Pitch (deg)", &camera_pitch, -89.0f, 89.0f);
			ImGui::SliderFloat("FOV (deg)", &fov_degrees, 10.0f, 120.0f);
			ImGui::SliderFloat("Near", &near_plane, 0.01f, 5.0f);
			ImGui::SliderFloat("Far", &far_plane, 6.0f, 200.0f);
		}

		if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Light direction", &directional_direction.x, 0.01f, -1.0f, 1.0f);
			ImGui::ColorEdit3("Light color", &directional_color.x);
			ImGui::SliderFloat("Light intensity", &directional_intensity, 0.0f, 5.0f);
		}

		if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit3("Ambient color", &ambient_color.x);
			ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);
		}

		if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::Button("Add point light")) {
				point_lights.push_back({ glm::vec3(0.0f, 2.0f, 2.0f), glm::vec3(1.0f, 1.0f, 1.0f), 10.0f });
			}

			int remove_index = -1;
			for (size_t i = 0; i < point_lights.size(); ++i) {
				PointLight& light = point_lights[i];
				ImGui::PushID(int(i));
				const std::string label = "Point light " + std::to_string(i + 1);
				if (ImGui::TreeNode(label.c_str())) {
					ImGui::DragFloat3("Position", &light.position.x, 0.05f, -20.0f, 20.0f);
					ImGui::ColorEdit3("Color", &light.color.x);
					ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
					if (ImGui::Button("Remove")) {
						remove_index = int(i);
					}
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			if (remove_index >= 0) {
				point_lights.erase(point_lights.begin() + remove_index);
			}
		}

		if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::Button("Add spot light")) {
				spot_lights.push_back({ glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f),
										glm::vec3(1.0f, 1.0f, 1.0f), 30.0f, 15.0f, 25.0f });
			}

			int remove_index = -1;
			for (size_t i = 0; i < spot_lights.size(); ++i) {
				SpotLight& light = spot_lights[i];
				ImGui::PushID(int(i));
				const std::string label = "Spot light " + std::to_string(i + 1);
				if (ImGui::TreeNode(label.c_str())) {
					ImGui::DragFloat3("Position", &light.position.x, 0.05f, -20.0f, 20.0f);
					ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f);
					ImGui::ColorEdit3("Color", &light.color.x);
					ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 200.0f);
					ImGui::SliderFloat("Outer angle (deg)", &light.outer_angle, 1.0f, 89.0f);
					light.inner_angle = std::min(light.inner_angle, light.outer_angle);
					ImGui::SliderFloat("Inner angle (deg)", &light.inner_angle, 0.0f, light.outer_angle);
					if (ImGui::Button("Remove")) {
						remove_index = int(i);
					}
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			if (remove_index >= 0) {
				spot_lights.erase(spot_lights.begin() + remove_index);
			}
		}

		if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit3("Albedo", &albedo.x);
			ImGui::ColorEdit3("Specular color", &specular.x);
			ImGui::SliderFloat("Alpha (shininess)", &alpha, 1.0f, 256.0f);
		}

		if (ImGui::CollapsingHeader("Object transform", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", &position.x, 0.02f, -5.0f, 5.0f);
			ImGui::SliderFloat3("Rotation (deg)", &rotation_degrees.x, -180.0f, 180.0f);
			ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.1f, 5.0f);
		}

		ImGui::End();
	}

	void render(const graphics::internal::FrameData& fd) {
		auto& context = graphics::internal::context;

		glm::vec3 light_direction = directional_direction;
		if (glm::length(light_direction) < 1e-4f) {
			light_direction = glm::vec3(0.0f, -1.0f, 0.0f);
		}
		light_direction = glm::normalize(light_direction);

		std::vector<PointLightData> point_data;
		for (const PointLight& light : point_lights) {
			point_data.push_back({ glm::vec4(light.position, 1.0f),
								   glm::vec4(light.color * light.intensity, 1.0f) });
		}

		std::vector<SpotLightData> spot_data;
		for (const SpotLight& light : spot_lights) {
			glm::vec3 direction = light.direction;
			if (glm::length(direction) < 1e-4f) {
				direction = glm::vec3(0.0f, -1.0f, 0.0f);
			}
			direction = glm::normalize(direction);

			const float inner = glm::radians(std::min(light.inner_angle, light.outer_angle));
			const float outer = glm::radians(light.outer_angle);

			spot_data.push_back({ glm::vec4(light.position, 1.0f),
								  glm::vec4(light.color * light.intensity, 1.0f),
								  glm::vec4(direction, 0.0f),
								  glm::vec4(std::cos(inner), std::cos(outer), 0.0f, 0.0f) });
		}

		ensureLightBuffer(point_light_buffer, 1, sizeof(PointLightData), point_data.size());
		ensureLightBuffer(spot_light_buffer, 2, sizeof(SpotLightData), spot_data.size());

		if (!point_data.empty()) {
			uploadBuffer(point_light_buffer.buffer, point_data.data(), point_data.size() * sizeof(PointLightData));
		}
		if (!spot_data.empty()) {
			uploadBuffer(spot_light_buffer.buffer, spot_data.data(), spot_data.size() * sizeof(SpotLightData));
		}

		const SceneUniforms scene = {
			.view_projection = buildViewProjection(context.swapchain_extent),
			.camera_position = glm::vec4(camera_position, 1.0f),
			.directional_light_direction = glm::vec4(light_direction, 0.0f),
			.directional_light_color = glm::vec4(directional_color * directional_intensity, 1.0f),
			.ambient_light = glm::vec4(ambient_color * ambient_intensity, 1.0f),
			.point_light_count = uint32_t(point_data.size()),
			.spot_light_count = uint32_t(spot_data.size()),
			.padding = {0, 0},
		};
		uploadBuffer(scene_buffer, &scene, sizeof(scene));

		const ModelUniform model = {
			.model = buildModelMatrix(),
			.albedo = glm::vec4(albedo, 1.0f),
			.specular = glm::vec4(specular, 1.0f),
			.alpha = alpha,
			.padding = {0.0f, 0.0f, 0.0f},
		};
		uploadBuffer(figure.uniform_buffer, &model, sizeof(model));

		const VkCommandBufferBeginInfo begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(fd.command_buffer, &begin_info);

		VkClearValue clear_values[2] = {};
		clear_values[0].color = { {0.1f, 0.1f, 0.1f, 1.0f} };
		clear_values[1].depthStencil = { 1.0f, 0 };

		const VkRenderPassBeginInfo render_pass_begin = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = context.render_pass,
			.framebuffer = fd.framebuffer,
			.renderArea = {.offset = {0, 0}, .extent = context.swapchain_extent },
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		const VkViewport viewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width = float(context.swapchain_extent.width),
			.height = float(context.swapchain_extent.height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

		const VkRect2D scissor = { .offset = {0, 0}, .extent = context.swapchain_extent };
		vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
			0, 1, &scene_descriptor_set, 0, nullptr);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
			1, 1, &figure.descriptor_set, 0, nullptr);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application