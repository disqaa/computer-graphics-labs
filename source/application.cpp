#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

#include "geometry.hpp"

namespace application {

	namespace {

		struct PushConstants {
			glm::mat4 mvp;
			glm::vec4 tint;
		};
		static_assert(sizeof(PushConstants) <= 128, "Push constants are too large");

		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		VkBuffer vertex_buffer = VK_NULL_HANDLE;
		VmaAllocation vertex_allocation = VK_NULL_HANDLE;
		VkBuffer index_buffer = VK_NULL_HANDLE;
		VmaAllocation index_allocation = VK_NULL_HANDLE;
		uint32_t index_count = 0;

		int projection_type = 0;
		float fov_degrees = 60.0f;
		float ortho_size = 2.0f;
		float near_plane = 0.1f;
		float far_plane = 100.0f;

		glm::vec3 position{ 0.0f, 0.0f, 0.0f };
		glm::vec3 rotation_degrees{ 20.0f, 30.0f, 0.0f };
		glm::vec3 scale{ 1.0f, 1.0f, 1.0f };
		glm::vec3 tint{ 1.0f, 1.0f, 1.0f };

		bool animation_playing = true;
		float animation_speed = 1.0f;
		float animation_time = 0.0f;
		float trajectory_radius = 2.0f;
		float trajectory_height = 0.8f;
		float trajectory_depth = 1.0f;
		float spin_speed = 60.0f;

		bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
			VkBuffer& buffer, VmaAllocation& allocation) {
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
				&buffer, &allocation, nullptr) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan buffer\n";
				return false;
			}

			if (vmaCopyMemoryToAllocation(context.allocator, data, allocation, 0, size) != VK_SUCCESS) {
				std::cerr << "Failed to upload data to Vulkan buffer\n";
				return false;
			}

			return true;
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

		bool createPipeline() {
			auto& context = graphics::internal::context;

			VkShaderModule vert = loadShader("shaders/triangle.vert.spv");
			VkShaderModule frag = loadShader("shaders/triangle.frag.spv");
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
					.offset = uint32_t(offsetof(geometry::Vertex, color)),
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

			const VkPushConstantRange push_range = {
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
				.offset = 0,
				.size = sizeof(PushConstants),
			};

			const VkPipelineLayoutCreateInfo layout_info = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_range,
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

		glm::mat4 buildMVP(VkExtent2D extent) {
			const float aspect = float(extent.width) / float(extent.height);

			const float t = animation_time;
			const glm::vec3 offset(trajectory_radius * std::sin(t),
				trajectory_height * std::sin(2.0f * t),
				trajectory_depth * std::sin(3.0f * t));
			const glm::vec3 angles = rotation_degrees +
				glm::vec3(0.5f * spin_speed * t, spin_speed * t, 0.0f);

			glm::mat4 model(1.0f);
			model = glm::translate(model, position + offset);
			model = glm::rotate(model, glm::radians(angles.z), glm::vec3(0.0f, 0.0f, 1.0f));
			model = glm::rotate(model, glm::radians(angles.y), glm::vec3(0.0f, 1.0f, 0.0f));
			model = glm::rotate(model, glm::radians(angles.x), glm::vec3(1.0f, 0.0f, 0.0f));
			model = glm::scale(model, scale);

			const glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f),
				glm::vec3(0.0f, 0.0f, 0.0f),
				glm::vec3(0.0f, 1.0f, 0.0f));

			glm::mat4 projection;
			if (projection_type == 0) {
				projection = glm::perspective(glm::radians(fov_degrees), aspect, near_plane, far_plane);
			}
			else {
				const float half_height = ortho_size;
				const float half_width = ortho_size * aspect;
				projection = glm::ortho(-half_width, half_width, -half_height, half_height,
					near_plane, far_plane);
			}

			projection[1][1] *= -1.0f;

			return projection * view * model;
		}

	}

	bool initialize() {
		const geometry::Mesh mesh = geometry::make_truncated_tetrahedron();
		index_count = uint32_t(mesh.indices.size());

		if (!createBuffer(mesh.vertices.data(),
			mesh.vertices.size() * sizeof(geometry::Vertex),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer, vertex_allocation)) {
			return false;
		}

		if (!createBuffer(mesh.indices.data(),
			mesh.indices.size() * sizeof(uint16_t),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer, index_allocation)) {
			return false;
		}

		return createPipeline();
	}

	void shutdown() {
		auto& context = graphics::internal::context;
		vkQueueWaitIdle(context.graphics_queue);

		vkDestroyPipeline(context.device, pipeline, nullptr);
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

		vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
	}

	void update([[maybe_unused]] double time) {
		if (animation_playing) {
			animation_time += ImGui::GetIO().DeltaTime * animation_speed;
		}

		ImGui::Begin("Truncated tetrahedron");

		ImGui::SeparatorText("Projection");
		ImGui::RadioButton("Perspective", &projection_type, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &projection_type, 1);

		if (projection_type == 0) {
			ImGui::SliderFloat("FOV (deg)", &fov_degrees, 10.0f, 120.0f);
		}
		else {
			ImGui::SliderFloat("Ortho size", &ortho_size, 0.5f, 10.0f);
		}
		ImGui::SliderFloat("Near", &near_plane, 0.01f, 5.0f);
		ImGui::SliderFloat("Far", &far_plane, 6.0f, 200.0f);

		ImGui::SeparatorText("Transform");
		ImGui::DragFloat3("Position", &position.x, 0.02f, -5.0f, 5.0f);
		ImGui::SliderFloat3("Rotation (deg)", &rotation_degrees.x, -180.0f, 180.0f);
		ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.1f, 5.0f);

		ImGui::SeparatorText("Color");
		ImGui::ColorEdit3("Tint", &tint.x);

		ImGui::SeparatorText("Animation");
		if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
			animation_playing = !animation_playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			animation_time = 0.0f;
		}
		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius (X)", &trajectory_radius, 0.0f, 4.0f);
		ImGui::SliderFloat("Height (Y)", &trajectory_height, 0.0f, 3.0f);
		ImGui::SliderFloat("Depth (Z)", &trajectory_depth, 0.0f, 3.0f);
		ImGui::SliderFloat("Spin (deg/s)", &spin_speed, 0.0f, 360.0f);

		if (ImGui::Button("Reset")) {
			projection_type = 0;
			fov_degrees = 60.0f;
			ortho_size = 2.0f;
			near_plane = 0.1f;
			far_plane = 100.0f;
			position = { 0.0f, 0.0f, 0.0f };
			rotation_degrees = { 20.0f, 30.0f, 0.0f };
			scale = { 1.0f, 1.0f, 1.0f };
			tint = { 1.0f, 1.0f, 1.0f };
			animation_playing = true;
			animation_speed = 1.0f;
			animation_time = 0.0f;
			trajectory_radius = 2.0f;
			trajectory_height = 0.8f;
			trajectory_depth = 1.0f;
			spin_speed = 60.0f;
		}

		ImGui::End();
	}

	void render(const graphics::internal::FrameData& fd) {
		auto& context = graphics::internal::context;

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

		const PushConstants push = {
			.mvp = buildMVP(context.swapchain_extent),
			.tint = glm::vec4(tint, 1.0f),
		};
		vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
			0, sizeof(push), &push);

		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &offset);
		vkCmdBindIndexBuffer(fd.command_buffer, index_buffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

}