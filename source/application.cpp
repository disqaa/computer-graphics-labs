#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
			glm::vec4 tint;
			float alpha;
			float padding[3];
		};
		static_assert(sizeof(ModelUniform) == 128, "ModelUniform must match std140 layout");

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

		struct GpuMesh {
			Buffer vertex_buffer;
			Buffer index_buffer;
			uint32_t index_count = 0;
		};

		struct SceneObject {
			glm::vec3 position{ 0.0f, 0.0f, 0.0f };
			glm::vec3 rotation_degrees{ 20.0f, 30.0f, 0.0f };
			glm::vec3 scale{ 1.0f, 1.0f, 1.0f };
			glm::vec3 albedo{ 1.0f, 1.0f, 1.0f };
			glm::vec3 specular{ 1.0f, 1.0f, 1.0f };
			glm::vec3 tint{ 1.0f, 1.0f, 1.0f };
			float alpha = 32.0f;
			bool animated = false;
		};

		struct StagingContext {
			VkCommandPool command_pool = VK_NULL_HANDLE;
			VkCommandBuffer command_buffer = VK_NULL_HANDLE;
			Buffer buffer;
			uint8_t* mapped = nullptr;
			VkDeviceSize offset = 0;
			VkDeviceSize capacity = 0;
		};

		constexpr uint32_t max_objects = 64;
		constexpr VkDeviceSize staging_buffer_size = 64ull * 1024ull * 1024ull;

		struct LightBuffer {
			Buffer buffer;
			size_t capacity = 0;
		};

		VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
		VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
		VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		GpuMesh figure_mesh;

		Buffer scene_buffer;
		VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

		Buffer model_buffer;
		VkDeviceSize model_uniform_stride = 0;
		VkDescriptorSet model_descriptor_set = VK_NULL_HANDLE;

		StagingContext staging;

		std::vector<SceneObject> objects;
		int selected_object = 0;

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
		int projection_type = 0;
		float ortho_size = 2.0f;

		glm::vec3 directional_direction{ -0.5f, -1.0f, -0.3f };
		glm::vec3 directional_color{ 1.0f, 1.0f, 1.0f };
		float directional_intensity = 1.0f;

		glm::vec3 ambient_color{ 1.0f, 1.0f, 1.0f };
		float ambient_intensity = 0.1f;

		bool animation_playing = true;
		float animation_speed = 1.0f;
		float animation_time = 0.0f;
		float trajectory_radius = 1.2f;
		float trajectory_height = 0.6f;
		float trajectory_depth = 1.0f;
		float spin_speed = 60.0f;


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

		void uploadBuffer(const Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset = 0) {
			auto& context = graphics::internal::context;
			vmaCopyMemoryToAllocation(context.allocator, data, buffer.allocation, offset, size);
		}

		void destroyBuffer(Buffer& buffer) {
			auto& context = graphics::internal::context;
			if (buffer.handle != VK_NULL_HANDLE) {
				vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
			}
			buffer = {};
		}

		bool beginStaging() {
			auto& context = graphics::internal::context;

			const VkCommandPoolCreateInfo pool_info = {
				.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				.queueFamilyIndex = context.graphics_queue_index,
			};

			if (vkCreateCommandPool(context.device, &pool_info, nullptr, &staging.command_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan command pool for uploads\n";
				return false;
			}

			const VkCommandBufferAllocateInfo allocate_info = {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = staging.command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};

			if (vkAllocateCommandBuffers(context.device, &allocate_info, &staging.command_buffer) != VK_SUCCESS) {
				std::cerr << "Failed to allocate Vulkan command buffer for uploads\n";
				return false;
			}

			const VkBufferCreateInfo buffer_info = {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = staging_buffer_size,
				.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			const VmaAllocationCreateInfo allocation_info = {
				.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO,
			};

			VmaAllocationInfo allocation_result = {};
			if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info, &staging.buffer.handle,
				&staging.buffer.allocation, &allocation_result) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan staging buffer\n";
				return false;
			}

			staging.mapped = static_cast<uint8_t*>(allocation_result.pMappedData);
			staging.offset = 0;
			staging.capacity = staging_buffer_size;

			const VkCommandBufferBeginInfo begin_info = {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			};

			if (vkBeginCommandBuffer(staging.command_buffer, &begin_info) != VK_SUCCESS) {
				std::cerr << "Failed to begin recording of the upload command buffer\n";
				return false;
			}

			return true;
		}

		bool stageData(const void* data, VkDeviceSize size, VkDeviceSize& offset) {
			auto& context = graphics::internal::context;

			const VkDeviceSize aligned_offset = (staging.offset + 15) & ~VkDeviceSize(15);
			if (aligned_offset + size > staging.capacity) {
				std::cerr << "Staging buffer is too small for the uploaded data\n";
				return false;
			}

			std::memcpy(staging.mapped + aligned_offset, data, size_t(size));
			vmaFlushAllocation(context.allocator, staging.buffer.allocation, aligned_offset, size);

			offset = aligned_offset;
			staging.offset = aligned_offset + size;

			return true;
		}

		bool endStaging() {
			auto& context = graphics::internal::context;

			bool ok = true;

			if (staging.command_buffer != VK_NULL_HANDLE) {
				const VkMemoryBarrier barrier = {
					.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
					.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
									 VK_ACCESS_SHADER_READ_BIT,
				};

				vkCmdPipelineBarrier(staging.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					0, 1, &barrier, 0, nullptr, 0, nullptr);

				if (vkEndCommandBuffer(staging.command_buffer) != VK_SUCCESS) {
					std::cerr << "Failed to end recording of the upload command buffer\n";
					ok = false;
				}
			}

			if (ok) {
				const VkFenceCreateInfo fence_info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
				VkFence fence = VK_NULL_HANDLE;

				if (vkCreateFence(context.device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan fence for uploads\n";
					ok = false;
				}
				else {
					const VkSubmitInfo submit_info = {
						.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
						.commandBufferCount = 1,
						.pCommandBuffers = &staging.command_buffer,
					};

					if (vkQueueSubmit(context.graphics_queue, 1, &submit_info, fence) != VK_SUCCESS) {
						std::cerr << "Failed to submit the upload command buffer\n";
						ok = false;
					}
					else {
						vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX);
					}

					vkDestroyFence(context.device, fence, nullptr);
				}
			}

			if (staging.command_pool != VK_NULL_HANDLE) {
				vkDestroyCommandPool(context.device, staging.command_pool, nullptr);
			}
			destroyBuffer(staging.buffer);
			staging = {};

			return ok;
		}

		bool createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
			auto& context = graphics::internal::context;

			const VkBufferCreateInfo buffer_info = {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			};

			const VmaAllocationCreateInfo allocation_info = {
				.usage = VMA_MEMORY_USAGE_AUTO,
			};

			if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
				&buffer.handle, &buffer.allocation, nullptr) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan buffer in video memory\n";
				return false;
			}

			VkDeviceSize staging_offset = 0;
			if (!stageData(data, size, staging_offset)) {
				return false;
			}

			const VkBufferCopy copy_region = {
				.srcOffset = staging_offset,
				.dstOffset = 0,
				.size = size,
			};
			vkCmdCopyBuffer(staging.command_buffer, staging.buffer.handle, buffer.handle, 1, &copy_region);

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
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
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
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
				{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 32 },
				{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 },
			};

			const VkDescriptorPoolCreateInfo pool_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 32,
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

			writeBufferDescriptor(scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				scene_buffer, sizeof(SceneUniforms));

			VkPhysicalDeviceProperties physical_device_props = {};
			vkGetPhysicalDeviceProperties(context.physical_device, &physical_device_props);

			const VkDeviceSize alignment = physical_device_props.limits.minUniformBufferOffsetAlignment;
			model_uniform_stride = (sizeof(ModelUniform) + alignment - 1) & ~(alignment - 1);

			if (!createBuffer(nullptr, model_uniform_stride * max_objects, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				model_buffer)) {
				return false;
			}

			const VkDescriptorSetAllocateInfo model_set_info = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &model_set_layout,
			};

			if (vkAllocateDescriptorSets(context.device, &model_set_info, &model_descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to allocate Vulkan descriptor set for the models\n";
				return false;
			}

			writeBufferDescriptor(model_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				model_buffer, sizeof(ModelUniform));

			if (!ensureLightBuffer(point_light_buffer, 1, sizeof(PointLightData), 1)) return false;
			if (!ensureLightBuffer(spot_light_buffer, 2, sizeof(SpotLightData), 1)) return false;

			return true;
		}

		bool createMesh(const geometry::Mesh& mesh, GpuMesh& gpu_mesh) {
			gpu_mesh.index_count = uint32_t(mesh.indices.size());

			if (!createDeviceBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(geometry::Vertex),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, gpu_mesh.vertex_buffer)) {
				return false;
			}

			if (!createDeviceBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint16_t),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT, gpu_mesh.index_buffer)) {
				return false;
			}

			return true;
		}

		void destroyMesh(GpuMesh& gpu_mesh) {
			destroyBuffer(gpu_mesh.index_buffer);
			destroyBuffer(gpu_mesh.vertex_buffer);
			gpu_mesh.index_count = 0;
		}

		void uploadModelUniform(uint32_t index, const glm::mat4& model, const SceneObject& object) {
			const ModelUniform uniform = {
				.model = model,
				.albedo = glm::vec4(object.albedo, 1.0f),
				.specular = glm::vec4(object.specular, 1.0f),
				.tint = glm::vec4(object.tint, 1.0f),
				.alpha = object.alpha,
				.padding = {0.0f, 0.0f, 0.0f},
			};
			uploadBuffer(model_buffer, &uniform, sizeof(uniform), VkDeviceSize(index) * model_uniform_stride);
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
				{
					.location = 2,
					.binding = 0,
					.format = VK_FORMAT_R32G32B32_SFLOAT,
					.offset = uint32_t(offsetof(geometry::Vertex, color)),
				},
				{
					.location = 3,
					.binding = 0,
					.format = VK_FORMAT_R32G32_SFLOAT,
					.offset = uint32_t(offsetof(geometry::Vertex, uv)),
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

			return projection * view;
		}

		glm::mat4 buildModelMatrix(const SceneObject& object) {
			glm::vec3 offset(0.0f);
			glm::vec3 angles = object.rotation_degrees;

			if (object.animated) {
				const float t = animation_time;
				offset = glm::vec3(trajectory_radius * std::sin(t),
					trajectory_height * std::sin(2.0f * t),
					trajectory_depth * std::sin(3.0f * t));
				angles += glm::vec3(0.5f * spin_speed * t, spin_speed * t, 0.0f);
			}

			glm::mat4 model(1.0f);
			model = glm::translate(model, object.position + offset);
			model = glm::rotate(model, glm::radians(angles.z), glm::vec3(0.0f, 0.0f, 1.0f));
			model = glm::rotate(model, glm::radians(angles.y), glm::vec3(0.0f, 1.0f, 0.0f));
			model = glm::rotate(model, glm::radians(angles.x), glm::vec3(1.0f, 0.0f, 0.0f));
			model = glm::scale(model, object.scale);
			return model;
		}

	} // namespace

	bool initialize() {
		if (!createDescriptors()) {
			return false;
		}

		if (!beginStaging()) {
			return false;
		}

		if (!createMesh(geometry::make_truncated_tetrahedron(), figure_mesh)) {
			endStaging();
			return false;
		}

		if (!endStaging()) {
			return false;
		}

		SceneObject main_figure;
		main_figure.animated = true;

		SceneObject left_figure;
		left_figure.position = glm::vec3(-3.0f, 0.0f, 0.0f);
		left_figure.rotation_degrees = glm::vec3(-20.0f, 60.0f, 10.0f);

		SceneObject right_figure;
		right_figure.position = glm::vec3(3.0f, 0.0f, 0.0f);
		right_figure.rotation_degrees = glm::vec3(30.0f, -40.0f, 0.0f);
		right_figure.scale = glm::vec3(0.8f, 0.8f, 0.8f);

		objects = { main_figure, left_figure, right_figure };

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
		destroyBuffer(model_buffer);
		destroyBuffer(scene_buffer);
		destroyMesh(figure_mesh);
	}

	void update([[maybe_unused]] double time) {
		if (animation_playing) {
			animation_time += ImGui::GetIO().DeltaTime * animation_speed;
		}

		ImGui::SetNextWindowSize(ImVec2(400.0f, 0.0f), ImGuiCond_FirstUseEver);
		ImGui::Begin("Lab 3: textures, camera and lighting");

		if (ImGui::CollapsingHeader("Camera and projection", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Camera position", &camera_position.x, 0.05f, -50.0f, 50.0f);
			ImGui::SliderFloat("Yaw (deg)", &camera_yaw, -180.0f, 180.0f);
			ImGui::SliderFloat("Pitch (deg)", &camera_pitch, -89.0f, 89.0f);

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

		if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Objects in scene: %d", int(objects.size()));

			if (ImGui::Button("Add object") && objects.size() < max_objects) {
				SceneObject copy = objects[size_t(selected_object)];
				copy.position += glm::vec3(0.5f, 0.0f, 0.0f);
				copy.animated = false;
				objects.push_back(copy);
				selected_object = int(objects.size()) - 1;
			}
			ImGui::SameLine();
			if (ImGui::Button("Remove selected") && objects.size() > 1) {
				objects.erase(objects.begin() + selected_object);
			}

			selected_object = std::clamp(selected_object, 0, int(objects.size()) - 1);
			ImGui::SliderInt("Selected object", &selected_object, 0, int(objects.size()) - 1);
		}

		SceneObject& object = objects[size_t(selected_object)];

		if (ImGui::CollapsingHeader("Material (selected object)", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit3("Albedo", &object.albedo.x);
			ImGui::ColorEdit3("Specular color", &object.specular.x);
			ImGui::SliderFloat("Alpha (shininess)", &object.alpha, 1.0f, 256.0f);
			ImGui::ColorEdit3("Tint (multiplies vertex colors)", &object.tint.x);
		}

		if (ImGui::CollapsingHeader("Object transform (selected object)", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", &object.position.x, 0.02f, -10.0f, 10.0f);
			ImGui::SliderFloat3("Rotation (deg)", &object.rotation_degrees.x, -180.0f, 180.0f);
			ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.1f, 5.0f);
			ImGui::Checkbox("Move along trajectory (animation)", &object.animated);
		}

		if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
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
		}

		if (ImGui::Button("Reset projection, animation and selected object")) {
			projection_type = 0;
			fov_degrees = 60.0f;
			ortho_size = 2.0f;
			near_plane = 0.1f;
			far_plane = 100.0f;
			object.position = { 0.0f, 0.0f, 0.0f };
			object.rotation_degrees = { 20.0f, 30.0f, 0.0f };
			object.scale = { 1.0f, 1.0f, 1.0f };
			object.tint = { 1.0f, 1.0f, 1.0f };
			animation_playing = true;
			animation_speed = 1.0f;
			animation_time = 0.0f;
			trajectory_radius = 1.2f;
			trajectory_height = 0.6f;
			trajectory_depth = 1.0f;
			spin_speed = 60.0f;
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

		for (size_t i = 0; i < objects.size(); ++i) {
			uploadModelUniform(uint32_t(i), buildModelMatrix(objects[i]), objects[i]);
		}

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

		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &figure_mesh.vertex_buffer.handle, &vertex_offset);
		vkCmdBindIndexBuffer(fd.command_buffer, figure_mesh.index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);

		for (size_t i = 0; i < objects.size(); ++i) {
			const uint32_t dynamic_offset = uint32_t(VkDeviceSize(i) * model_uniform_stride);
			vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
				1, 1, &model_descriptor_set, 1, &dynamic_offset);
			vkCmdDrawIndexed(fd.command_buffer, figure_mesh.index_count, 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(fd.command_buffer);
		vkEndCommandBuffer(fd.command_buffer);
	}

} // namespace application