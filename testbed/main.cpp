#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <algorithm>
#include <corecrt_math_defines.h>
#include <cmath>
#include <algorithm>
#include <veekay/veekay.hpp>
#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

size_t aligned_sizeof;
float sens = 0.1f;
float ambient_light = 0.025f;
bool is_daytime = false;
bool toggle_camera = false;
bool animate = true;

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 16;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;

	veekay::vec3 view_position; float _pad0;

	veekay::vec3 ambient_light_intensity; float _pad1;

	veekay::vec3 sun_light_direction; float _pad2;
	veekay::vec3 sun_light_color; float _pad3;

	uint32_t point_lights_count;
	uint32_t spot_lights_count;

	uint32_t _pad4[2];
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	
	veekay::vec3 specular_color; float _pad2;
	uint32_t t_idx;
	uint32_t _pad3;
	float shininess; 
	uint32_t _pad4;
};

typedef struct PointLight {
    veekay::vec3 position; float radius; 
    veekay::vec3 color; float _pad0;
} PointLight;

typedef struct SpotLight {
    veekay::vec3 position; float radius;   
    veekay::vec3 direction; float angle_cos;
    veekay::vec3 color; float _pad0;
} SpotLight;

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	
	veekay::vec3 specular_color;
	uint32_t t_idx;
	float shininess; 
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
	
	// NOTE: Get camera forward direction
	veekay::vec3 forward() const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {-0.3f, -2.0f, -3.6f},
		.rotation = {0.32f, 0.2f, 0.0f}
	};

	std::vector<Model> models;
	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	veekay::graphics::Buffer* point_lights_buffer;
	veekay::graphics::Buffer* spot_lights_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	veekay::graphics::Texture* batman_texture;
	VkSampler texture_sampler;
	VkSampler batman_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
    veekay::mat4 scale_mat = veekay::mat4::scaling(scale);
    veekay::mat4 yaw_rot = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    veekay::mat4 pitch_rot = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    veekay::mat4 roll_rot = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
    veekay::mat4 rotation_mat = yaw_rot * pitch_rot * roll_rot;
    veekay::mat4 translation_mat = veekay::mat4::translation(position);
    return scale_mat * rotation_mat * translation_mat;
}

veekay::mat4 look_at_matrix(const veekay::vec3& eye, const veekay::vec3& target, const veekay::vec3& world_up) {
    veekay::vec3 forward = veekay::vec3::normalized(eye - target);
    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(world_up, forward));
    veekay::vec3 up = veekay::vec3::cross(forward, right);

    veekay::mat4 result;
    result[0][0] = right.x;   result[0][1] = up.x;   result[0][2] = forward.x;   result[0][3] = 0.0f;
    result[1][0] = right.y;   result[1][1] = up.y;   result[1][2] = forward.y;   result[1][3] = 0.0f;
    result[2][0] = right.z;   result[2][1] = up.z;   result[2][2] = forward.z;   result[2][3] = 0.0f;
    result[3][0] = -veekay::vec3::dot(right, eye);
    result[3][1] = -veekay::vec3::dot(up, eye);
    result[3][2] = -veekay::vec3::dot(forward, eye);
    result[3][3] = 1.0f;

    return result;
}

veekay::vec3 Camera::forward() const {
	float yaw_rad = rotation.y;
    float pitch_rad = rotation.x;
    
    float cos_yaw = cosf(yaw_rad);
    float sin_yaw = sinf(yaw_rad);
    float cos_pitch = -cosf(pitch_rad);
    float sin_pitch = -sinf(pitch_rad);
    
    veekay::vec3 forward = {
		sin_yaw * cos_pitch,
        sin_pitch,
        cos_yaw * cos_pitch,
    };
    forward = veekay::vec3::normalized(forward);
	return forward;
}

veekay::mat4 Camera::view() const {
    veekay::vec3 forward = Camera::forward();

    veekay::vec3 target = position + forward;
    veekay::vec3 up = {0.0f, 1.0f, 0.0f};

    return look_at_matrix(position, target, up);
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physical_device, &props);
	uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
	aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 4,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 5,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},

			};


			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	point_lights_buffer = new veekay::graphics::Buffer( 
        max_point_lights * sizeof(PointLight), 
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    spot_lights_buffer = new veekay::graphics::Buffer(  
        max_spot_lights * sizeof(SpotLight), 
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	{
		
		// Считываем данные об изображении из файла
		uint32_t width, height;
		std::vector<uint8_t> pixels;
		lodepng::decode(pixels, width, height, "./assets/lenna.png");

		// Создаем текстуру с данными об изображении
		texture = new veekay::graphics::Texture(
			cmd, width, height,
			VK_FORMAT_R8G8B8A8_UNORM, // 8 бит на каждый канал цвета
			pixels.data());
		
		pixels.clear();
		lodepng::decode(pixels, width, height, "./assets/batman.png");

		// Создаем текстуру с данными об изображении
		batman_texture = new veekay::graphics::Texture(
			cmd, width, height,
			VK_FORMAT_R8G8B8A8_UNORM, // 8 бит на каждый канал цвета
			pixels.data());
		
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR, // Фильтрация если плотность текселей меньше
			.minFilter = VK_FILTER_LINEAR, // Фильтрация если плотность больше
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, // Фильтрация мип-мапов
			// Что делать, если по какой-то из осей вышли за границы текстурных коорд-т
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.anisotropyEnable = true, // Включить анизотропную фильтрацию?
			.maxAnisotropy = 16.0f,   // Кол-во сэмплов анизотропной фильтрации
			.minLod = 0.0f, // Минимальный уровень мипа
		.maxLod = VK_LOD_CLAMP_NONE, // Максимальный уровень мипа (тут бескоченость)
		};

		vkCreateSampler(device, &info, nullptr, &texture_sampler);
		vkCreateSampler(device, &info, nullptr, &batman_sampler);
	}
	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	VkDescriptorImageInfo image_infos[] = {
		{
			.sampler = texture_sampler,         // Какой сэмплер будет использоваться
			.imageView = texture->view, // Какая текстура будет использоваться
	// Формат текстуры будет использован оптимальный для чтения в шейдере
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		},
		{
			.sampler = texture_sampler,         // Какой сэмплер будет использоваться
			.imageView = batman_texture->view, // Какая текстура будет использоваться
	// Формат текстуры будет использован оптимальный для чтения в шейдере
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		},
	};

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
			{
				.buffer = spot_lights_buffer->buffer,
				.offset = 0,
				.range = max_spot_lights * sizeof(SpotLight),
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 4,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 5,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image_infos[1],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
							write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
		
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}}, 
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}}, //front

		
			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},   
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, //right

		
			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},   
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}, //back

		
			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},   
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, //left

		
			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},   
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}}, //top

		
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},   
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}}, //bottom
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Add models to scene
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{},
		.albedo_color = veekay::vec3{0.4f, 0.4f, 0.4f},
		
		.specular_color = veekay::vec3{0.4f, 0.4f, 0.4f},
		.t_idx = 0,
		.shininess = 1.0f,
	});


	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {2.0f, -0.5f, -1.0f},
			.scale = {0.8f, 0.8f, 0.8f},
			.rotation = {0.0f, M_PI/6.0f, 0.0f},
		},
		.albedo_color = veekay::vec3{0.0f, 1.0f, 0.0f},
		
		.specular_color = veekay::vec3{0.7f, 0.8f, 0.7f},
		.t_idx = 1,
		.shininess = 128.0f,
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.0f, -0.5f, -0.5f},
			.scale = {0.9f, 0.9f, 0.9f},
			.rotation = {0.0f, M_PI/4.0f, 0.0f},
		},
		.albedo_color = veekay::vec3{1.0f, 0.0f, 0.0f},
		
		.specular_color = veekay::vec3{0.8f, 0.7f, 0.7f},
		.t_idx = 1,
		.shininess = 84.0f,
	});	

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.0f, -0.5f, 1.0f},
		},
		.albedo_color = veekay::vec3{0.0f, 0.0f, 1.0f},
		
		.specular_color = veekay::vec3{0.7f, 0.7f, 0.8f},
		.t_idx = 1,
		.shininess = 156.0f,
	});

	point_lights.emplace_back(PointLight{
		.position = veekay::vec3{2.5f, -1.2f, -0.45f},
		.radius = 4.0f,
		.color = veekay::vec3{1.0f, 0.35f, 0.96f},
	});

	point_lights.emplace_back(PointLight{
		.position = veekay::vec3{0.0f, -0.5f, -0.2f},
		.radius = 4.0f,
		.color = veekay::vec3{1.0f, 0.7f, 0.3f},
	});

	point_lights.emplace_back(PointLight{
		.position = veekay::vec3{-2.2f, -1.1f, -0.25f},
		.radius = 4.0f,
		.color = veekay::vec3{0.55f, 1.0f, 0.92f},
	});

	spot_lights.emplace_back(SpotLight{
		.position = camera.position,
		.radius = 6.0f,
		.direction = veekay::vec3::normalized({0.0, 0.5, 0.5}),
		.angle_cos = cos(toRadians(50.0f)),
		.color = veekay::vec3{1.0f, 1.0f, 0.95f},
	});
};

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	delete point_lights_buffer;
    delete spot_lights_buffer;

	vkDestroySampler(device, batman_sampler, nullptr);
	delete batman_texture;

	vkDestroySampler(device, texture_sampler, nullptr);
	delete texture;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	// Настройки освещения и управления
    ImGui::Begin("Controls");
    ImGui::SliderFloat("Ambient:", &ambient_light, 0.0f, 1.0f);
    ImGui::SliderFloat("Sensitivity:", &sens, 0.001f, 1.0f);
    
    // Переключатель день-ночь
    ImGui::Checkbox("Daytime", &is_daytime);
	ImGui::Checkbox("Animate", &animate);
    
    // Кнопки для добавления источников света
    if (ImGui::Button("Add Point Light at Camera")) {
        if (point_lights.size() < max_point_lights) {
            PointLight new_light{
                .position = camera.position,
                .radius = 5.0f,
                .color = {1.0f, 1.0f, 0.8f} // теплый белый свет
            };
            point_lights.push_back(new_light);
        }
    }
    
    if (ImGui::Button("Add Spot Light at Camera")) {
        if (spot_lights.size() < max_spot_lights) {
            SpotLight new_light{
                .position = camera.position,
                .radius = 10.0f,
                .direction = -camera.forward(),
                .angle_cos = cos(toRadians(30.0f)), // 30 градусов
                .color = {1.0f, 1.0f, 1.0f} // белый свет
            };
            spot_lights.push_back(new_light);
        }
    }
    
    // Кнопки для очистки источников света
    if (ImGui::Button("Clear Point Lights")) {
        point_lights.clear();
    }
    
    if (ImGui::Button("Clear Spot Lights")) {
        spot_lights.clear();
    }
    
    ImGui::Text("Point Lights: %zu", point_lights.size());
    ImGui::Text("Spot Lights: %zu", spot_lights.size());
    
    // Настройка точечных источников света
    if (!point_lights.empty()) {
        ImGui::Separator();
        ImGui::Text("Point Lights Settings:");
        
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            
			
            if (ImGui::CollapsingHeader(("Point Light " + std::to_string(i)).c_str())) {
                PointLight& light = point_lights[i];
                
                ImGui::SliderFloat3("Position", &light.position.x, -10.0f, 10.0f);
                ImGui::SliderFloat("Radius", &light.radius, 0.1f, 20.0f);
                ImGui::ColorEdit3("Color", &light.color.x);
                
                if (ImGui::Button("Set to Camera Position")) {
                    light.position = camera.position;
                }
                
                ImGui::SameLine();
                if (ImGui::Button("Remove")) {
                    point_lights.erase(point_lights.begin() + i);
                    ImGui::PopID();
                    break;
                }
            }
            
            ImGui::PopID();
        }
    }
    
    // Настройка прожекторных источников света
    if (!spot_lights.empty()) {
        ImGui::Separator();
        ImGui::Text("Spot Lights Settings:");
        
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i + 100));
            
            if (ImGui::CollapsingHeader(("Spot Light " + std::to_string(i)).c_str())) {
                SpotLight& light = spot_lights[i];
                
                ImGui::SliderFloat3("Position", &light.position.x, -10.0f, 10.0f);
                ImGui::SliderFloat("Radius", &light.radius, 0.1f, 20.0f);
                ImGui::SliderFloat3("Direction", &light.direction.x, -1.0f, 1.0f);
                
                // Нормализуем направление
                if (ImGui::Button("Normalize Direction")) {
                    light.direction = veekay::vec3::normalized(light.direction);
                }
                
                // Угол в градусах для удобства
                float angle_degrees = acos(light.angle_cos) * 180.0f / M_PI;
                if (ImGui::SliderFloat("Angle (degrees)", &angle_degrees, 1.0f, 89.0f)) {
                    light.angle_cos = cos(toRadians(angle_degrees));
                }
                
                ImGui::ColorEdit3("Color", &light.color.x);
                
                if (ImGui::Button("Set to Camera Position")) {
                    light.position = camera.position;
                }
                
                ImGui::SameLine();
                if (ImGui::Button("Set to Camera Direction")) {
                    light.direction = -camera.forward();
                }
                
                ImGui::SameLine();
                if (ImGui::Button("Remove")) {
                    spot_lights.erase(spot_lights.begin() + i);
                    ImGui::PopID();
                    break;
                }
            }
            
            ImGui::PopID();
        }
    }
    
    ImGui::Text("Camera Pos: %.2f, %.2f, %.2f", 
        camera.position.x, camera.position.y, camera.position.z);
    ImGui::Text("Camera Rot: %.2f, %.2f", 
        camera.rotation.y, camera.rotation.x);
    ImGui::End();

    // Управление камерой
	{
		using namespace veekay::input;
		if (toggle_camera) {

			auto move_delta = mouse::cursorDelta();

			camera.rotation.y = fmod(2.0f * M_PI + camera.rotation.y + move_delta.x * sens * 0.02f, 2.0f * M_PI);
			camera.rotation.x = fmod(camera.rotation.x + move_delta.y * sens * 0.02f, 2.0f * M_PI);
			camera.rotation.x = std::clamp(camera.rotation.x, -0.96f * float(M_PI_2), 0.96f * float(M_PI_2));
		}
			
		auto view = camera.view();

		veekay::vec3 right = {0.1f * cos(camera.rotation.y), 0.0f, -0.1f * sin(camera.rotation.y)};
		veekay::vec3 up = {0.0f, -0.1f, 0.0f};
		veekay::vec3 front = {0.1f * sin(camera.rotation.y), 0.0f, 0.1f * cos(camera.rotation.y)};

		if (keyboard::isKeyPressed(keyboard::Key::c)) {
			toggle_camera = !toggle_camera;
			mouse::setCaptured(toggle_camera);
		}
		
		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position += front;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position -= front;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right;

		if (keyboard::isKeyDown(keyboard::Key::q))
			camera.position += up;

		if (keyboard::isKeyDown(keyboard::Key::z))
			camera.position -= up;
	}

	//animations
	if (animate) {
		for (size_t i = 0; i < point_lights.size(); ++i) {
			point_lights[i].position.x += sinf(time + i * M_PI_4) / 300.0f;
			point_lights[i].position.z += cosf(time + i * M_PI_4) / 300.0f;
		}
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			spot_lights[i].direction.x += sinf(time + i * M_PI_4) / 400.0f;
			spot_lights[i].direction.z += cosf(time + i * M_PI_4) / 400.0f;
		}
	}

    // Настройки освещения в зависимости от времени суток
    veekay::vec3 final_sun_direction, final_sun_color;
    float ambient_level = ambient_light;
    
    if (is_daytime) {
        // Дневное освещение
        final_sun_direction = veekay::vec3::normalized({0.3f, 1.0f, 0.15f});
        final_sun_color = {0.8f, 0.8f, 0.7f};
    } else {
        // Ночное освещение
        final_sun_direction = veekay::vec3::normalized({0.15f, 1.0f, 0.3f});
        final_sun_color = {0.2f, 0.2f, 0.25f};
		ambient_level *= 0.4;
    }

    // Обновление uniform буферов
    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .view_position = camera.position,
        .ambient_light_intensity = {ambient_level, ambient_level, ambient_level},
        .sun_light_direction = final_sun_direction,
        .sun_light_color = final_sun_color,
        .point_lights_count = static_cast<uint32_t>(point_lights.size()),
        .spot_lights_count = static_cast<uint32_t>(spot_lights.size())
    };

    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms& uniforms = model_uniforms[i];

        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.albedo_color;
		
        uniforms.specular_color = model.specular_color;
		uniforms.t_idx = model.t_idx;
        uniforms.shininess = model.shininess;
    }

    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    uint8_t* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
        std::memcpy(base + i * aligned_sizeof, &model_uniforms[i], sizeof(ModelUniforms));
    }

    // Копируем данные источников света в буферы
    if (point_lights_buffer) {
        if (!point_lights.empty()) {
            std::memcpy(point_lights_buffer->mapped_region, 
                       point_lights.data(), 
                       point_lights.size() * sizeof(PointLight));
		}
    }
    
    if (spot_lights_buffer) {
        if (!spot_lights.empty()) {
            std::memcpy(spot_lights_buffer->mapped_region, 
                       spot_lights.data(), 
                       spot_lights.size() * sizeof(SpotLight));
        } 
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = uint32_t(i * aligned_sizeof);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	 std::srand(time(nullptr));  
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}