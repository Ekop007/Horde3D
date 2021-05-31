 [[FX]]


// Buffers
buffer ParticleData;

// Uniforms
float totalParticles = 1000;
float deltaTime = 0;
float4 attractor = {0.0, 0.0, 0.0, 0.0};

// Contexts
OpenGL4
{
	context COMPUTE
	{
		ComputeShader = compile GLSL CS_TILECOMPUTE_GL4;
	}
	
	context AMBIENT
	{
		VertexShader = compile GLSL VS_GENERAL_GL4;
		GeometryShader = compile GLSL GS_TRIANGULATE_GL4;
		PixelShader = compile GLSL FS_AMBIENT_GL4;
		
//		ZWriteEnable = false;
		BlendMode = AddBlended;
	}
}

OpenGLES3
{
	context COMPUTE
	{
		ComputeShader = compile GLSL CS_PARTICLESOLVER_GLES3;
	}
	
	context AMBIENT
	{
		VertexShader = compile GLSL VS_GENERAL_GLES3;
		GeometryShader = compile GLSL GS_TRIANGULATE_GLES3;
		PixelShader = compile GLSL FS_AMBIENT_GLES3;
		
//		ZWriteEnable = false;
		BlendMode = AddBlended;
	}
}

[[CS_TILECOMPUTE_GL4]]
// =================================================================================================

#version 430

struct PointLight {
	vec4 color;
	vec4 position;
	vec4 paddingAndRadius;
};

struct VisibleIndex {
	int index;
};

// Shader storage buffer objects
layout(std430, binding = 0) readonly buffer LightBuffer {
	PointLight data[];
} lightBuffer;

layout(std430, binding = 1) writeonly buffer VisibleLightIndicesBuffer {
	VisibleIndex data[];
} visibleLightIndicesBuffer;

// Uniforms
uniform sampler2D depthMap;
uniform mat4 view;
uniform mat4 projection;
uniform ivec2 screenSize;
uniform int lightCount;

// Shared values between all the threads in the group
shared uint minDepthInt;
shared uint maxDepthInt;
shared uint visibleLightCount;
shared vec4 frustumPlanes[6];
// Shared local storage for visible indices, will be written out to the global buffer at the end
shared int visibleLightIndices[1024];
shared mat4 viewProjection;

// Took some light culling guidance from Dice's deferred renderer
// http://www.dice.se/news/directx-11-rendering-battlefield-3/

#define TILE_SIZE 16
layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;
void main() {
	ivec2 location = ivec2(gl_GlobalInvocationID.xy);
	ivec2 itemID = ivec2(gl_LocalInvocationID.xy);
	ivec2 tileID = ivec2(gl_WorkGroupID.xy);
	ivec2 tileNumber = ivec2(gl_NumWorkGroups.xy);
	uint index = tileID.y * tileNumber.x + tileID.x;

	// Initialize shared global values for depth and light count
	if (gl_LocalInvocationIndex == 0) {
		minDepthInt = 0xFFFFFFFF;
		maxDepthInt = 0;
		visibleLightCount = 0;
		viewProjection = projection * view;
	}

	barrier();

	// Step 1: Calculate the minimum and maximum depth values (from the depth buffer) for this group's tile
	float maxDepth, minDepth;
	vec2 text = vec2(location) / screenSize;
	float depth = texture(depthMap, text).r;
	// Linearize the depth value from depth buffer (must do this because we created it using projection)
	depth = (0.5 * projection[3][2]) / (depth + 0.5 * projection[2][2] - 0.5);

	// Convert depth to uint so we can do atomic min and max comparisons between the threads
	uint depthInt = floatBitsToUint(depth);
	atomicMin(minDepthInt, depthInt);
	atomicMax(maxDepthInt, depthInt);

	barrier();

	// Step 2: One thread should calculate the frustum planes to be used for this tile
	if (gl_LocalInvocationIndex == 0) {
		// Convert the min and max across the entire tile back to float
		minDepth = uintBitsToFloat(minDepthInt);
		maxDepth = uintBitsToFloat(maxDepthInt);

		// Steps based on tile sale
		vec2 negativeStep = (2.0 * vec2(tileID)) / vec2(tileNumber);
		vec2 positiveStep = (2.0 * vec2(tileID + ivec2(1, 1))) / vec2(tileNumber);

		// Set up starting values for planes using steps and min and max z values
		frustumPlanes[0] = vec4(1.0, 0.0, 0.0, 1.0 - negativeStep.x); // Left
		frustumPlanes[1] = vec4(-1.0, 0.0, 0.0, -1.0 + positiveStep.x); // Right
		frustumPlanes[2] = vec4(0.0, 1.0, 0.0, 1.0 - negativeStep.y); // Bottom
		frustumPlanes[3] = vec4(0.0, -1.0, 0.0, -1.0 + positiveStep.y); // Top
		frustumPlanes[4] = vec4(0.0, 0.0, -1.0, -minDepth); // Near
		frustumPlanes[5] = vec4(0.0, 0.0, 1.0, maxDepth); // Far

		// Transform the first four planes
		for (uint i = 0; i < 4; i++) {
			frustumPlanes[i] *= viewProjection;
			frustumPlanes[i] /= length(frustumPlanes[i].xyz);
		}

		// Transform the depth planes
		frustumPlanes[4] *= view;
		frustumPlanes[4] /= length(frustumPlanes[4].xyz);
		frustumPlanes[5] *= view;
		frustumPlanes[5] /= length(frustumPlanes[5].xyz);
	}

	barrier();

	// Step 3: Cull lights.
	// Parallelize the threads against the lights now.
	// Can handle 256 simultaniously. Anymore lights than that and additional passes are performed
	uint threadCount = TILE_SIZE * TILE_SIZE;
	uint passCount = (lightCount + threadCount - 1) / threadCount;
	for (uint i = 0; i < passCount; i++) {
		// Get the lightIndex to test for this thread / pass. If the index is >= light count, then this thread can stop testing lights
		uint lightIndex = i * threadCount + gl_LocalInvocationIndex;
		if (lightIndex >= lightCount) {
			break;
		}

		vec4 position = lightBuffer.data[lightIndex].position;
		float radius = lightBuffer.data[lightIndex].paddingAndRadius.w;

		// We check if the light exists in our frustum
		float distance = 0.0;
		for (uint j = 0; j < 6; j++) {
			distance = dot(position, frustumPlanes[j]) + radius;

			// If one of the tests fails, then there is no intersection
			if (distance <= 0.0) {
				break;
			}
		}

		// If greater than zero, then it is a visible light
		if (distance > 0.0) {
			// Add index to the shared array of visible indices
			uint offset = atomicAdd(visibleLightCount, 1);
			visibleLightIndices[offset] = int(lightIndex);
		}
	}

	barrier();

	// One thread should fill the global light buffer
	if (gl_LocalInvocationIndex == 0) {
		uint offset = index * 1024; // Determine bosition in global buffer
		for (uint i = 0; i < visibleLightCount; i++) {
			visibleLightIndicesBuffer.data[offset + i].index = visibleLightIndices[i];
		}

		if (visibleLightCount != 1024) {
			// Unless we have totally filled the entire array, mark it's end with -1
			// Final shader step will use this to determine where to stop (without having to pass the light count)
			visibleLightIndicesBuffer.data[offset + visibleLightCount].index = -1;
		}
	}
}


[[VS_GENERAL_GL4]]
// =================================================================================================
#include "shaders/utilityLib/vertCommon.glsl"

layout (location = 0) in vec4 partPosition;
layout (location = 1) in vec4 partVelocity;

//uniform mat4 projMat;
uniform mat4 viewProjMat;

out vec3 partColor;

void main( void )
{
	vec4 pos = calcWorldPos( partPosition );
	
	float speed = length( partVelocity.xyz );
	partColor = mix(vec3(0.1, 0.5, 1.0), vec3(1.0, 0.5, 0.1), speed * 0.1 );
	
	gl_Position = pos;
}


[[GS_TRIANGULATE_GL4]]
// =================================================================================================
uniform mat4 viewMat;
uniform mat4 viewProjMat;
uniform vec3 viewerPos;
 
layout(points) in;
layout (triangle_strip) out;
layout(max_vertices = 4) out;
 
in vec3 partColor[];
 
out vec2 vertTexCoords;
out vec3 color;
 
void main()
{
// create billboards from points
	color = partColor[0];
	
	float particle_size = 0.1;
	
	vec3 up = vec3( viewMat[0][1], viewMat[1][1], viewMat[2][1] );
				 
	vec3 P = gl_in[0].gl_Position.xyz;
	
	vec3 toCamera = normalize( viewerPos - P );
//	vec3 up = vec3(0.0, 1.0, 0.0);
	vec3 right = cross( toCamera, up );
	
	vec3 va = P - (right + up) * particle_size;
	gl_Position = viewProjMat * vec4(va, 1.0);
	vertTexCoords = vec2(0.0, 0.0);
	EmitVertex();  
	  
	vec3 vb = P - (right - up) * particle_size;
	gl_Position = viewProjMat * vec4(vb, 1.0);
	vertTexCoords = vec2(0.0, 1.0);
	EmitVertex();  

	vec3 vd = P + (right - up) * particle_size;
	gl_Position = viewProjMat * vec4(vd, 1.0);
	vertTexCoords = vec2(1.0, 0.0);
	EmitVertex();  

	vec3 vc = P + (right + up) * particle_size;
	gl_Position = viewProjMat * vec4(vc, 1.0);
	vertTexCoords = vec2(1.0, 1.0);
	EmitVertex();  
	  
	EndPrimitive();
    
}


[[FS_AMBIENT_GL4]]
// =================================================================================================
uniform sampler2D albedoMap;

in vec2 vertTexCoords;
in vec3 color;

out vec4 fragColor;

void main()
{
	vec4 texColor = texture( albedoMap, vec2( vertTexCoords.s, -vertTexCoords.t ) );
	if ( texColor.a < 0.1 ) discard;
	
	fragColor = vec4( color * texColor.xyz, texColor.a );
}


// =================================================================================================
// GLES 3
// =================================================================================================

[[CS_PARTICLESOLVER_GLES3]]
// =================================================================================================

uniform float totalParticles;
uniform float deltaTime;
uniform vec4 attractor;

struct Particle
{
	vec4 position;
	vec4 velocity;
};

const uint maxThreadsInGroup = 128u;
 
layout (std430, binding = 1) buffer ParticleData
{ 
	Particle particlesBuf[];
} data;

layout(local_size_x = 16, local_size_y = 8) in;

/////////////////////
vec3 calculate(vec3 anchor, vec3 position)
{
	vec3 direction = anchor - position;
	float dist = length(direction);
	direction /= dist;

	return direction * max(0.01, (1.0 / (dist * dist)));
}

void main()
{
	uint index = uint(gl_WorkGroupID.x * maxThreadsInGroup) + uint(gl_WorkGroupID.y * gl_NumWorkGroups.x * maxThreadsInGroup) + gl_LocalInvocationIndex; 
	uint tp = uint(totalParticles); 
	if( index >= tp ) 
		return;

	Particle particle = data.particlesBuf[ index ];

	vec4 position = particle.position; 
	vec4 velocity = particle.velocity; 

	velocity += vec4( calculate( attractor.xyz, position.xyz ), 0 );
	velocity += vec4( calculate( -attractor.xyz, position.xyz ), 0 ) ;
	
	particle.position = position + velocity * deltaTime;
	particle.velocity = velocity;

	data.particlesBuf[index] = particle;
}

[[VS_GENERAL_GLES3]]
// =================================================================================================
#include "shaders/utilityLib/vertCommon.glsl"

layout (location = 0) in vec4 partPosition;
layout (location = 1) in vec4 partVelocity;

//uniform mat4 projMat;
uniform mat4 viewProjMat;

out vec3 partColor;

void main( void )
{
	vec4 pos = calcWorldPos( partPosition );
	
	float speed = length( partVelocity.xyz );
	partColor = mix(vec3(0.1, 0.5, 1.0), vec3(1.0, 0.5, 0.1), speed * 0.1 );
	
	gl_Position = pos;
}


[[GS_TRIANGULATE_GLES3]]
// =================================================================================================
uniform mat4 viewMat;
uniform mat4 viewProjMat;
uniform vec3 viewerPos;
 
layout(points) in;
layout (triangle_strip) out;
layout(max_vertices = 4) out;
 
in vec3 partColor[];
 
out vec2 vertTexCoords;
out vec3 color;
 
void main()
{
// create billboards from points
	color = partColor[0];
	
	float particle_size = 0.1;
	
	vec3 up = vec3( viewMat[0][1], viewMat[1][1], viewMat[2][1] );
				 
	vec3 P = gl_in[0].gl_Position.xyz;
	
	vec3 toCamera = normalize( viewerPos - P );
//	vec3 up = vec3(0.0, 1.0, 0.0);
	vec3 right = cross( toCamera, up );
	
	vec3 va = P - (right + up) * particle_size;
	gl_Position = viewProjMat * vec4(va, 1.0);
	vertTexCoords = vec2(0.0, 0.0);
	EmitVertex();  
	  
	vec3 vb = P - (right - up) * particle_size;
	gl_Position = viewProjMat * vec4(vb, 1.0);
	vertTexCoords = vec2(0.0, 1.0);
	EmitVertex();  

	vec3 vd = P + (right - up) * particle_size;
	gl_Position = viewProjMat * vec4(vd, 1.0);
	vertTexCoords = vec2(1.0, 0.0);
	EmitVertex();  

	vec3 vc = P + (right + up) * particle_size;
	gl_Position = viewProjMat * vec4(vc, 1.0);
	vertTexCoords = vec2(1.0, 1.0);
	EmitVertex();  
	  
	EndPrimitive();
    
}


[[FS_AMBIENT_GLES3]]
// =================================================================================================
uniform sampler2D albedoMap;

in vec2 vertTexCoords;
in vec3 color;

out vec4 fragColor;

void main()
{
	vec4 texColor = texture( albedoMap, vec2( vertTexCoords.s, -vertTexCoords.t ) );
	if ( texColor.a < 0.1 ) discard;
	
	fragColor = vec4( color * texColor.xyz, texColor.a );
}
