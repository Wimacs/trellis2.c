#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec3 v_position;

void main() {
    v_position = in_position;
    gl_Position = vec4(in_uv.x * 2.0 - 1.0, in_uv.y * 2.0 - 1.0, 0.0, 1.0);
}
