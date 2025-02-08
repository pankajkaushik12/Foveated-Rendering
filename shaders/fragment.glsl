#version 330 core

uniform bool foveatedRendering;
uniform vec2 eyePos;
uniform vec2 focalLength;
float circleThickness = 0.001;  // The thickness of the circle

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D textureSampler;

vec2 rectangularFoveatedInverseMapping(vec2 pixel) {

    float fx = focalLength.x;                                              //optixLaunchParams.camera.fovx;
    float fy = focalLength.y;                                                 //optixLaunchParams.camera.fovy;

    vec2 cursorPos = eyePos;

    float maxDxPos = 1.0 - cursorPos[0];
    float maxDyPos = 1.0 - cursorPos[1];
    float maxDxNeg = cursorPos[0];
    float maxDyNeg = cursorPos[1];

    float norDxPos = fx * maxDxPos / (fx + maxDxPos);
    float norDyPos = fy * maxDyPos / (fy + maxDyPos);
    float norDxNeg = fx * maxDxNeg / (fx + maxDxNeg);
    float norDyNeg = fy * maxDyNeg / (fy + maxDyNeg);

    vec2 tc = pixel - cursorPos;
    float x, y;

    if (tc.x >= 0) {
        x = fx * tc.x / (fx + tc.x); //>0
        x = x / norDxPos;
        x = x * maxDxPos + cursorPos.x;
    }
    else {
        x = fx * tc.x / (fx - tc.x); //<0
        x = x / norDxNeg;
        x = x * maxDxNeg + cursorPos.x;
    }

    if (tc.y >= 0) {
        y = fy * tc.y / (fy + tc.y);
        y = y / norDyPos;
        y = y * maxDyPos + cursorPos.y;
    }
    else {
        y = fy * tc.y / (fy - tc.y);
        y = y / norDyNeg;
        y = y * maxDyNeg + cursorPos.y;
    }
    return vec2(x, y);
}

void main() {
    if (foveatedRendering) {
        float dist = length(TexCoord - eyePos);
        float region1 = length(focalLength) / 3.f;
        float region2 = length(focalLength) / 1.5f;
        FragColor = texture(textureSampler, rectangularFoveatedInverseMapping(TexCoord));
        if (dist > region1 - circleThickness && dist < region1 + circleThickness){
            FragColor = mix(FragColor, vec4(1, 1, 0, 1), 0.5);
        }
        if (dist > region2 - circleThickness && dist < region2 + circleThickness){
            FragColor = mix(FragColor, vec4(1, 0, 1, 1), 0.5);
        }
        if (dist < 0.005){
            FragColor = mix(FragColor, vec4(1, 0, 0, 1), 0.5);
        }
	}
	else
	{
		FragColor = texture(textureSampler, TexCoord);
	}
}