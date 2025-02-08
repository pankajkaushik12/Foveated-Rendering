#pragma once

#include "cutil_math.h"
#include <vector>
#include <algorithm>

enum colorProperty {
	Specular,
	Diffuse,
	Roughness,
	TF,
	NUM_COLOR_PROPERTIES
};

struct Node {
	float intensity;
	float3 specular;
	float3 diffuse;
	float3 roughness;
	float alpha;
};

class TransferFunction
{
private:
	int totalNodes = 4;
	std::vector<Node> Nodes;
public:
	TransferFunction();
	~TransferFunction() {
		Nodes.clear();
	}
	void addNode(Node node);
	void setNodes(std::vector<Node> nodes);
	float3 Value(float f, int colorProperty);

	std::vector<Node> getNodes();
};