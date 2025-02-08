#include "transferFunction.h"

TransferFunction::TransferFunction()
{
	Node node;
	node.intensity = 0.f;
	node.specular = make_float3(float(rand() % 255) / 255.f, float(rand() % 255) / 255.f, float(rand() % 255) / 255.f);
	node.diffuse = make_float3(float(rand() % 255) / 255.f, float(rand() % 255) / 255.f, float(rand() % 255) / 255.f);
	node.alpha = 0.f;
	node.roughness.x = (rand() % 25) / 25.f;
	addNode(node);
	for (int i = 0; i < totalNodes - 2; i++)
	{
		node.intensity = (rand() % 25) / 25.f;
		node.specular = make_float3(float(rand() % 255) / 255.f, float(rand() % 255) / 255.f, float(rand() % 255) / 255.f);
		node.diffuse = make_float3(float(rand() % 255) / 255.f, float(rand() % 255) / 255.f, float(rand() % 255) / 255.f);
		node.roughness.x = (rand() % 25) / 25.f;
		addNode(node);
	}
	node.intensity = 1.f;
	node.specular = make_float3(float(rand() % 255) / 255.f, float(rand() % 255) / 255.f, float(rand() % 255) / 255.f);
	node.diffuse = make_float3(float(rand() % 255) / 255.f, float(rand() % 255) / 255.f, float(rand() % 255) / 255.f);
	node.alpha = 1.f;
	node.roughness.x = (rand() % 25) / 25.f;
	addNode(node);
}

bool compareByIntensity(const Node& a, const Node& b) {
	return a.intensity < b.intensity;
}

void TransferFunction::addNode(Node node)
{
	Nodes.push_back(node);
	std::sort(Nodes.begin(), Nodes.end(), compareByIntensity);
}

float3 TransferFunction::Value(float f, int colorProperty)
{
	int totalNodes = (int)Nodes.size();
	for (int i = 0; i < totalNodes - 1; i++)
	{
		if (f >= Nodes[i].intensity && f < Nodes[i + 1].intensity)
		{
			float T = (float)(f - Nodes[i].intensity) / (Nodes[i + 1].intensity - Nodes[i].intensity);
			switch (colorProperty) {
			case 0:
				return lerp(Nodes[i].specular, Nodes[i + 1].specular, T);
			case 1:
				return lerp(Nodes[i].diffuse, Nodes[i + 1].diffuse, T);
			case 2:
				return lerp(Nodes[i].roughness, Nodes[i + 1].roughness, T);
			case 3:
				float val = lerp(Nodes[i].alpha, Nodes[i + 1].alpha, T);
				return make_float3(val, val, val);
			}
		}
	}
	return make_float3(0.f, 0.f, 0.f);
}

std::vector<Node> TransferFunction::getNodes()
{
	return Nodes;
}

void TransferFunction::setNodes(std::vector<Node> Nodes_) {
	Nodes = Nodes_;
}
