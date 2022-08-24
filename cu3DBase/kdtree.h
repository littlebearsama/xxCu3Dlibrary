#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <glm/glm.hpp>

class Node{
public :
    //can't use ptr for cuda
    //using NodePtr = std::shared_ptr<Node>;
    //has to be public due to call in __device__ function
    //NodePtr left;
    //NodePtr right;
    //NodePtr parent;

    int left, right, parent;

    glm::vec3 data;
    int axis;

    Node();
	Node(const glm::vec3 &value, int axis);
    Node(const glm::vec3 &value, int left, int right);
    ~Node()= default;
    //int getAxis();



};

using ptsIter = std::vector<glm::vec3>::iterator;

class KDTree{
public:
    KDTree()= default;
    ~KDTree()= default;
    KDTree(std::vector <glm::vec3>& pts,float scale);
	std::vector<Node> getTree()const { return tree; }

private:
    using ptsIter = std::vector<glm::vec3>::iterator;
    void make_tree(ptsIter &begin, ptsIter &end, int axis, int length, int index);
    std::vector<Node> tree;
    float scale;
};


bool sortDimx(const glm::vec3 &pt1, glm::vec3& pt2);
bool sortDimy(const glm::vec3 &pt1, glm::vec3& pt2);
bool sortDimz(const glm::vec3 &pt1, glm::vec3& pt2);