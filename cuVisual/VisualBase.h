#pragma once

#include <glm/glm.hpp>
#include <iostream>
#include <vector>
void getAABBBox(const std::vector<glm::vec3>& cloud, glm::vec3& lefttop, glm::vec3& rightdown);