#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    auto nx = tensor + tensor;
    std::cout << nx << std::endl;
}
