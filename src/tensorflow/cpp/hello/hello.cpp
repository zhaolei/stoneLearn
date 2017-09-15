#include <iostream>
#include <tensorflow/c/c_api.h>

using namespace std;

int main() {
    //printf("Hello from TensorFlow C library version %s\n", TF_Version());
    cout<<"Hello from TensorFlow C library version "<<endl<<TF_Version()<<endl;
    return 0;
}
